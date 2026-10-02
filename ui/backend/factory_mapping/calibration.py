"""Independent, persistent calibration datasets and fixed upstream orchestration."""
import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
import yaml
from . import commands
from .storage import atomic_json,read_json,now,size
from .calibration_data import require_intrinsics,parse_result,transform,atomic_yaml
from .camera_config import config_path

async def ensure_display():
    if not os.environ.get('DISPLAY'):raise ValueError('Native calibration GUI requires a server DISPLAY; copy the dataset to a desktop workstation')
    code='import ctypes; x=ctypes.CDLL("libX11.so.6"); x.XOpenDisplay.restype=ctypes.c_void_p; d=x.XOpenDisplay(None); raise SystemExit(0 if d else 1)'
    p=await asyncio.create_subprocess_exec(sys.executable,'-c',code,stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
    try:rc=await asyncio.wait_for(p.wait(),5)
    except asyncio.TimeoutError:
        p.kill();await p.wait();raise ValueError('Server DISPLAY did not respond; open a working desktop session')
    if rc:raise ValueError('Cannot connect to server DISPLAY; check X11 display permissions and desktop session')


STATES=('CREATED','CAPTURING','CAPTURED','PREPROCESSED','INITIALIZED','CALIBRATED','IMPORTED','VALIDATED')


def hashes(path):
    result={}
    for f in sorted(path.rglob('*')):
        if f.is_symlink():raise ValueError('Calibration data cannot contain symlinks')
        if not f.is_file():continue
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        result[str(f.relative_to(path))]=h.hexdigest()
    return result


def bag_statistics(path,config):
    try:
        info=yaml.safe_load((path/'metadata.yaml').read_text())['rosbag2_bagfile_information']
        seconds=info['duration']['nanoseconds']/1e9
        entries={x['topic_metadata']['name']:x for x in info['topics_with_message_count']}
        stats={}
        for key,kind in [('points_topic','PointCloud2'),('image_topic','Image'),('camera_info_topic','CameraInfo')]:
            topic=config['sensor']['points_topic'] if key=='points_topic' else config['camera'][key]
            entry=entries.get(topic)
            if not entry or entry['message_count']<=0 or entry['topic_metadata']['type']!='sensor_msgs/msg/'+kind:raise ValueError('Capture missing usable '+topic+' ('+kind+')')
            stats[key]={'topic':topic,'count':entry['message_count'],'hz':entry['message_count']/seconds if seconds>0 else 0}
        rate=stats['image_topic']['hz'];expected=config['camera']['expected_hz']
        if not expected*.7<=rate<=expected*1.3:raise ValueError(f'Camera capture rate {rate:.2f} Hz differs from expected {expected} Hz; record a longer static capture and check USB3/exposure')
        return dict(duration=seconds,topics=stats)
    except (OSError,KeyError,TypeError,yaml.YAMLError) as e:raise ValueError('Invalid or unfinished calibration bag metadata') from e


def tool_command(stage,work,config,intr,inputs=None):
    if stage not in ('preprocess','initial_guess_manual','calibrate'):raise ValueError('Unsupported calibration command')
    args=['ros2','run','direct_visual_lidar_calibration',stage]
    if stage=='preprocess':
        args += [str(inputs),str(work),'--points_topic',config['sensor']['points_topic'],'--image_topic',config['camera']['image_topic'],'--camera_info_topic',config['camera']['camera_info_topic'],'--camera_model',intr['model'],'--camera_intrinsics',','.join(map(str,intr['intrinsics'])),'--camera_distortion_coeffs',','.join(map(str,intr['distortion'])),'--intensity_channel','intensity']
    else:
        args.append(str(work))
        if stage=='calibrate':args+=['--auto_quit']
    return args


class Calibrations:
    def __init__(self,service):
        self.s=service;self.root=service.root;self.base=self.root/'data/calibrations';self.base.mkdir(parents=True,exist_ok=True);self.active=None
        for p in self.base.iterdir():
            if not p.is_dir() or p.is_symlink():continue
            m=read_json(p/'metadata.json',{})
            if m.get('mock',False)!=service.mock:continue
            if m.get('state')=='CAPTURING':
                m.update(state=m.get('before_capture','CREATED'),error='Capture interrupted by backend restart; preserve and inspect its bag')
                atomic_json(p/'metadata.json',m)
                for f in (p/'captures').glob('*/metadata.json'):
                    cap=read_json(f,{})
                    if cap.get('state')=='CAPTURING':cap.update(state='INTERRUPTED');atomic_json(f,cap)
            for f in (p/'jobs').glob('*/job.json'):
                job=read_json(f,{})
                if job.get('state')=='running':job.update(state='interrupted');atomic_json(f,job)
    def get(self,cid):
        if not re.fullmatch(r'cal_[a-f0-9]{16}',cid):raise ValueError('Invalid calibration ID')
        p=self.base/cid
        if not p.is_dir() or p.is_symlink() or p.resolve().parent!=self.base.resolve():raise ValueError('Calibration not found')
        if read_json(p/'metadata.json',{}).get('mock',False)!=self.s.mock:raise ValueError('Calibration belongs to another data mode')
        return p
    def update(self,p,**fields):
        m=read_json(p/'metadata.json');m.update(fields);atomic_json(p/'metadata.json',m);return m
    def list(self):
        return [read_json(p/'metadata.json') for p in sorted(self.base.iterdir()) if p.is_dir() and not p.is_symlink() and read_json(p/'metadata.json',{}).get('mock',False)==self.s.mock and (p/'metadata.json').is_file()]
    def detail(self,cid):
        p=self.get(cid);m=read_json(p/'metadata.json');m['captures']=[read_json(f) for f in sorted((p/'captures').glob('*/metadata.json'))];m['jobs']=[read_json(f) for f in sorted((p/'jobs').glob('*/job.json'))];return m
    def create(self,name):
        c=self.s.config
        if not c['system']['camera']['enabled']:raise ValueError('Enable camera acquisition before creating a calibration dataset')
        intr=require_intrinsics(self.root,c['camera']);cid='cal_'+uuid.uuid4().hex[:16];p=self.base/cid;p.mkdir()
        for d in ('captures','jobs','result','validation'):(p/d).mkdir()
        shutil.copytree(self.root/'config',p/'config_snapshot');atomic_json(p/'active_config.json',c)
        m=dict(id=cid,name=name,state='CREATED',mock=self.s.mock,created_at=now(),intrinsics=intr,config=c,validated=False)
        atomic_json(p/'metadata.json',m);return m
    def idle(self):
        if any(self.s.pm.active(k) for k in ('calibration_record','calibration_tool','recording','glim','offline','export','tool')):raise ValueError('Finish active recording, mapping or calibration work first')
    def verify_captures(self,p):
        captures=[]
        for cap in sorted((p/'captures').iterdir()):
            meta=read_json(cap/'metadata.json',{})
            if meta.get('state')!='CAPTURED':continue
            if hashes(cap/'raw_bag')!=meta['raw_hashes']:raise ValueError('Finalized capture changed: '+cap.name)
            captures.append(cap)
        if not captures:raise ValueError('At least one accepted static capture is required')
        return captures
    async def capture_start(self,cid):
        self.idle();p=self.get(cid);m=read_json(p/'metadata.json');c=m['config']
        if m['state'] not in ('CREATED','CAPTURED'):raise ValueError('Dataset is sealed for processing; create another dataset for additional captures')
        if c['camera']!=self.s.config.get('camera') or c['sensor']!=self.s.config['sensor']:raise ValueError('Acquisition configuration changed; create a new calibration dataset')
        if require_intrinsics(self.root,c['camera'])['sha256']!=m['intrinsics']['sha256']:raise ValueError('Intrinsics changed; create a new calibration dataset')
        camera=self.s.camera_health()
        if not self.s.pm.active('driver') or (not self.s.mock and self.s.health().get('lidar',{}).get('state')!='healthy'):raise ValueError('Calibration needs a healthy LiDAR stream (IMU is not required)')
        if not camera['healthy'] or not camera['camera_info_valid']:raise ValueError('Calibration needs healthy camera frames and valid CameraInfo')
        if shutil.disk_usage(self.root).free<self.s.config['system']['storage']['minimum_free_gb']*1e9:raise ValueError('Insufficient disk space')
        index=1
        while (p/'captures'/f'capture_{index:03d}').exists():index+=1
        cap=p/'captures'/f'capture_{index:03d}';cap.mkdir();(cap/'config_snapshot').mkdir()
        meta=dict(id=cap.name,state='CAPTURING',started_at=now(),mock=self.s.mock,topics=commands.acquisition_topics(c,True))
        atomic_json(cap/'metadata.json',meta);self.update(p,state='CAPTURING',before_capture=m['state'],error=None);self.active=(p,cap)
        async def done(item):
            try:
                if item['state']!='completed':raise ValueError('Capture did not finish cleanly; raw files retained')
                if self.s.mock:
                    stats=dict(mock=True,topics={k:{'topic':v,'count':1,'hz':None} for k,v in meta['topics'].items()})
                else:stats=await asyncio.to_thread(bag_statistics,cap/'raw_bag',c)
                meta.update(state='CAPTURED',statistics=stats,raw_hashes=await asyncio.to_thread(hashes,cap/'raw_bag'))
                self.update(p,state='CAPTURED',error=None)
            except Exception as e:
                meta.update(state='REJECTED',error=str(e));self.update(p,state=m['state'],error=str(e));self.s.errors.append(str(e))
            finally:
                meta['ended_at']=now();atomic_json(cap/'metadata.json',meta);self.active=None
        try:await self.s.start_process('calibration_record',commands.record(cap,c,True),cap/'capture.log',cap/'raw_bag',done)
        except Exception as e:
            meta.update(state='REJECTED',error=str(e));atomic_json(cap/'metadata.json',meta);self.update(p,state=m['state'],error=str(e));self.active=None;raise
        return meta
    async def capture_stop(self,cid):
        self.get(cid)
        if not self.active or self.active[0].name!=cid:raise ValueError('No active capture for this calibration')
        await self.s.pm.stop('calibration_record',self.s.config['system']['shutdown']['recording_timeout'])
        return self.detail(cid)
    async def run(self,cid,stage):
        self.idle();p=self.get(cid);m=read_json(p/'metadata.json')
        transitions={'preprocess':('CAPTURED','PREPROCESSED'),'initial_guess_manual':('PREPROCESSED','INITIALIZED'),'calibrate':('INITIALIZED','CALIBRATED')}
        if stage not in transitions:raise ValueError('Only preprocess, manual initial guess and NID calibration are supported')
        before,after=transitions[stage]
        if m['state']!=before:raise ValueError(f'{stage} requires {before}; current state is {m["state"]}')
        if self.s.mock:raise ValueError('Mock datasets cannot produce genuine calibration results')
        if stage!='preprocess':await ensure_display()
        exe=self.root/'ros2_ws/install/direct_visual_lidar_calibration/lib/direct_visual_lidar_calibration'/stage
        if not exe.is_file():raise ValueError('Install the optional workstation calibrator with scripts/install_calibration.sh')
        captures=await asyncio.to_thread(self.verify_captures,p)
        needed=sum(size(cap/'raw_bag') for cap in captures) if stage=='preprocess' else size(p/m['work'])
        if shutil.disk_usage(self.root).free<needed+self.s.config['system']['storage']['minimum_free_gb']*1e9:raise ValueError('Insufficient space for calibration working copies')
        job_id='job_'+uuid.uuid4().hex[:12];job=p/'jobs'/job_id;job.mkdir();work=job/'work'
        if stage=='preprocess':
            inputs=job/'input_bags';inputs.mkdir()
            # External preprocess reads separate copies, never capture originals.
            for cap in captures:await asyncio.to_thread(shutil.copytree,cap/'raw_bag',inputs/cap.name)
        else:
            inputs=None;await asyncio.to_thread(shutil.copytree,p/m['work'],work)
        j=dict(id=job_id,stage=stage,state='running',started_at=now(),work=str(work.relative_to(p)));atomic_json(job/'job.json',j)
        async def done(item):
            try:
                if item['state']!='completed':raise ValueError('Calibration tool failed or was cancelled')
                obj=read_json(work/'calib.json')
                if not obj:raise ValueError('Tool did not write calib.json; save manually before closing the GUI')
                if stage=='preprocess':
                    names=obj.get('meta',{}).get('bag_names',[])
                    if sorted(names)!=sorted(cap.name for cap in captures) or any(not (work/(name+'.ply')).is_file() or not (work/(name+'.png')).is_file() for name in names):raise ValueError('Preprocess did not produce all image/cloud pairs')
                elif stage=='initial_guess_manual':transform(obj.get('results',{}).get('init_T_lidar_camera'))
                else:parse_result(work/'calib.json',m['intrinsics'])
                await asyncio.to_thread(self.verify_captures,p)
                self.update(p,state=after,work=str(work.relative_to(p)),error=None)
                j['state']='completed'
            except Exception as e:j.update(state='failed',error=str(e));self.update(p,error=str(e));self.s.errors.append(str(e))
            finally:j['ended_at']=now();atomic_json(job/'job.json',j)
        try:await self.s.start_process('calibration_tool',tool_command(stage,work,m['config'],m['intrinsics'],inputs),job/'tool.log',done=done)
        except Exception as e:j.update(state='failed',error=str(e));atomic_json(job/'job.json',j);raise
        return j
    def import_result(self,cid):
        self.idle();p=self.get(cid);m=read_json(p/'metadata.json')
        if self.s.mock:raise ValueError('Mock results cannot be imported as real calibration')
        if m['state']!='CALIBRATED':raise ValueError('A completed calibration is required before import')
        if any(self.s.pm.active(k) for k in ('camera','driver')):raise ValueError('Stop acquisition before updating the active extrinsic calibration')
        if self.s.config['camera']!=m['config']['camera']:raise ValueError('Current camera configuration differs from calibration snapshot')
        if require_intrinsics(self.root,self.s.config['camera'])['sha256']!=m['intrinsics']['sha256']:raise ValueError('Current intrinsics differ from calibration snapshot')
        captures=self.verify_captures(p)
        src=p/m['work']/'calib.json';pose=parse_result(src,m['intrinsics']);dst=p/'result/calib.json'
        if dst.exists():raise ValueError('Calibration result is already preserved')
        shutil.copy2(src,dst)
        c=m['config']['camera'];result=dict(version=1,camera_name=c['camera_name'],camera_frame_id=c['frame_id'],lidar_frame_id=m['config']['sensor']['frame_id'],intrinsics_file=c['intrinsics_file'],intrinsics_sha256=m['intrinsics']['sha256'],camera_model=m['intrinsics']['model'],intrinsics=m['intrinsics']['intrinsics'],distortion=m['intrinsics']['distortion'],width=c['width'],height=c['height'],T_lidar_camera=pose,transform_convention='p_lidar = T_lidar_camera * p_camera',time_offset_sec=c['time_offset_sec'],source_calibration_id=cid,source_result=str(dst.relative_to(self.root)),calibrated=True,validated=False)
        target=config_path(self.root,c['extrinsics_file'])
        if target.exists():
            history=target.parent/'history';history.mkdir(exist_ok=True);shutil.copy2(target,history/(uuid.uuid4().hex+'_'+target.name))
        atomic_yaml(p/'result/lidar_camera.yaml',result);atomic_yaml(target,result)
        atomic_json(p/'validation/inputs.json',dict(result,calibration_snapshot=str((p/'config_snapshot').relative_to(self.root)),capture_ids=[cap.name for cap in captures],validation_required='Independent reprojection/overlay and measured residual review; convergence is not validation'))
        return self.update(p,state='IMPORTED',imported_at=now(),validated=False)
    def validate(self,cid,notes):
        self.idle();p=self.get(cid);m=read_json(p/'metadata.json')
        if m['state']!='IMPORTED':raise ValueError('Import before recording independent validation')
        # Explicit human evidence, never inferred from optimizer convergence.
        if len(notes.strip())<20:raise ValueError('Describe independent overlay/residual checks and acceptance criteria (at least 20 characters)')
        atomic_json(p/'validation/review.json',dict(reviewed_at=now(),notes=notes,source='operator_review'))
        result=yaml.safe_load((p/'result/lidar_camera.yaml').read_text());result['validated']=True;atomic_yaml(p/'result/lidar_camera.yaml',result)
        target=config_path(self.root,m['config']['camera']['extrinsics_file'])
        current=yaml.safe_load(target.read_text()) if target.exists() else {}
        if current.get('source_calibration_id')==cid:current['validated']=True;atomic_yaml(target,current)
        return self.update(p,state='VALIDATED',validated=True)
