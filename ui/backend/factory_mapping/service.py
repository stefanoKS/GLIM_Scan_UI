import asyncio, copy, json, os, shutil, sys, time, uuid
from datetime import datetime
from pathlib import Path
from . import commands, glim_tools
from .config import ROOT, PRESETS, load, validate_sensor, wired_connection
from .storage import Sessions, atomic_json, read_json, now, size
from .processes import ProcessManager
from .health import network, system_status
from .calibration_data import camera_metadata, intrinsics_status
from .camera import check_camera_dependencies, detect_camera

class Service:
    def __init__(self, root=ROOT, mock=False):
        self.root=root; self.mock=mock; self.config=load(root); self.sessions=Sessions(root,mock)
        self.pm=ProcessManager(root/'.state'); self.active=None; self.lock=asyncio.Lock(); self.errors=list(self.pm.recovery); self.net={}; self.closed=False
        self.intrinsic_samples=[]
        self.record_started=None; self.rate_baseline={}; self.live_preset=None
        from .calibration import Calibrations
        self.calibrations=Calibrations(self)
        for m in self.sessions.list():
            p=self.sessions.get(m['id'])
            if m['state'] in ('recording','mapping','stopping'):
                self.sessions.update(p,state='interrupted',recovery_note='Backend stopped unexpectedly. Inspect bag metadata before processing; raw data was preserved.')
            for f in (p/'processing').glob('*/job.json'):
                job=read_json(f,{})
                if job.get('state')=='running': job.update(state='interrupted'); atomic_json(f,job)
        from .capture import CaptureController
        self.capture=CaptureController(self)
    def event(self,kind,**fields):
        entry=dict(time=now(),event=kind,**fields)
        path=self.root/'.state/application.jsonl'
        with path.open('a') as f: f.write(json.dumps(entry)+'\n')
        if self.active:
            with (self.active/'logs/application.jsonl').open('a') as f: f.write(json.dumps(entry)+'\n')
    def health(self):
        if self.mock:
            active=self.pm.active('driver')
            return {k:dict(state='mock' if active else 'stopped',hz=(10 if k=='lidar' else 200) if active else 0,stamp=time.time(),count=0) for k in ('lidar','imu')}
        obj=read_json(self.root/'.state/health.json',{})
        if time.time()-obj.get('updated_at',0)>3:
            return {k:dict(state='monitor_stale',hz=0) for k in ('lidar','imu')}
        return obj
    def status(self):
        processes=self.pm.view(); h=self.health(); h['camera']=self.camera_health()
        detection={'camera':detect_camera(self.config.get('camera',{})), 'mid360':dict(detected=self.net.get('state')=='reachable',basis='Configured IP responds to ICMP; stream health is separate')}
        if self.mock: detection={k:dict(detected=False,basis='MOCK: physical detection is not performed') for k in detection}
        return dict(capture=self.capture.view(),detection=detection,mock=self.mock,glim_available=self.glim_available(),live_preset=self.live_preset,system=system_status(self.root),config=self.config,camera_calibration=self.camera_calibration(),network=self.net,health=h,processes=processes,active_session=self.active.name if self.active else None,recording_elapsed=time.time()-self.record_started if self.record_started else 0,bag_size_bytes=size(self.active/'raw_bag') if self.active else 0,loop_detection='UNAVAILABLE' if self.config['system']['loop_closure']['enabled'] else 'OFF',errors=self.errors[-15:])
    async def start_process(self,key,args,log,out=None,done=None):
        if self.mock:
            args=[sys.executable,'-m','factory_mapping.mock_worker',key,str(out or self.root/'.state/mock')]
        async def finished(item):
            if done: await done(item)
            self.event('process_exit',role=key,state=item['state'],returncode=item['returncode'],forced=item['forced'])
            if item['state']=='failed':
                self.errors.append(f"{key} exited with code {item['returncode']}; inspect {log}. Restart the affected process after correcting the cause.")
        result=await self.pm.start(key,args,log,commands.ros_env(self.config, offline=key in ('offline','export','tool','calibration_tool')),finished)
        self.event('process_start',role=key,pid=result['pid']); return result
    async def start_driver(self):
        if not self.mock:
            self.config['sensor']['interface'],self.config['sensor']['host_ip']=wired_connection(self.config['sensor'])
            self.net=await network(self.config['sensor'])
            if self.net['state']!='reachable': raise ValueError(self.net['action'])
        result=await self.start_process('driver',commands.driver(self.root,self.config),self.root/'.state/driver.log')
        try:
            if not self.mock:
                for role in ('monitor','preview'):
                    await self.start_process(role,[sys.executable,'-c',f'from factory_mapping.ros_nodes import {role}; {role}()'],self.root/f'.state/{role}.log')
        except Exception:
            for role in ('preview','monitor','driver'): await self.pm.stop(role)
            raise
        return result
    async def stop_driver(self):
        if any(self.pm.active(k) for k in ('recording','glim','calibration_record')): raise ValueError('Stop recording and GLIM before stopping the sensor')
        for k in ('preview','monitor','driver'): await self.pm.stop(k,20)
    def require_health(self):
        if not self.pm.active('driver'): raise ValueError('Start Mid-360 first')
        if not self.mock and not all(self.health().get(k,{}).get('state')=='healthy' for k in ('lidar','imu')): raise ValueError('LiDAR and IMU must both have healthy message rates; run diagnostics')
        if self.config['system']['camera']['enabled'] and self.config['camera']['required_for_mapping'] and not self.camera_health()['healthy']: raise ValueError('Required camera frames are not healthy; start camera and check rates/geometry')
        if shutil.disk_usage(self.root).free<self.config['system']['storage']['minimum_free_gb']*1e9: raise ValueError('Insufficient free disk space')
    async def start_preview_sensors(self):
        if self.mock: return
        async with self.lock:
            if not self.pm.active('driver'):
                try: await self.start_driver()
                except Exception as error: self.errors.append('LiDAR preview unavailable at startup: '+str(error))
            if self.config.get('camera') and not self.pm.active('camera'):
                try: await self.start_camera(force=True)
                except Exception as error: self.errors.append('Camera preview unavailable at startup: '+str(error))
    async def prepare_camera(self):
        if 'camera' not in self.config: raise ValueError('Camera is not configured')
        if not self.pm.active('camera'): await self.start_camera(force=True)
        for attempt in range(16):
            h=self.camera_health()
            if h['healthy'] and h['camera_info_seen']: return
            if attempt==15: raise ValueError('Camera frames or CameraInfo are unavailable; check camera diagnostics')
            await asyncio.sleep(1)

    async def prepare_recording(self):
        # Enabling RGB means Record must acquire both sensors. Disabling it keeps
        # the independent LiDAR-only / record-only path available.
        if self.config['system']['camera']['enabled'] and not self.pm.active('camera'):
            await self.start_camera()
        if not self.pm.active('driver'): await self.start_driver()
        for attempt in range(16):
            try:
                self.require_health()
                if self.config['system']['camera']['enabled']:
                    h=self.camera_health()
                    if not h['healthy'] or not h['camera_info_seen']:
                        raise ValueError('Camera-enabled recording requires 720×540 images and CameraInfo; check camera diagnostics or disable RGB for LiDAR-only recording')
                return
            except ValueError:
                if attempt==15: raise
                await asyncio.sleep(1)

    async def start_recording(self,sid,camera_only=False):
        if self.pm.active('recording'): raise ValueError('Recording is already active; stop the current session before starting another')
        if any(self.pm.active(k) for k in ('calibration_record','calibration_tool','offline','export','tool')): raise ValueError('Finish processing, editing or calibration before recording')
        p=self.sessions.get(sid)
        if self.active and self.active!=p: raise ValueError('Another session is active')
        if (p/'raw_bag').exists(): raise ValueError('This session already contains a raw bag; create another session')
        if camera_only:
            if shutil.disk_usage(self.root).free<self.config['system']['storage']['minimum_free_gb']*1e9: raise ValueError('Insufficient free disk space')
            await self.prepare_camera()
        else: await self.prepare_recording()
        acquisition=copy.deepcopy(self.config)
        if camera_only: acquisition['system']['camera']['enabled']=True
        # Re-snapshot at acquisition time, not merely when a name is reserved.
        shutil.copytree(self.root/'config',p/'config_snapshot',dirs_exist_ok=True); atomic_json(p/'active_config.json',acquisition)
        self.active=p; self.rate_baseline={k:self.health().get(k,{}).get('count',0) for k in ('lidar','imu')}
        async def done(item):
            if item['state']=='failed': self.errors.append('Recorder exited unexpectedly; inspect recording.log and bag metadata'); self.sessions.update(p,state='failed')
        try:
            result=await self.start_process('recording',commands.record(p,acquisition,camera_only=camera_only),p/'logs/recording.log',p/'raw_bag',done)
            # Do not announce SCANNING before the recorder has initialized its
            # signal handlers/output. An immediate Stop must finalize safely.
            for attempt in range(100):
                if not self.pm.active('recording'): raise ValueError('Recorder exited during startup; inspect recording.log')
                ready=('MOCK recording started' in (p/'logs/recording.log').read_text(errors='replace')) if self.mock else any((p/'raw_bag').glob('*.db3'))
                if ready: break
                await asyncio.sleep(.05)
            else: raise ValueError('Recorder did not create its bag within 5 seconds')
        except Exception:
            await self.pm.stop('recording',5)
            self.sessions.update(p,state='failed',failure_stage='recorder_start')
            self.active=None
            raise
        self.record_started=time.time(); self.sessions.update(p,state='recording',start_time=now(),kind='camera' if camera_only else 'scan',topics=commands.acquisition_topics(acquisition,camera_only=camera_only),camera=camera_metadata(self.root,acquisition),lidar_ip=self.config['sensor']['lidar_ip'])
        return result
    async def stop_recording(self):
        if not self.active or self.record_started is None: return
        p=self.active; duration=time.time()-self.record_started if self.record_started else 0
        item=await self.pm.stop('recording',self.config['system']['shutdown']['recording_timeout'])
        finalized=self.mock or (p/'raw_bag/metadata.yaml').is_file()
        ok=bool(item and item['returncode']==0 and not item['forced'] and finalized)
        h=self.health(); rates={k:(h.get(k,{}).get('count',0)-self.rate_baseline.get(k,0))/duration if duration and not self.mock else None for k in ('lidar','imu')}
        camera_stats={}
        acquired=read_json(p/'active_config.json',self.config)
        recording_errors=[]
        if finalized and not self.mock:
            import yaml
            try:
                info=yaml.safe_load((p/'raw_bag/metadata.yaml').read_text())['rosbag2_bagfile_information']
                seconds=info['duration']['nanoseconds']/1e9
                entries={entry['topic_metadata']['name']:entry for entry in info['topics_with_message_count']}
                expected=read_json(p/'metadata.json',{}).get('topics',{})
                for topic in expected.values():
                    if entries.get(topic,{}).get('message_count',0)<=0: recording_errors.append('No recorded messages on '+topic)
                if seconds<=0: recording_errors.append('Recording has no positive duration')
                for topic,entry in entries.items():
                    if acquired['system']['camera']['enabled'] and topic==acquired['camera']['image_topic']:
                        camera_stats=dict(image_count=entry['message_count'],measured_image_hz=entry['message_count']/seconds if seconds else None)
                    for kind,key in [('lidar','points_topic'),('imu','imu_topic')]:
                        if topic==acquired['sensor'][key]: rates[kind]=entry['message_count']/seconds if seconds else None
            except (OSError,KeyError,TypeError,ValueError,yaml.YAMLError) as error:
                finalized=False;recording_errors.append('Invalid bag metadata: '+str(error))
            if recording_errors: ok=False;self.errors.extend(recording_errors)
        if read_json(p/'metadata.json',{}).get('kind')=='camera': rates={'lidar':None,'imu':None}
        self.sessions.update(p,state='recorded' if ok else 'failed',end_time=now(),duration=duration,average_lidar_hz=rates['lidar'],average_imu_hz=rates['imu'],disk_usage_bytes=size(p),bag_finalized=finalized,recording_errors=recording_errors)
        meta=read_json(p/'metadata.json',{})
        if meta.get('camera',{}).get('enabled'):
            camera_stats.setdefault('image_count',0);camera_stats.setdefault('measured_image_hz',None)
            self.sessions.update(p,camera={**meta['camera'],**camera_stats})
        self.record_started=None
        if not ok: self.errors.append('Recording did not finalize cleanly; preserve raw_bag and inspect recording.log')
    def glim_available(self):
        return self.mock or all((self.root/'ros2_ws/install/glim_ros/lib/glim_ros'/name).is_file() for name in ('glim_rosnode','glim_rosbag'))
    def check_preset(self,preset):
        if preset not in PRESETS: raise ValueError('Unknown GLIM preset')
        if not self.glim_available(): raise ValueError('GLIM is not installed here. Use Record-only Session and process the completed session on the workstation, or install GLIM.')
        if not self.mock and preset in ('jetson_gpu','offline_quality') and (not read_json(self.root/'.state/build_capabilities.json',{}).get('cuda',False) or not (self.root/'ros2_ws/install/glim/lib/libodometry_estimation_gpu.so').exists()):
            raise ValueError('This installation has no CUDA GLIM module; select the CPU preset or build CUDA support')
    async def start_glim(self,sid,preset,viewer=False):
        self.check_preset(preset)
        if viewer and not os.environ.get('DISPLAY'): raise ValueError('Native GLIM viewer needs a server display')
        self.require_health()
        if self.config['system']['loop_closure']['enabled']: raise ValueError('ScanContext is unavailable until separately validated')
        if any(self.pm.active(k) for k in ('offline','export','tool','calibration_record','calibration_tool')): raise ValueError('Finish offline processing/export and close the native editor first')
        p=self.sessions.get(sid)
        if self.active and self.active!=p: raise ValueError('Another session is active')
        if self.pm.active('glim'): raise ValueError('GLIM is already active')
        if read_json(p/'metadata.json',{}).get('kind')=='camera': raise ValueError('Camera-only recordings cannot be processed by LiDAR/IMU GLIM')
        out=p/'glim_dump'
        if any(out.iterdir()): raise ValueError('Live output exists; use a new session or offline reprocessing')
        cfg=p/'config_snapshot'/('live_'+uuid.uuid4().hex[:8]); commands.preset_snapshot(self.root,p,preset,cfg)
        if viewer:
            path=cfg/'config_ros.json'; obj=read_json(path); obj['glim_ros']['extension_modules']=['libstandard_viewer.so']; atomic_json(path,obj)
        self.active=p
        meta=read_json(p/'metadata.json',{})
        self.sessions.update(p,glim_live=True,live_preset=preset,**({'start_time':now(),'state':'mapping'} if not meta.get('start_time') else {}))
        async def done(item):
            valid=self.mock or (out/'graph.bin').exists()
            self.sessions.update(p,live_result='completed' if item['returncode']==0 and valid and not item['forced'] else 'failed')
        result=await self.start_process('glim',commands.glim(cfg,out),p/'logs/glim.log',out,done)
        self.live_preset=preset
        return result
    async def stop_glim(self):
        await self.pm.stop('glim',self.config['system']['shutdown']['glim_timeout'])
        if self.active and not (self.active/'raw_bag').exists():
            meta=read_json(self.active/'metadata.json',{})
            if meta.get('start_time'):
                duration=time.time()-datetime.fromisoformat(meta['start_time']).timestamp()
                self.sessions.update(self.active,end_time=now(),duration=duration,state='mapped' if meta.get('live_result')=='completed' else 'failed')
    async def stop_session(self):
        p=self.active
        if not p: return
        if self.record_started is not None or self.pm.active('recording'): await self.stop_recording()
        await self.stop_glim()
        for role in ('driver','monitor','preview','camera','camera_monitor','camera_preview'):
            src=self.root/f'.state/{role}.log'
            if src.exists(): await asyncio.to_thread(shutil.copy2,src,p/'logs'/f'{role}.log')
        self.sessions.update(p,finalized_at=now(),disk_usage_bytes=await asyncio.to_thread(size,p)); self.event('session_stopped',session=p.name); self.active=None
    async def offline(self,sid,preset):
        self.check_preset(preset)
        if self.config['system']['loop_closure']['enabled']: raise ValueError('ScanContext is unavailable until validated')
        if any(self.pm.active(k) for k in ('recording','glim','offline','export','tool','calibration_record','calibration_tool')): raise ValueError('GLIM or export is already active')
        p=self.sessions.get(sid)
        if self.active==p or read_json(p/'metadata.json',{})['state'] in ('created','recording','interrupted','failed'): raise ValueError('Session needs a finalized recording before processing')
        if not self.mock and not (p/'raw_bag/metadata.yaml').exists(): raise ValueError('No finalized ROS bag found')
        if read_json(p/'metadata.json',{}).get('kind')=='camera': raise ValueError('Camera-only recordings cannot be processed by LiDAR/IMU GLIM')
        runs=p/'processing'; i=1
        while (runs/f'run_{i:03d}').exists(): i+=1
        run=runs/f'run_{i:03d}'; run.mkdir(); cfg=commands.preset_snapshot(self.root,p,preset,run/'config'); dump=run/'glim_dump'
        job=dict(id=run.name,session=sid,preset=preset,state='running',started_at=now(),mock=self.mock); atomic_json(run/'job.json',job)
        async def done(item):
            valid=self.mock or ((dump/'graph.bin').exists() and (dump/'traj_lidar.txt').exists() and (dump/'traj_lidar.txt').stat().st_size>0)
            if not self.mock:
                bad=('timestamp rewind detected', 'large time difference between points and imu', 'waiting for IMU data')
                with (run/'job.log').open(errors='replace') as log:
                    issues=sorted({pattern for line in log for pattern in bad if pattern in line})
                job['validation_errors']=issues
                if issues: valid=False
            job.update(state=item['state'] if item['state']!='completed' or valid else 'failed',returncode=item['returncode'],forced=item['forced'],ended_at=now(),result=str(dump.relative_to(p)) if valid else None)
            atomic_json(run/'job.json',job)
            if job['state']=='failed': self.errors.append(f'GLIM processing failed: {sid}/{run.name}; inspect job.log')
        try: await self.start_process('offline',commands.glim(cfg,dump,p/'raw_bag'),run/'job.log',dump,done)
        except Exception as e: job.update(state='failed',error=str(e)); atomic_json(run/'job.json',job); raise
        return job
    async def export(self,sid,run_id):
        p=self.sessions.get(sid); run=self.get_run(p,run_id); job=read_json(run/'job.json',{})
        if job.get('state')!='completed': raise ValueError('Process this session successfully before export')
        if self.mock: raise ValueError('Mock results cannot be exported as maps')
        if any(self.pm.active(k) for k in ('recording','glim','offline','export','tool','calibration_record','calibration_tool')): raise ValueError('GLIM or export is active')
        if not os.environ.get('DISPLAY'): raise ValueError('Official GLIM exporter needs an OpenGL display; run on a desktop workstation or configured Xvfb')
        output=p/'exports'/f'{run_id}_{uuid.uuid4().hex[:8]}.ply'
        async def done(item):
            if item['returncode']!=0 or not output.is_file(): raise ValueError('PLY export failed; inspect export.log')
        return await self.start_process('export',commands.export(run/'glim_dump',output,run/'config'),run/'export.log',done=done)
    def get_run(self,p,rid):
        import re
        if not re.fullmatch(r'run_\d{3,}',rid): raise ValueError('Invalid processing run')
        run=p/'processing'/rid
        if not run.is_dir() or run.is_symlink(): raise ValueError('Processing run not found')
        return run
    def delete_run(self,sid,rid):
        if any(self.pm.active(k) for k in ('recording','glim','offline','export','tool','calibration_record','calibration_tool')): raise ValueError('Stop processing before deleting derived data')
        run=self.get_run(self.sessions.get(sid),rid); shutil.rmtree(run)
    async def open_tool(self,sid,rid,kind,additional):
        if self.mock: raise ValueError('Native GLIM editing requires a real saved map')
        if any(self.pm.active(k) for k in ('recording','glim','offline','export','tool','calibration_record','calibration_tool')): raise ValueError('Finish GLIM processing or close the current editor first')
        if not os.environ.get('DISPLAY'): raise ValueError('GLIM editing opens on the server desktop. Use a local display on Jetson or copy the session to a workstation.')
        def source(sid,rid):
            p=self.sessions.get(sid); run=self.get_run(p,rid)
            if read_json(run/'job.json',{}).get('state')!='completed': raise ValueError('Select a successfully processed run')
            return dict(session=sid,run=rid,dump=run/'glim_dump',config=run/'config')
        p=self.sessions.get(sid); sources=[source(sid,rid)]+[source(x['session'],x['run']) for x in additional]
        required=sum(size(x['dump']) for x in sources)
        if shutil.disk_usage(self.root).free<required+self.config['system']['storage']['minimum_free_gb']*1e9: raise ValueError('Not enough space for safe map working copies')
        workspace,meta=await asyncio.to_thread(glim_tools.prepare,self.root,p,rid,sources,kind)
        async def done(item):
            meta.update(state='closed' if item['returncode']==0 else 'interrupted',closed_at=now(),returncode=item['returncode'],forced=item['forced'])
            atomic_json(workspace/'workspace.json',meta)
        try: await self.start_process('tool',glim_tools.command(kind,Path(meta['maps'][0])),workspace/'tool.log',done=done)
        except Exception as e: meta.update(state='failed',error=str(e));atomic_json(workspace/'workspace.json',meta);raise
        meta['state']='open';atomic_json(workspace/'workspace.json',meta);return meta
    async def start_validator(self):
        if not self.pm.active('driver'): raise ValueError('Start the sensor first')
        c=self.config['sensor']
        return await self.start_process('validator',['ros2','run','glim_ros','validator_node','--ros-args','-r',f"imu:={c['imu_topic']}",'-r',f"points:={c['points_topic']}"],self.root/'.state/validator.log')
    def edit_workspace(self,sid,eid):
        import re
        if not re.fullmatch(r'edit_[a-f0-9]{12}',eid): raise ValueError('Invalid edit workspace')
        p=self.sessions.get(sid)/'edits'/eid
        if not p.is_dir() or p.is_symlink(): raise ValueError('Edit workspace not found')
        return p
    async def export_edit(self,sid,eid):
        if any(self.pm.active(k) for k in ('recording','glim','offline','export','tool','calibration_record','calibration_tool')): raise ValueError('Close editing and finish processing before export')
        if not os.environ.get('DISPLAY'): raise ValueError('Official GLIM exporter requires a display')
        workspace=self.edit_workspace(sid,eid); dump=workspace/'saved_map'
        # Native tools must save explicitly into the displayed output location.
        glim_tools.validate_dump(dump)
        target=self.sessions.get(sid)/'exports'/f'{eid}_{uuid.uuid4().hex[:8]}.ply'
        cfg=dump/'config'
        if not cfg.is_dir(): cfg=workspace/'map_01/config'
        async def done(item):
            if item['returncode']!=0 or not target.is_file(): raise ValueError('Edited-map export failed; inspect the workspace export.log')
        return await self.start_process('export',commands.export(dump,target,cfg),workspace/'export.log',done=done)
    async def background(self):
        while not self.closed:
            try:
                if not self.mock: await self.refresh_host_ip()
                async with self.lock: await self.capture.reconcile()
                self.net={'state':'mock'} if self.mock else await network(self.config['sensor'])
                if (self.pm.active('recording') or self.pm.active('calibration_record')) and shutil.disk_usage(self.root).free<self.config['system']['storage']['minimum_free_gb']*1e9:
                    async with self.lock:
                        self.errors.append('Low disk threshold reached; stopping acquisition gracefully')
                        if self.calibrations.active: await self.calibrations.capture_stop(self.calibrations.active[0].name)
                        await self.stop_session()
            except Exception as e: self.errors.append(str(e))
            await asyncio.sleep(2)
    async def refresh_host_ip(self):
        async with self.lock:
            sensor=self.config['sensor']; interface,current=wired_connection(sensor)
            if (interface,current)==(sensor['interface'],sensor['host_ip']): return
            was_running=self.pm.active('driver')
            if self.calibrations.active: await self.calibrations.capture_stop(self.calibrations.active[0].name)
            if self.active and read_json(self.active/'metadata.json',{}).get('kind')!='camera': await self.stop_session()
            if was_running: await self.stop_driver()
            previous=sensor['host_ip']; sensor['host_ip']=current; sensor['interface']=interface
            self.event('host_ip_changed',previous=previous,current=current)
            if was_running and current: await self.start_driver()
    async def close(self):
        self.closed=True
        await self.capture.close()
        if self.calibrations.active: await self.calibrations.capture_stop(self.calibrations.active[0].name)
        await self.stop_session()
        for k in ('nksr','nksr_check','reconstruction','calibration_record','calibration_tool','offline','export','tool','validator','camera_preview','camera_monitor','camera','preview','monitor','driver'):
            if self.pm.items.get(k,{}).get('state')!='orphaned': await self.pm.stop(k,self.config['system']['shutdown']['glim_timeout'] if k in ('offline','export','tool','calibration_tool') else 20,cancel=True)

    def camera_calibration(self):
        from .camera_config import config_path
        c=self.config.get('camera')
        intr=intrinsics_status(self.root,c)
        ext={}
        if c:
            import yaml
            try:ext=yaml.safe_load(config_path(self.root,c['extrinsics_file']).read_text()) or {}
            except (OSError,yaml.YAMLError):pass
        return dict(intrinsics=intr,extrinsics=ext)

    def camera_health(self):
        running=self.pm.active('camera');enabled=self.config['system']['camera']['enabled'] or running
        base=dict(state='disabled' if not enabled else 'stopped',healthy=False,camera_running=running,hz=0,image_hz=0,image_age=None,last_image_timestamp=None,width=None,height=None,camera_info_seen=False,camera_info_valid=False,frame_id=None)
        if not enabled or not running:return base
        if self.mock:
            c=self.config['camera'];valid=intrinsics_status(self.root,c)['status']=='VALID'
            return dict(base,state='mock',healthy=True,hz=c['expected_hz'],image_hz=c['expected_hz'],image_age=0.01,last_image_timestamp=time.time()-.01,width=c['width'],height=c['height'],frame_id=c['frame_id'],camera_info_seen=True,camera_info_valid=valid,timestamp_age_sec=.01,timestamp_jitter_sec=0,mock=True)
        data=read_json(self.root/'.state/camera_health.json',{})
        if not self.pm.active('camera_monitor') or time.time()-data.get('updated_at',0)>3:return dict(base,state='monitor_stale')
        return {**base,**data,'camera_running':running}

    async def start_camera(self,force=False):
        if not force and not self.config['system']['camera']['enabled']:raise ValueError('Camera is disabled in config/system.yaml')
        if self.pm.active('camera'):raise ValueError('Camera is already running')
        if not self.mock:
            args=commands.camera(self.root,self.config)
            await check_camera_dependencies(self.config['camera'])
        else:args=[]
        for name in ('camera_health.json','camera_preview.jpg','camera_calibration_frame.jpg'):(self.root/'.state'/name).unlink(missing_ok=True)
        result=await self.start_process('camera',args,self.root/'.state/camera.log')
        if not self.mock:
            for role in ('camera_monitor','camera_preview'):
                try:await self.start_process(role,[sys.executable,'-c',f'from factory_mapping.ros_nodes import {role}; {role}()'],self.root/f'.state/{role}.log')
                except Exception as e:self.errors.append(f'{role} unavailable: {e}; camera publishing continues')
        return result

    async def stop_camera(self):
        if self.pm.active('calibration_record'):raise ValueError('Stop calibration capture before stopping camera')
        for role in ('camera_preview','camera_monitor','camera'):await self.pm.stop(role,20)
        self.event('camera_stopped',recording_continues=self.pm.active('recording'))
