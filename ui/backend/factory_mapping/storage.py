from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib, json, os, platform, re, shutil, subprocess, tempfile, uuid, zipfile

def now(): return datetime.now(timezone.utc).isoformat()
def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
def read_json(path, default=None):
    try: return json.loads(Path(path).read_text())
    except (OSError, ValueError): return default

def size(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file() and not p.is_symlink()) if Path(path).exists() else 0

def sha(path):
    p = subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True)
    return p.stdout.strip() if p.returncode == 0 else None

def machine():
    def read(p):
        try: return Path(p).read_text().replace('\0','').strip()
        except OSError: return None
    return dict(model=read('/proc/device-tree/model') or platform.machine()+' workstation',os=platform.platform(),jetpack=read('/etc/nv_tegra_release'),architecture=platform.machine())

class Sessions:
    def __init__(self, root, mock=False):
        self.root=root; self.base=root/'data/sessions'; self.base.mkdir(parents=True, exist_ok=True); self.mock=mock
    def get(self, sid):
        if not re.fullmatch(r'\d{8}_\d{6}_[A-Za-z0-9_-]{1,80}',sid): raise ValueError('Invalid session ID')
        p=self.base/sid
        if not p.is_dir() or p.is_symlink() or p.resolve().parent != self.base.resolve(): raise ValueError('Session not found')
        if read_json(p/'metadata.json',{}).get('mock',False) != self.mock: raise ValueError('Session belongs to a different data mode')
        return p
    def create(self, name, notes, config):
        created=datetime.now()
        name=name.strip()
        slug=re.sub(r'[^A-Za-z0-9_-]+','_',name).strip('_')[:60] if name else 'session'
        if not name: name=created.strftime('Session %Y-%m-%d %H:%M:%S')
        if not slug: raise ValueError('Session name needs letters or numbers')
        sid=created.strftime('%Y%m%d_%H%M%S')+'_'+slug
        p=self.base/sid
        if p.exists(): sid+='_'+uuid.uuid4().hex[:6]; p=self.base/sid
        p.mkdir()
        for d in ('glim_dump','exports','logs','processing'): (p/d).mkdir()
        # rosbag itself creates raw_bag; do not pre-create that directory.
        shutil.copytree(self.root/'config',p/'config_snapshot')
        atomic_json(p/'active_config.json',config)
        meta=dict(id=sid,name=name,notes=notes,created_at=now(),start_time=None,end_time=None,duration=0,state='created',mock=self.mock,**machine(),ros_version=config['system']['ros_distro'],glim_sha=sha(self.root/'external/glim'),livox_sha=sha(self.root/'external/livox_ros_driver2'),repository_sha=sha(self.root),lidar_serial=config['sensor'].get('serial_number'),lidar_ip=config['sensor']['lidar_ip'],topics={k:config['sensor'][k] for k in ('points_topic','imu_topic')},average_lidar_hz=None,average_imu_hz=None,disk_usage_bytes=0,glim_live=False,loop_closure=config['system']['loop_closure']['enabled'])
        from .calibration_data import camera_metadata
        meta['camera']=camera_metadata(self.root,config)
        atomic_json(p/'metadata.json',meta)
        return meta
    def update(self,p,**fields):
        meta=read_json(p/'metadata.json',{}); meta.update(fields); atomic_json(p/'metadata.json',meta); return meta
    def list(self):
        result=[]
        for p in sorted(self.base.iterdir(),reverse=True):
            if not p.is_dir() or p.is_symlink(): continue
            m=read_json(p/'metadata.json')
            if m and m.get('mock',False)==self.mock:
                m['bag_size_bytes']=size(p/'raw_bag'); m['processing']=[read_json(f) for f in sorted((p/'processing').glob('*/job.json'))]; m['exports']=[str(f.relative_to(p)) for f in (p/'exports').glob('*') if f.is_file()]; result.append(m)
        return result

    def export_archive(self,sid,output):
        session=self.get(sid);files=[]
        with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            for path in sorted(session.rglob('*')):
                if path.is_symlink():raise ValueError('Session contains a symlink and cannot be exported')
                if not path.is_file():continue
                relative=path.relative_to(session).as_posix();digest=hashlib.sha256();length=0
                with path.open('rb') as source,archive.open('session/'+relative,'w',force_zip64=True) as destination:
                    for chunk in iter(lambda:source.read(1024*1024),b''):
                        destination.write(chunk);digest.update(chunk);length+=len(chunk)
                files.append(dict(path=relative,size=length,sha256=digest.hexdigest()))
            archive.writestr('manifest.json',json.dumps(dict(schema=1,session_id=sid,mock=self.mock,files=files)))

    def import_archive(self,source):
        with zipfile.ZipFile(source) as archive:
            try:header=archive.getinfo('manifest.json')
            except KeyError as error:raise ValueError('Project manifest is missing') from error
            if header.file_size>16*1024*1024 or header.compress_type!=zipfile.ZIP_STORED:raise ValueError('Unsupported project manifest')
            try:manifest=json.loads(archive.read('manifest.json'))
            except (KeyError,ValueError,UnicodeError) as error:raise ValueError('Invalid project manifest') from error
            if not isinstance(manifest,dict) or manifest.get('schema')!=1 or type(manifest.get('mock')) is not bool or manifest['mock']!=self.mock:
                raise ValueError('Incompatible project archive or data mode')
            sid=manifest.get('session_id')
            if not isinstance(sid,str) or not re.fullmatch(r'\d{8}_\d{6}_[A-Za-z0-9_-]{1,80}',sid):raise ValueError('Invalid project session ID')
            destination=self.base/sid
            if destination.exists():raise ValueError('Session ID already exists; import into a different workspace')
            files=manifest.get('files')
            if not isinstance(files,list) or not files:raise ValueError('Project has no files')
            entries=archive.infolist();names=[entry.filename for entry in entries]
            if len(names)!=len(set(names)) or len(files)!=len(set(item.get('path') for item in files if isinstance(item,dict))):
                raise ValueError('Duplicate project paths')
            expected={'manifest.json'}
            for item in files:
                if not isinstance(item,dict) or not isinstance(item.get('path'),str):raise ValueError('Invalid project file entry')
                path=PurePosixPath(item['path'])
                if path.is_absolute() or not item['path'] or '\\' in item['path'] or any(part in ('..','.') for part in item['path'].split('/')) or path.as_posix()!=item['path']:
                    raise ValueError('Unsafe project path')
                if type(item.get('size')) is not int or item['size']<0 or not isinstance(item.get('sha256'),str) or not re.fullmatch(r'[0-9a-f]{64}',item['sha256']):
                    raise ValueError('Invalid project file metadata')
                expected.add('session/'+item['path'])
            if set(names)!=expected or 'session/metadata.json' not in expected or 'session/active_config.json' not in expected or not any(name.startswith('session/config_snapshot/') for name in expected):
                raise ValueError('Incomplete project archive')
            if any('session/'+str(parent) in expected for item in files for parent in PurePosixPath(item['path']).parents if str(parent)!='.'):
                raise ValueError('Project file conflicts with a directory')
            by_name={entry.filename:entry for entry in entries}
            total=sum(item['size'] for item in files)
            if total>shutil.disk_usage(self.base).free-1024**3:raise ValueError('Insufficient space for imported project')
            for item in files:
                entry=by_name['session/'+item['path']]
                if entry.file_size!=item['size'] or entry.compress_type!=zipfile.ZIP_STORED or entry.flag_bits & 1 or (entry.external_attr>>16)&0o170000==0o120000:
                    raise ValueError('Unsupported or unsafe project entry')
            staged=Path(tempfile.mkdtemp(prefix='.import-',dir=self.base))
            try:
                for item in files:
                    path=staged/item['path'];path.parent.mkdir(parents=True,exist_ok=True)
                    digest=hashlib.sha256();length=0
                    with archive.open('session/'+item['path']) as reader,path.open('wb') as writer:
                        for chunk in iter(lambda:reader.read(1024*1024),b''):
                            writer.write(chunk);digest.update(chunk);length+=len(chunk)
                    if length!=item['size'] or digest.hexdigest()!=item['sha256']:raise ValueError('Project file checksum mismatch')
                metadata=read_json(staged/'metadata.json',{})
                if metadata.get('id')!=sid or metadata.get('mock',False)!=self.mock:raise ValueError('Project metadata does not match manifest')
                config=read_json(staged/'active_config.json')
                if not isinstance(config,dict) or not isinstance(config.get('system'),dict) or not isinstance(config.get('sensor'),dict) or not (staged/'config_snapshot/system.yaml').is_file():
                    raise ValueError('Project configuration snapshot is incomplete')
                for folder in ('glim_dump','exports','logs','processing'):(staged/folder).mkdir(exist_ok=True)
                staged.rename(destination)
            finally:
                if staged.exists():shutil.rmtree(staged)
            return metadata
