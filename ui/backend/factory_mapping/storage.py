from datetime import datetime, timezone
from pathlib import Path
import json, os, platform, re, shutil, subprocess, uuid

def now(): return datetime.now(timezone.utc).isoformat()
def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
def read_json(path, default=None):
    try: return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError): return default

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
        slug=re.sub(r'[^A-Za-z0-9_-]+','_',name).strip('_')[:60]
        if not slug: raise ValueError('Session name needs letters or numbers')
        sid=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+slug
        p=self.base/sid
        if p.exists(): sid+='_'+uuid.uuid4().hex[:6]; p=self.base/sid
        p.mkdir()
        for d in ('glim_dump','exports','logs','processing'): (p/d).mkdir()
        # rosbag itself creates raw_bag; do not pre-create that directory.
        shutil.copytree(self.root/'config',p/'config_snapshot')
        atomic_json(p/'active_config.json',config)
        meta=dict(id=sid,name=name,notes=notes,created_at=now(),start_time=None,end_time=None,duration=0,state='created',mock=self.mock,**machine(),ros_version=config['system']['ros_distro'],glim_sha=sha(self.root/'external/glim'),livox_sha=sha(self.root/'external/livox_ros_driver2'),repository_sha=sha(self.root),lidar_serial=config['sensor'].get('serial_number'),lidar_ip=config['sensor']['lidar_ip'],topics={k:config['sensor'][k] for k in ('points_topic','imu_topic')},average_lidar_hz=None,average_imu_hz=None,disk_usage_bytes=0,glim_live=False,loop_closure=config['system']['loop_closure']['enabled'])
        atomic_json(p/'metadata.json',meta)
        return meta
    def update(self,p,**fields):
        meta=read_json(p/'metadata.json',{}); meta.update(fields); atomic_json(p/'metadata.json',meta); return meta
    def list(self):
        result=[]
        for p in sorted(self.base.iterdir(),reverse=True):
            m=read_json(p/'metadata.json')
            if m and m.get('mock',False)==self.mock:
                m['bag_size_bytes']=size(p/'raw_bag'); m['processing']=[read_json(f) for f in sorted((p/'processing').glob('*/job.json'))]; m['exports']=[str(f.relative_to(p)) for f in (p/'exports').glob('*') if f.is_file()]; result.append(m)
        return result
