"""Replay recorded sensor data while offline processing the same bag.

This intentionally puts a real recorded stream in the offline ROS domain to
verify input remapping as well as domain separation. It requires no live sensor.
"""
import argparse,asyncio,hashlib,json,fcntl
from pathlib import Path
from factory_mapping.config import ROOT
from factory_mapping.service import Service
from factory_mapping.storage import read_json,atomic_json
from factory_mapping.commands import ros_env

def hashes(path):
    result={}
    for p in sorted(path.iterdir()):
        if not p.is_file(): continue
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        result[p.name]=h.hexdigest()
    return result

async def main(a):
    with (ROOT/'.state/backend.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        s=Service();p=s.sessions.get(a.session);before=await asyncio.to_thread(hashes,p/'raw_bag')
        await s.pm.start('replay_validation',['ros2','bag','play',str(p/'raw_bag'),'--loop'],ROOT/'.state/replay-validation.log',ros_env(s.config,True))
        try:
            await asyncio.sleep(2)
            job=await s.offline(a.session,'jetson_cpu');await s.pm.items['offline']['watcher']
            result=read_json(p/'processing'/job['id']/'job.json')
            after=await asyncio.to_thread(hashes,p/'raw_bag')
            report={'session':a.session,'job':result,'raw_unchanged':before==after,'hashes':after}
            atomic_json(ROOT/'.state/offline-isolation-validation.json',report);print(json.dumps(report,indent=2))
            if result['state']!='completed' or before!=after:raise RuntimeError('Offline isolation validation failed')
        finally:
            await s.pm.stop('replay_validation',20,cancel=True);await s.close()
p=argparse.ArgumentParser();p.add_argument('session');asyncio.run(main(p.parse_args()))
