"""Explicit mock subprocess; never publishes ROS or creates a production rosbag."""
import argparse,signal,time
from pathlib import Path
from .storage import atomic_json
p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('out');a=p.parse_args();running=True

def stop(*_):
 global running
 running=False
signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
print('MOCK '+a.role+' started',flush=True)
start=time.time()
while running and (a.role not in ('offline','export') or time.time()-start<4):
 print('MOCK elapsed %.1fs'%(time.time()-start),flush=True);time.sleep(.5)
out=Path(a.out)
if a.role=='recording':
 out.mkdir(parents=True,exist_ok=True);(out/'MOCK_ONLY.txt').write_text('Not a ROS bag. No production sensor data.\n')
if a.role in ('offline','glim'):
 out.mkdir(parents=True,exist_ok=True);atomic_json(out/'mock_result.json',{'mock':True})
print('MOCK '+a.role+' stopped',flush=True)
