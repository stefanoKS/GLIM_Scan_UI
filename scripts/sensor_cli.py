#!/usr/bin/env python3
"""Terminal-only acquisition proof before UI development."""
import argparse, asyncio, os, signal, time, fcntl
from factory_mapping.config import ROOT,load
from factory_mapping import commands
from factory_mapping.processes import ProcessManager
from factory_mapping.storage import Sessions, now, size, read_json

async def main(args):
 c=load(); sessions=Sessions(ROOT); pm=ProcessManager(ROOT/'.state'); session=sessions.create(args.name,'Terminal acquisition test',c); p=sessions.get(session['id'])
 try:
  await pm.start('driver',commands.driver(ROOT,c),p/'logs/driver.log',commands.ros_env(c))
  await pm.start('monitor',['python','-c','from factory_mapping.ros_nodes import monitor; monitor()'],p/'logs/monitor.log',commands.ros_env(c))
  for _ in range(20):
   await asyncio.sleep(1); h=read_json(ROOT/'.state/health.json',{})
   if time.time()-h.get('updated_at',0)<3 and all(h.get(k,{}).get('state')=='healthy' for k in ('lidar','imu')): break
  else: raise RuntimeError('PointCloud2 and IMU are not healthy; see session logs')
  print('Sensor health:',h,flush=True)
  await pm.start('recording',commands.record(p,c),p/'logs/recording.log',commands.ros_env(c))
  start=time.time(); sessions.update(p,start_time=now(),state='recording')
  await asyncio.sleep(args.seconds)
  item=await pm.stop('recording',60)
  ok=(p/'raw_bag/metadata.yaml').exists() and not item['forced'] and item['returncode']==0
  sessions.update(p,end_time=now(),duration=time.time()-start,state='recorded' if ok else 'failed',disk_usage_bytes=size(p),average_lidar_hz=h['lidar']['hz'],average_imu_hz=h['imu']['hz'])
  if not ok: raise RuntimeError('Bag did not finalize cleanly')
  print('Recorded session:',session['id'],flush=True)
 finally:
  for k in ('recording','monitor','driver'): await pm.stop(k,60)

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--name',default='terminal_test'); p.add_argument('--seconds',type=int,default=30)
 with (ROOT/'.state/backend.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  asyncio.run(main(p.parse_args()))
