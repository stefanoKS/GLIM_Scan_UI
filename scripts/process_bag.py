import argparse,asyncio,fcntl
from factory_mapping.config import ROOT
from factory_mapping.service import Service
from factory_mapping.storage import read_json
async def main(a):
 with (ROOT/'.state/backend.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  s=Service();job=await s.offline(a.session,a.preset);print(job,flush=True)
  try: await asyncio.shield(s.pm.items['offline']['watcher'])
  finally: await s.close()
  p=s.sessions.get(a.session)/'processing'/job['id']; result=read_json(p/'job.json');print(result,flush=True)
  if result['state']!='completed': raise SystemExit(1)
p=argparse.ArgumentParser();p.add_argument('session');p.add_argument('--preset',default='jetson_cpu');asyncio.run(main(p.parse_args()))
