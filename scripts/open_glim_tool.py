"""Launch official map tools in a protected derived workspace, with UI stopped."""
import argparse,asyncio,fcntl,json
from factory_mapping.config import ROOT
from factory_mapping.service import Service
async def main(a):
 with (ROOT/'.state/backend.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  s=Service()
  try:
   meta=await s.open_tool(a.session,a.run,a.tool,[dict(session=v[0],run=v[1]) for v in a.add_map]);print(json.dumps(meta,indent=2),flush=True)
   await asyncio.shield(s.pm.items['tool']['watcher'])
  finally:await s.close()
p=argparse.ArgumentParser();p.add_argument('tool',choices=['offline_viewer','map_editor']);p.add_argument('session');p.add_argument('run');p.add_argument('--add-map',nargs=2,action='append',default=[],metavar=('SESSION','RUN'));asyncio.run(main(p.parse_args()))
