"""Launch-check the actual native tools using copies of a recorded result.
No edits, loop constraints or quality claims are fabricated by this smoke test.
"""
import asyncio,fcntl,json,argparse
from factory_mapping.config import ROOT
from factory_mapping.service import Service
from factory_mapping.storage import atomic_json

async def main(args):
 with (ROOT/'.state/backend.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  s=Service();results=[]
  try:
   for kind in ('map_editor','offline_viewer'):
    meta=await s.open_tool(args.session,args.run,kind,[])
    await asyncio.sleep(3)
    running=s.pm.active('tool');log=(s.sessions.get(meta['source_session'])/'edits'/meta['id']/'tool.log').read_text()
    results.append({'tool':kind,'launched':running,'workspace':meta['id'],'log':log[-3000:]})
    await s.pm.stop('tool',3,cancel=True)
    if not running:raise RuntimeError(kind+' exited unexpectedly')
  finally:await s.close()
  atomic_json(ROOT/'.state/native-tools-validation.json',results);print(json.dumps(results,indent=2))
p=argparse.ArgumentParser();p.add_argument('session');p.add_argument('run');asyncio.run(main(p.parse_args()))
