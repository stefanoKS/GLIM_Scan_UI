#!/usr/bin/env python3
"""Terminal acquisition uses the same preflight/recorder as the dashboard."""
import argparse, asyncio, fcntl
from factory_mapping.config import ROOT
from factory_mapping.service import Service
from factory_mapping.storage import read_json

async def main(args):
    service=Service()
    session=service.sessions.create(args.name,'Terminal acquisition test',service.config)
    try:
        await service.start_recording(session['id'])
        print('Sensor health:',service.health(),flush=True)
        await asyncio.sleep(args.seconds)
        await service.stop_session()
        meta=read_json(service.sessions.get(session['id'])/'metadata.json',{})
        if meta.get('state')!='recorded' or not meta.get('bag_finalized'): raise RuntimeError('Bag did not finalize cleanly')
        print('Recorded session:',session['id'],flush=True)
    finally: await service.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='terminal_test');parser.add_argument('--seconds',type=int,default=30)
    args=parser.parse_args()
    if not 1<=args.seconds<=86400:parser.error('--seconds must be 1..86400')
    with (ROOT/'.state/backend.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(main(args))
