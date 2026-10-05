#!/usr/bin/env python3
"""Exercise real camera selection with the dashboard stopped. Does not record bags."""
import argparse
import asyncio
import fcntl
import json
import platform
import time
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from factory_mapping.config import ROOT
from factory_mapping.service import Service
from factory_mapping.storage import atomic_json


def installed(name):
    try: return version(name)
    except PackageNotFoundError: return None


def contents(path):
    try: return Path(path).read_text().strip()
    except OSError: return None


async def diagnose(args):
    service=Service()
    service.config['system']['camera']['profile']=args.preference
    service.config['system']['camera']['enabled']=True
    log=ROOT/'.state/camera.log';offset=log.stat().st_size if log.exists() else 0
    report=dict(architecture=platform.machine(),ubuntu=contents('/etc/os-release'),
                jetson_release=contents('/etc/nv_tegra_release'),pyrealsense2=installed('pyrealsense2'),
                requested_case=args.case,preference=args.preference,samples=[])
    try:
        started=time.monotonic()
        usable=await service.camera_selection.resolve(required=False)
        report['selection_seconds']=time.monotonic()-started
        deadline=time.monotonic()+args.seconds
        while usable and time.monotonic()<deadline:
            report['samples'].append(service.camera_health())
            await asyncio.sleep(.5)
        report['camera_selection']=service.camera_selection.view()
        report['configured_camera']=service.config['camera']
        report['stream_usable']=usable and service.camera_selection.usable(service.camera_health())
        if log.exists():
            with log.open() as stream:
                stream.seek(offset);report['publisher_log']=stream.read()[-16000:]
        expected={'d405':{'d405'},'dfk':{'dfk33ux287'},'both':{'d405','dfk33ux287'},'neither':set()}[args.case]
        found={p for p,c in report['camera_selection']['candidates'].items() if c['detected']}
        selected=report['camera_selection']['active_profile']
        order=['dfk33ux287','d405'] if args.preference=='dfk33ux287' else ['d405','dfk33ux287']
        wanted=next((p for p in order if p in expected),None)
        report['passed']=found==expected and selected==wanted and (report['stream_usable'] if expected else not usable)
        report['note']='Camera stream check only. Run record_test.sh with Mid-360 attached to validate the completed bag and target storage.'
    except Exception as error:
        report.update(passed=False,error=str(error))
    finally:
        await service.close()
    output=ROOT/f'.state/camera_diagnostic_{args.case}.json'
    atomic_json(output,report)
    print(json.dumps(report,indent=2));print('Report:',output)
    return 0 if report['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,choices=['d405','dfk','both','neither'])
    parser.add_argument('--preference',default='auto',choices=['auto','d405','dfk33ux287'])
    parser.add_argument('--seconds',type=int,default=10,choices=range(2,301),metavar='2..300')
    args=parser.parse_args()
    (ROOT/'.state').mkdir(exist_ok=True)
    with (ROOT/'.state/backend.lock').open('w') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Stop the dashboard before running camera diagnostics')
        raise SystemExit(asyncio.run(diagnose(args)))


if __name__=='__main__': main()
