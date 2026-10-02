#!/usr/bin/env python3
"""Fetch immutable revisions. Never reset a dirty external checkout."""
import argparse,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def run(*args):subprocess.run(args,check=True)
dependencies=json.loads((root/'dependencies.lock').read_text())['dependencies']
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--only',nargs='+',choices=list(dependencies),help='Fetch only these dependencies (default: all)')
args=parser.parse_args()
for name,dep in dependencies.items():
 if args.only and name not in args.only: continue
 if not args.only and dep.get('optional',False): continue
 p=root/'external'/name
 fresh=not p.exists()
 if fresh: run('git','clone','--no-checkout',dep['url'],str(p))
 current=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
 if fresh or current!=dep['sha']:
  if not fresh and subprocess.check_output(['git','-C',str(p),'status','--porcelain'],text=True).strip():raise SystemExit(f'{p} is dirty; refusing to change revision')
  run('git','-C',str(p),'fetch','origin',dep['sha']);run('git','-C',str(p),'checkout','--detach',dep['sha'])
 run('git','-C',str(p),'submodule','update','--init','--recursive')
