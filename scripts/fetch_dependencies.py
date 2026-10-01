#!/usr/bin/env python3
"""Fetch immutable revisions. Never reset a dirty external checkout."""
import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def run(*args):subprocess.run(args,check=True)
for name,dep in json.loads((root/'dependencies.lock').read_text())['dependencies'].items():
 p=root/'external'/name
 if not p.exists(): run('git','clone','--no-checkout',dep['url'],str(p))
 current=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
 if current!=dep['sha']:
  if subprocess.check_output(['git','-C',str(p),'status','--porcelain'],text=True).strip():raise SystemExit(f'{p} is dirty; refusing to change revision')
  run('git','-C',str(p),'fetch','origin',dep['sha']);run('git','-C',str(p),'checkout','--detach',dep['sha'])
 run('git','-C',str(p),'submodule','update','--init','--recursive')
