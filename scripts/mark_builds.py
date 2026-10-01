"""Record build evidence separately from ARM64 validation."""
import json,platform,subprocess
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'dependencies.lock'; d=json.loads(p.read_text())
for name in __import__('sys').argv[1:]:
 d['dependencies'][name]['validation']='built_on_'+platform.machine()
 d['dependencies'][name]['arm64_validated']=False
p.write_text(json.dumps(d,indent=2)+'\n')
