#!/usr/bin/env python3
"""Audit build artifacts without pretending this replaces an ARM64 build."""
import json, platform, re, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
files=list((root/'ros2_ws/build').glob('**/flags.make'))+list((root/'external').glob('*/build/**/flags.make'))
forbidden=re.compile(r'(?<![A-Za-z])(?:-mavx\S*|-msse\S*|-march=native|-march=x86\S*)')
violations=[]
for f in files:
 for match in forbidden.findall(f.read_text(errors='replace')): violations.append({'file':str(f.relative_to(root)),'flag':match})
report=dict(architecture=platform.machine(),files_checked=len(files),violations=violations,arm64_build_verified=platform.machine()=='aarch64' and bool(files))
(root/'.state/portability.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if violations:raise SystemExit(1)
