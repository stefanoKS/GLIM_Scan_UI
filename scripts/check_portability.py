#!/usr/bin/env python3
"""Audit build artifacts without pretending this replaces an ARM64 build."""
import json, platform, re, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
files=list((root/'ros2_ws/build').glob('**/flags.make'))+list((root/'external').glob('*/build/**/flags.make'))
forbidden=re.compile(r'(?<![A-Za-z])(?:-mavx\S*|-msse\S*|-march=native|-march=x86\S*)')
violations=[]; platform_dispatch_flags=[]
# Pinned tiscamera selects dutils_img_filter_arm.cmake for aarch64; these
# two PC targets legitimately inherit SSE4.1 from its Intel-only backend.
intel_camera_targets={
 'external/tiscamera/build/src/gstreamer-1.0/tcamconvert/CMakeFiles/tcamconvert.dir/flags.make',
 'external/tiscamera/build/libs/dutils_image/src/dutils_img_filter/CMakeFiles/dutils_img_filter_sse41.dir/flags.make',
}
for f in files:
 for match in forbidden.findall(f.read_text(errors='replace')):
  entry={'file':str(f.relative_to(root)),'flag':match}
  allowed=platform.machine()=='x86_64' and entry['file'] in intel_camera_targets and match=='-msse4.1'
  (platform_dispatch_flags if allowed else violations).append(entry)
report=dict(architecture=platform.machine(),files_checked=len(files),violations=violations,platform_dispatch_flags=platform_dispatch_flags,arm64_build_verified=platform.machine()=='aarch64' and bool(files))
(root/'.state/portability.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if violations:raise SystemExit(1)
