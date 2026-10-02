#!/usr/bin/env python3
"""Query the installed CUDA runtime/device; no desktop CUDA version assumption."""
import json,os,re,shutil,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];state=root/'.state';state.mkdir(exist_ok=True)
compiler=shutil.which('nvcc')
if not compiler:raise SystemExit('CUDA compiler not found; use the CPU build')
version=subprocess.check_output([compiler,'--version'],text=True)
override=os.environ.get('CUDA_ARCHITECTURES')
if override:
 if not re.fullmatch(r'[0-9]+(?:;[0-9]+)*',override):raise SystemExit('CUDA_ARCHITECTURES must contain numeric SM targets, e.g. 87')
 arch=override;method='explicit_override'
else:
 source=state/'cuda_probe.cu';binary=state/'cuda_probe'
 source.write_text('''#include <cuda_runtime.h>
#include <cstdio>
int main() {int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||count<1)return 2;
 for(int i=0;i<count;i++){cudaDeviceProp p;if(cudaGetDeviceProperties(&p,i)!=cudaSuccess)return 3;printf("%s%d%d",i?";":"",p.major,p.minor);}return 0;}
''')
 subprocess.run([compiler,str(source),'-o',str(binary)],check=True,stdout=subprocess.DEVNULL)
 arch=subprocess.check_output([str(binary)],text=True).strip();method='cuda_runtime'
 if not re.fullmatch(r'[0-9]+(?:;[0-9]+)*',arch):raise SystemExit('Could not detect a supported CUDA GPU')
(state/'cuda_info.json').write_text(json.dumps({'compiler':compiler,'version':version,'architectures':arch,'detection':method},indent=2)+'\n')
print(arch)
