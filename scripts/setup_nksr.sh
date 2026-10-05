#!/usr/bin/env bash
# Explicit installation only. Does not modify the ROS/system Python environment.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="${NKSR_CACHE:-$HOME/.cache/factory-mapping}"
SOURCE="$CACHE/NKSR"
PREFIX="${NKSR_PREFIX:-$CACHE/nksr-env}"
COMMIT="e40336845e67761343a756788e5a98b827d4a143"
[[ "$(uname -s)/$(uname -m)" == Linux/x86_64 ]] || { echo 'This verified upstream build targets Linux x86_64'; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi is required to validate the build host'; exit 1; }
nvidia-smi
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
[[ "${DRIVER%%.*}" -ge 570 ]] || { echo 'CUDA 12.8 requires a compatible driver (570+ recommended)'; exit 1; }
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
[[ -x "$CONDA" ]] || CONDA="$HOME/miniconda3/bin/conda"
[[ -x "$CONDA" ]] || { echo 'Install Conda first'; exit 1; }
mkdir -p "$CACHE" "$ROOT/.state"
if [[ ! -d "$SOURCE/.git" ]]; then git clone https://github.com/nv-tlabs/NKSR.git "$SOURCE"; fi
[[ "$(git -C "$SOURCE" remote get-url origin)" == https://github.com/nv-tlabs/NKSR.git ]] || { echo 'Expected official NVIDIA source'; exit 1; }
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$COMMIT" ]] || { echo "Review changed upstream requirements/API before updating (verified $COMMIT)"; exit 1; }
if [[ ! -x "$PREFIX/bin/python" ]]; then
 "$CONDA" env create --prefix "$PREFIX" -f "$SOURCE/environment.yml" -y
fi
export PATH="$PREFIX/bin:$PATH"
export CUDA_HOME="$PREFIX"
export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-1}"
# Conda CUDA 12.8 places headers under targets/, while torch's C++ builder
# searches CUDA_HOME/include. CPATH supplies the official layout without copies.
export CPATH="$PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
unset PYTHONPATH PYTHONHOME
# The upstream requirements point torch-scatter at torch 2.8 despite pinning torch 2.7.
# Preinstall the ABI-matching official PyG wheel; the requirements then keep it.
"$PREFIX/bin/python" -m pip install 'torch==2.7.0+cu128' --index-url https://download.pytorch.org/whl/cu128
"$PREFIX/bin/python" -m pip install torch-scatter --no-deps --only-binary=:all: -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
"$PREFIX/bin/python" -m pip install -r "$SOURCE/requirements.txt"
# NKSR imports pycg.vis at runtime; python-pycg leaves its Open3D extra optional.
# This is an upstream import dependency, not our mesh writer or preparation.
"$PREFIX/bin/python" -m pip install open3d
if ! "$PREFIX/bin/python" - "$COMMIT" <<'PY_CHECK'
import json, pathlib, sys
assert json.loads((pathlib.Path(sys.prefix)/'nksr-provenance.json').read_text())['nksr_git_commit']==sys.argv[1]
import nksr
PY_CHECK
then
 "$PREFIX/bin/python" -m pip install --no-build-isolation "$SOURCE/package/"
fi
"$PREFIX/bin/python" - "$PREFIX" "$COMMIT" <<'PY'
import json, pathlib, sys
(pathlib.Path(sys.argv[1])/'nksr-provenance.json').write_text(json.dumps({'nksr_git_commit':sys.argv[2]}))
import torch, nksr
print('Torch:',torch.__version__,'NKSR:',nksr.__version__,'CUDA:',torch.cuda.is_available())
PY
printf '%s\n' "$PREFIX/bin/python" > "$ROOT/.state/nksr_python.txt"
NKSR_PYTHON="$PREFIX/bin/python" "$ROOT/scripts/check_nksr.sh"
