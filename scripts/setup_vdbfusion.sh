#!/usr/bin/env bash
# Explicit, isolated VDBFusion installation.
#
# This never touches the NKSR environment, the application virtualenv, the ROS
# Python ABI, CUDA/Torch or any locked project dependency. It only creates one
# private virtual environment under $HOME/.cache and records its interpreter in
# .state/vdbfusion_python.txt.
#
# Numpy is pinned to the project's locked version on purpose: the upstream wheel
# is built against the NumPy 1.x ABI and segfaults against NumPy 2.x.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="${VDBFUSION_CACHE:-$HOME/.cache/factory-mapping}"
PREFIX="${VDBFUSION_PREFIX:-$CACHE/vdbfusion-env}"
VERSION="${VDBFUSION_VERSION:-0.1.6}"
BASE_PYTHON="${VDBFUSION_BASE_PYTHON:-/usr/bin/python3.10}"

# Upstream publishes manylinux x86_64 wheels only. A Jetson recording-only
# installation must never gain VDBFusion (guardrail 12).
if [[ "$(uname -s)/$(uname -m)" != Linux/x86_64 ]]; then
  echo 'VDBFusion is supported on Linux x86_64 only; this host is recording-only.' >&2
  echo "Refusing to install on $(uname -s)/$(uname -m)." >&2
  exit 1
fi
if [[ -f "$ROOT/.state/deployment.json" ]] && grep -q 'record_only' "$ROOT/.state/deployment.json"; then
  echo 'This installation is configured as recording-only; VDBFusion is not installed here.' >&2
  exit 1
fi
if [[ ! -x "$BASE_PYTHON" ]]; then
  echo "Base interpreter $BASE_PYTHON is missing. Set VDBFUSION_BASE_PYTHON to a Python 3.10 interpreter." >&2
  exit 1
fi
PY_MINOR="$("$BASE_PYTHON" -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
if [[ "$PY_MINOR" != 3.10 ]]; then
  echo "VDBFusion $VERSION publishes wheels for Python 3.6-3.10 only; $BASE_PYTHON is Python $PY_MINOR." >&2
  echo 'Set VDBFUSION_BASE_PYTHON to a Python 3.10 interpreter.' >&2
  exit 1
fi

mkdir -p "$CACHE" "$ROOT/.state"
if [[ ! -x "$PREFIX/bin/python" ]]; then
  echo "Creating isolated VDBFusion environment at $PREFIX"
  # --system-site-packages keeps the base interpreter's standard environment
  # reachable while the versions pinned below always win for this environment.
  "$BASE_PYTHON" -m venv --system-site-packages "$PREFIX"
fi
PIP=("$PREFIX/bin/python" -m pip)
echo "Installing pinned VDBFusion runtime into $PREFIX"
"${PIP[@]}" install --upgrade --quiet pip
# NumPy 1.26.4 is the project-locked version, so no new NumPy enters the project.
"${PIP[@]}" install --quiet 'numpy==1.26.4' 'scipy==1.15.2' 'plyfile==1.1.2' 'psutil==7.0.0'
# The upstream wheel statically links OpenVDB and TBB, so it cannot conflict with
# the ROS Humble system TBB. Only wheels are accepted; never build from source here.
"${PIP[@]}" install --quiet --only-binary=:all: "vdbfusion==$VERSION"

"$PREFIX/bin/python" - "$PREFIX" "$VERSION" "$BASE_PYTHON" <<'PY'
import json, pathlib, sys, platform
import numpy, scipy, plyfile, psutil
import vdbfusion
prefix, version, base = sys.argv[1], sys.argv[2], sys.argv[3]
module = pathlib.Path(vdbfusion.__file__).resolve()
provenance = dict(vdbfusion_version=getattr(vdbfusion, '__version__', None), requested_version=version,
                  module=str(module), module_bytes=module.stat().st_size,
                  numpy_version=numpy.__version__, scipy_version=scipy.__version__,
                  plyfile_version=getattr(plyfile, '__version__', None), psutil_version=psutil.__version__,
                  base_python=base, platform=platform.platform(),
                  upstream='https://github.com/PRBonn/vdbfusion',
                  native_libraries='OpenVDB and TBB statically linked by the upstream wheel')
(pathlib.Path(prefix) / 'vdbfusion-provenance.json').write_text(json.dumps(provenance, indent=2))
if numpy.__version__.split('.')[0] != '1':
    raise SystemExit(f'The upstream wheel requires the NumPy 1.x ABI but {numpy.__version__} is installed')
print('VDBFusion', provenance['vdbfusion_version'], 'NumPy', numpy.__version__, 'SciPy', scipy.__version__)
PY

printf '%s\n' "$PREFIX/bin/python" > "$ROOT/.state/vdbfusion_python.txt"
echo "Recorded $PREFIX/bin/python in .state/vdbfusion_python.txt"
VDBFUSION_PYTHON="$PREFIX/bin/python" "$ROOT/scripts/check_vdbfusion.sh"
