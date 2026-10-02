#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
"$ROOT/.venv/bin/python" "$ROOT/scripts/fetch_dependencies.py"
# Query the installed runtime/device rather than compiling desktop SM targets.
REQUESTED_CUDA="${USE_CUDA:-AUTO}"
CUDA_ARGS=()
if [[ "$REQUESTED_CUDA" == OFF ]]; then USE_CUDA=OFF
elif CUDA_ARCHITECTURES="$("$ROOT/.venv/bin/python" "$ROOT/scripts/detect_cuda.py")"; then
 USE_CUDA=ON; export CUDA_ARCHITECTURES; CUDA_ARGS=("-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES")
elif [[ "$REQUESTED_CUDA" == AUTO ]]; then
 echo 'CUDA runtime probe unavailable; selecting the CPU build.' >&2; USE_CUDA=OFF
else echo 'Explicit CUDA build requested, but GPU/toolchain detection failed.' >&2; exit 1
fi
export USE_CUDA
export BUILD_VIEWER="${BUILD_VIEWER:-ON}"
JOBS="${BUILD_JOBS:-2}"
build() {
 local name="$1"; shift
 cmake -S "$ROOT/external/$name" -B "$ROOT/external/$name/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT/.local" "$@" || return "$?"
 cmake --build "$ROOT/external/$name/build" -j"$JOBS" || return "$?"
 cmake --install "$ROOT/external/$name/build"
}
build gtsam -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_TESTS=OFF -DGTSAM_WITH_TBB=OFF -DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF
if [[ "$BUILD_VIEWER" == ON ]]; then build iridescence -DBUILD_WITH_MARCH_NATIVE=OFF; fi
if ! build gtsam_points -DBUILD_WITH_CUDA="$USE_CUDA" -DBUILD_WITH_MARCH_NATIVE=OFF "${CUDA_ARGS[@]}"; then
 echo 'Build failed. Inspect the output; rerun USE_CUDA=OFF scripts/install_glim.sh for a CPU build.' >&2; exit 1
fi
for name in glim glim_ros2; do [[ -e "$ROOT/ros2_ws/src/$name" ]] || ln -s "../../external/$name" "$ROOT/ros2_ws/src/$name"; done
"$ROOT/scripts/build_ros2.sh" --packages-select glim glim_ros

"$ROOT/.venv/bin/python" - <<'PYINFO'
import json,os,platform
from pathlib import Path
root=Path(os.environ['FACTORY_MAPPING_ROOT'])
(root/'.state/build_capabilities.json').write_text(json.dumps({'architecture':platform.machine(),'cuda':os.environ['USE_CUDA']=='ON','viewer':os.environ['BUILD_VIEWER']=='ON'},indent=2)+'\n')
PYINFO
