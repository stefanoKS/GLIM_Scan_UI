#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
"$ROOT/.venv/bin/python" "$ROOT/scripts/fetch_dependencies.py"
# CUDA is selected from the actual local compiler, never a desktop version string.
if [[ "${USE_CUDA:-AUTO}" == AUTO ]]; then
 if command -v nvcc >/dev/null; then USE_CUDA=ON; else USE_CUDA=OFF; fi
fi
export USE_CUDA
export BUILD_VIEWER="${BUILD_VIEWER:-ON}"
JOBS="${BUILD_JOBS:-2}"
build() {
 local name="$1"; shift
 cmake -S "$ROOT/external/$name" -B "$ROOT/external/$name/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT/.local" "$@"
 cmake --build "$ROOT/external/$name/build" -j"$JOBS"
 cmake --install "$ROOT/external/$name/build"
}
build gtsam -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_TESTS=OFF -DGTSAM_WITH_TBB=OFF -DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF
if [[ "$BUILD_VIEWER" == ON ]]; then build iridescence -DBUILD_WITH_MARCH_NATIVE=OFF; fi
if ! build gtsam_points -DBUILD_WITH_CUDA="$USE_CUDA" -DBUILD_WITH_MARCH_NATIVE=OFF; then
 echo 'Build failed. Inspect the output; rerun USE_CUDA=OFF scripts/install_glim.sh for a CPU build.' >&2; exit 1
fi
for name in glim glim_ros2; do [[ -e "$ROOT/ros2_ws/src/$name" ]] || ln -s "../../external/$name" "$ROOT/ros2_ws/src/$name"; done
"$ROOT/scripts/build_ros2.sh" --packages-select glim glim_ros
