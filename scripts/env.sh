#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FACTORY_MAPPING_ROOT="$ROOT"
if [[ -z "${BUILD_JOBS:-}" ]]; then
 BUILD_JOBS="$(awk '/MemTotal/ {print ($2 < 4500000 ? 1 : ($2 < 20000000 ? 2 : 4))}' /proc/meminfo)"
fi
export BUILD_JOBS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
if [[ -x /usr/local/cuda/bin/nvcc ]]; then export PATH="/usr/local/cuda/bin:$PATH"; fi
export PATH="$ROOT/.venv/bin:$ROOT/.local/bin:/usr/bin:/bin:$PATH"
# ROS setup scripts are not nounset safe.
set +u
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
if [[ -f "$ROOT/ros2_ws/install/setup.bash" ]]; then source "$ROOT/ros2_ws/install/setup.bash"; fi
export LD_LIBRARY_PATH="$ROOT/.local/lib:$ROOT/.local/lib64:$ROOT/.local/usr/lib/$(gcc -dumpmachine):${LD_LIBRARY_PATH:-}"
export CPATH="$ROOT/.local/usr/include:${CPATH:-}"
export LIBRARY_PATH="$ROOT/.local/lib:$ROOT/.local/usr/lib/$(gcc -dumpmachine):${LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$ROOT/.local:$ROOT/.local/usr:${CMAKE_PREFIX_PATH:-}"
export PYTHONPATH="$ROOT/ui/backend:${PYTHONPATH:-}"
export ROS_LOG_DIR="$ROOT/.state/ros_logs"
mkdir -p "$ROS_LOG_DIR"

# Optional project-local GStreamer plugins; does not replace system plugins.
export GST_PLUGIN_PATH="$ROOT/.local/lib/gstreamer-1.0:$ROOT/.local/lib/$(gcc -dumpmachine)/gstreamer-1.0:${GST_PLUGIN_PATH:-}"
export GI_TYPELIB_PATH="$ROOT/.local/lib/girepository-1.0:$ROOT/.local/lib/$(gcc -dumpmachine)/girepository-1.0:${GI_TYPELIB_PATH:-}"
export PKG_CONFIG_PATH="$ROOT/.local/lib/pkgconfig:$ROOT/.local/lib/$(gcc -dumpmachine)/pkgconfig:${PKG_CONFIG_PATH:-}"
