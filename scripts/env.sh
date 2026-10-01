#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FACTORY_MAPPING_ROOT="$ROOT"
export PATH="$ROOT/.venv/bin:/usr/bin:/bin:$PATH"
# ROS setup scripts are not nounset safe.
set +u
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
if [[ -f "$ROOT/ros2_ws/install/setup.bash" ]]; then source "$ROOT/ros2_ws/install/setup.bash"; fi
export LD_LIBRARY_PATH="$ROOT/.local/lib:$ROOT/.local/lib64:$ROOT/.local/usr/lib/$(gcc -dumpmachine):${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$ROOT/.local:$ROOT/.local/usr:${CMAKE_PREFIX_PATH:-}"
export PYTHONPATH="$ROOT/ui/backend:${PYTHONPATH:-}"
export ROS_LOG_DIR="$ROOT/.state/ros_logs"
mkdir -p "$ROS_LOG_DIR"
