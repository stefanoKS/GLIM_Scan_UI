#!/usr/bin/env bash
# Workstation option; never required by acquisition or GLIM.
set -eo pipefail
source "$(dirname "$0")/env.sh"
INSTALL_SYSTEM=true
case "${1:-}" in
 --skip-system) INSTALL_SYSTEM=false; shift;;
 --help|-h) echo 'Usage: scripts/install_calibration.sh [--skip-system] (use existing system dependencies)'; exit 0;;
esac
[[ $# == 0 ]] || { echo 'Unknown calibration installer option' >&2; exit 2; }
[[ -d "$ROOT/.local/lib/cmake/GTSAM" ]] || { echo 'Build the workstation GLIM dependencies first with scripts/install_glim.sh'; exit 1; }
if [[ "$INSTALL_SYSTEM" == true ]]; then
 sudo apt-get install -y libceres-dev libopencv-dev ros-humble-pcl-ros ros-humble-cv-bridge ros-humble-ament-cmake-python
fi
"$ROOT/.venv/bin/python" "$ROOT/scripts/fetch_dependencies.py" --only direct_visual_lidar_calibration
# Ubuntu 22.04 ships Ceres 2.0; upstream Sophus expects the 2.1 manifold API.
# The pinned GTSAM also uses std pointers instead of upstream Boost pointers.
# Apply once, preserving external edits and refusing conflicting changes.
PATCH="$ROOT/patches/direct-visual-lidar-calibration-native-compat.patch"
if ! git -C "$ROOT/external/direct_visual_lidar_calibration" apply --reverse --check "$PATCH" 2>/dev/null; then
 git -C "$ROOT/external/direct_visual_lidar_calibration" apply --check "$PATCH"
 git -C "$ROOT/external/direct_visual_lidar_calibration" apply "$PATCH"
fi
[[ -e "$ROOT/ros2_ws/src/direct_visual_lidar_calibration" ]] || ln -s ../../external/direct_visual_lidar_calibration "$ROOT/ros2_ws/src/direct_visual_lidar_calibration"
"$ROOT/scripts/build_ros2.sh" --packages-select direct_visual_lidar_calibration
printf '%s\n' 'Only manual initialization + NID are exposed by this project. No SuperGlue packages or models are installed.'
