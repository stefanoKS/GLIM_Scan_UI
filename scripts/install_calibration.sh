#!/usr/bin/env bash
# Workstation option; never required by acquisition or GLIM.
set -eo pipefail
source "$(dirname "$0")/env.sh"
[[ -d "$ROOT/.local/lib/cmake/GTSAM" ]] || { echo 'Build the workstation GLIM dependencies first with scripts/install_glim.sh'; exit 1; }
sudo apt-get install -y libceres-dev libopencv-dev ros-humble-pcl-ros ros-humble-cv-bridge ros-humble-ament-cmake-python
"$ROOT/.venv/bin/python" "$ROOT/scripts/fetch_dependencies.py" --only direct_visual_lidar_calibration
[[ -e "$ROOT/ros2_ws/src/direct_visual_lidar_calibration" ]] || ln -s ../../external/direct_visual_lidar_calibration "$ROOT/ros2_ws/src/direct_visual_lidar_calibration"
"$ROOT/scripts/build_ros2.sh" --packages-select direct_visual_lidar_calibration
printf '%s\n' 'Only manual initialization + NID are exposed by this project. No SuperGlue packages or models are installed.'
