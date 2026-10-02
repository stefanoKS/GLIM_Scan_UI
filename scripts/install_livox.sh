#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
"$ROOT/.venv/bin/python" "$ROOT/scripts/fetch_dependencies.py" --only Livox-SDK2 livox_ros_driver2
cmake -S "$ROOT/external/Livox-SDK2" -B "$ROOT/external/Livox-SDK2/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT/.local"
cmake --build "$ROOT/external/Livox-SDK2/build" -j"${BUILD_JOBS:-2}"
cmake --install "$ROOT/external/Livox-SDK2/build"
cp "$ROOT/external/livox_ros_driver2/package_ROS2.xml" "$ROOT/external/livox_ros_driver2/package.xml"
[[ -e "$ROOT/ros2_ws/src/livox_ros_driver2" ]] || ln -s ../../external/livox_ros_driver2 "$ROOT/ros2_ws/src/livox_ros_driver2"
"$ROOT/scripts/build_ros2.sh" --packages-select livox_ros_driver2
