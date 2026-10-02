#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
[[ "$(uname -m)" == aarch64 ]] || { echo 'This is not aarch64. Jetson validation cannot be claimed on this host.' >&2; exit 1; }
"$ROOT/scripts/check_system.sh" "$ROOT/docs/jetson_validation_environment.md"
"$ROOT/.venv/bin/python" -c 'import rclpy, numpy, fastapi; print("ROS and web Python imports OK")'
"$ROOT/.venv/bin/python" "$ROOT/scripts/check_portability.py"
cd "$ROOT"
"$ROOT/.venv/bin/python" -m pytest -q --basetemp=.state/pytest-jetson
ros2 pkg executables glim_ros
for executable in glim_rosnode glim_rosbag offline_viewer map_editor validator_node; do
 [[ -x "$ROOT/ros2_ws/install/glim_ros/lib/glim_ros/$executable" ]] || { echo "Missing official tool: $executable. Rebuild with BUILD_VIEWER=ON."; exit 1; }
done
if [[ "${1:-}" == --hardware ]]; then
 "$ROOT/scripts/record_test.sh" --name jetson_acceptance --seconds 30
 echo 'Next: process the printed session with scripts/process_bag.sh SESSION_ID --preset jetson_cpu (or jetson_gpu for a CUDA build).'
fi
echo 'ARM64 software checks passed. A walking-route, loop closure and thermal soak test are still required.'
