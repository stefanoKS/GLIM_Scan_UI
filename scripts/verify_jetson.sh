#!/usr/bin/env bash
set -eo pipefail
RECORD_ONLY=true
HARDWARE=false
CAMERA_CASE=
for arg in "$@"; do
 case "$arg" in
  --record-only) RECORD_ONLY=true;;
  --workstation) RECORD_ONLY=false;;
  --camera-case=*) CAMERA_CASE="${arg#*=}";;
  --hardware) HARDWARE=true;;
  --help|-h) echo "Usage: $0 [--record-only|--workstation] [--hardware] [--camera-case=d405|dfk|both|neither]"; exit 0;;
  *) echo "Unknown option: $arg" >&2; exit 2;;
 esac
done
source "$(dirname "$0")/env.sh"
[[ "$(uname -m)" == aarch64 ]] || { echo 'This is not aarch64. Jetson validation cannot be claimed on this host.' >&2; exit 1; }
"$ROOT/scripts/check_system.sh" "$ROOT/docs/jetson_validation_environment.md"
"$ROOT/.venv/bin/python" -c 'import rclpy, numpy, fastapi; print("ROS and web Python imports OK")'
"$ROOT/.venv/bin/python" "$ROOT/scripts/check_portability.py"
cd "$ROOT"
"$ROOT/.venv/bin/python" -m pytest -q --basetemp=.state/pytest-jetson
[[ -x "$ROOT/ros2_ws/install/livox_ros_driver2/lib/livox_ros_driver2/livox_ros_driver2_node" ]] || { echo 'Missing Livox driver; run scripts/install_livox.sh'; exit 1; }
if [[ "$RECORD_ONLY" == false ]]; then
ros2 pkg executables glim_ros
for executable in glim_rosnode glim_rosbag offline_viewer map_editor validator_node; do
 [[ -x "$ROOT/ros2_ws/install/glim_ros/lib/glim_ros/$executable" ]] || { echo "Missing official tool: $executable. Rebuild with BUILD_VIEWER=ON."; exit 1; }
done
fi
if [[ -n "$CAMERA_CASE" ]]; then
 "$ROOT/.venv/bin/python" "$ROOT/scripts/diagnose_camera.py" --case "$CAMERA_CASE"
fi
if [[ "$HARDWARE" == true ]]; then
 "$ROOT/scripts/record_test.sh" --name jetson_acceptance --seconds 30
 echo 'Next: transfer the completed session to the workstation for mapping and surfacing. Never process it on the recording Jetson.'
fi
if [[ "$RECORD_ONLY" == true ]]; then
 echo 'ARM64 acquisition software checks passed. Sustained LiDAR/IMU recording and thermal/memory checks still require the sensor.'
else
 echo 'ARM64 software checks passed. A walking-route, loop closure and thermal soak test are still required.'
fi
