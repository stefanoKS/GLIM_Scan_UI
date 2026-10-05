#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RECORD_ONLY=false
[[ "$(uname -m)" != aarch64 ]] || RECORD_ONLY=true
case "${1:-}" in
 --record-only) RECORD_ONLY=true; shift;;
 --workstation) RECORD_ONLY=false; shift;;
 --help|-h) echo "Usage: $0 [--record-only|--workstation]"; exit 0;;
esac
[[ $# == 0 ]] || { echo "Usage: $0 [--record-only|--workstation]" >&2; exit 2; }
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 22.04 ]] || { echo 'This installer requires Ubuntu 22.04 and ROS 2 Humble (JetPack 6 on supported Jetsons).' >&2; exit 1; }
"$ROOT/scripts/check_system.sh"
case "$(uname -m)" in aarch64|x86_64) ;; *) echo 'Unsupported architecture'; exit 1;; esac
[[ -f /opt/ros/humble/setup.bash ]] || { echo 'Install ROS 2 Humble first; this script preserves existing ROS installations.'; exit 1; }
# Installs prerequisites only; no OS/JetPack upgrade, no replacement ROS workspace.
PACKAGES=(
 build-essential cmake git python3-venv python3-pip python3-colcon-common-extensions \
 libpcl-dev libapr1-dev \
 ros-humble-ament-cmake-auto ros-humble-rosbag2 ros-humble-sensor-msgs-py \
 ros-humble-pcl-conversions ros-humble-rosidl-default-generators \
 ros-humble-rclcpp-components ros-humble-rosbag2-storage-default-plugins
 ros-humble-ros2bag ros-humble-rclpy ros-humble-launch-ros
 python3-opencv ros-humble-cv-bridge
)
if [[ "$RECORD_ONLY" == false ]]; then
 PACKAGES+=(libboost-all-dev libeigen3-dev libomp-dev libmetis-dev libfmt-dev libspdlog-dev
  libglm-dev libglfw3-dev libassimp-dev zenity libpng-dev libjpeg-dev
  ros-humble-cv-bridge ros-humble-image-transport ros-humble-nav-msgs ros-humble-tf2-ros)
fi
sudo apt-get install -y "${PACKAGES[@]}"
/usr/bin/python3 -m venv --system-site-packages "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install -r "$ROOT/requirements.lock"
"$ROOT/scripts/install_livox.sh"
if [[ "$RECORD_ONLY" == false ]]; then "$ROOT/scripts/install_glim.sh"; fi
"$ROOT/scripts/build_ros2.sh" --packages-select factory_mapping_bringup factory_mapping_monitor factory_mapping_preview

# Persist only on this host, outside versioned config and session imports.
mkdir -p "$ROOT/.state"
if [[ "$RECORD_ONLY" == true ]]; then MODE=record_only; else MODE=workstation; fi
printf '{"mode":"%s"}\n' "$MODE" > "$ROOT/.state/deployment.json"
echo "Deployment mode: $MODE. For D405 RGB recording, run scripts/install_d405.sh; for LiDAR/IMU only, disable RGB in Settings."
