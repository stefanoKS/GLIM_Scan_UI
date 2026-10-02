#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/scripts/check_system.sh"
case "$(uname -m)" in aarch64|x86_64) ;; *) echo 'Unsupported architecture'; exit 1;; esac
[[ -f /opt/ros/humble/setup.bash ]] || { echo 'Install ROS 2 Humble first; this script preserves existing ROS installations.'; exit 1; }
# Installs prerequisites only; no OS/JetPack upgrade, no replacement ROS workspace.
sudo apt-get install -y \
 build-essential cmake git python3-venv python3-pip python3-colcon-common-extensions \
 libboost-all-dev libeigen3-dev libomp-dev libmetis-dev libfmt-dev libspdlog-dev \
 libglm-dev libglfw3-dev libassimp-dev zenity libpng-dev libjpeg-dev libpcl-dev libapr1-dev \
 ros-humble-ament-cmake-auto ros-humble-rosbag2 ros-humble-sensor-msgs-py \
 ros-humble-pcl-conversions ros-humble-rosidl-default-generators \
 ros-humble-rclcpp-components ros-humble-cv-bridge ros-humble-image-transport \
 ros-humble-nav-msgs ros-humble-tf2-ros
/usr/bin/python3 -m venv --system-site-packages "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install -r "$ROOT/requirements.lock"
"$ROOT/scripts/install_livox.sh"
"$ROOT/scripts/install_glim.sh"
"$ROOT/scripts/build_ros2.sh" --packages-select factory_mapping_bringup factory_mapping_monitor factory_mapping_preview
