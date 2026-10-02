#!/usr/bin/env bash
# Optional only: never called by the default/record-only bootstrap.
set -eo pipefail
source "$(dirname "$0")/env.sh"
sudo apt-get install -y pkg-config libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
 gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
 libglib2.0-dev libgirepository1.0-dev libusb-1.0-0-dev libudev-dev libzip-dev \
 libxml2-dev libjson-glib-dev libunwind-dev python3-gi python3-opencv \
 ros-humble-camera-info-manager ros-humble-camera-calibration-parsers ros-humble-cv-bridge ros-humble-image-transport
"$ROOT/.venv/bin/python" "$ROOT/scripts/fetch_dependencies.py" --only gscam2 tiscamera
cmake -S "$ROOT/external/tiscamera" -B "$ROOT/external/tiscamera/build" \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT/.local" \
 -DTCAM_INSTALL_FORCE_PREFIX=ON -DTCAM_BUILD_DOCUMENTATION=OFF -DTCAM_BUILD_WITH_GUI=OFF \
 -DTCAM_ARAVIS_USB_VISION=OFF -DTCAM_BUILD_ARAVIS=OFF -DTCAM_BUILD_V4L2=ON -DTCAM_BUILD_LIBUSB=ON
cmake --build "$ROOT/external/tiscamera/build" -j"$BUILD_JOBS"
cmake --install "$ROOT/external/tiscamera/build"
[[ -e "$ROOT/ros2_ws/src/gscam2" ]] || ln -s ../../external/gscam2 "$ROOT/ros2_ws/src/gscam2"
"$ROOT/scripts/build_ros2.sh" --packages-select gscam2
printf '%s\n' 'Camera software built. USB permissions and the physical DFK pipeline still require validation; see docs/camera_calibration.md.'
