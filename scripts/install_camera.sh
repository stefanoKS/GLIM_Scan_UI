#!/usr/bin/env bash
# Optional only: never called by the default/record-only bootstrap.
set -eo pipefail
source "$(dirname "$0")/env.sh"
INSTALL_SYSTEM=true
case "${1:-}" in
 --skip-system) INSTALL_SYSTEM=false; shift;;
 --help|-h) echo 'Usage: scripts/install_camera.sh [--skip-system] (use existing system dependencies/USB permissions)'; exit 0;;
esac
[[ $# == 0 ]] || { echo 'Unknown camera installer option' >&2; exit 2; }
case "$(uname -m)" in aarch64|x86_64) ;; *) echo 'Supported: native ARM64 Jetson or x86_64 Ubuntu 22.04'; exit 1;; esac
[[ -x "$ROOT/.venv/bin/python" ]] || { echo 'Run bootstrap_jetson.sh [--record-only] first'; exit 1; }
if [[ "$INSTALL_SYSTEM" == true ]]; then
 sudo apt-get install -y pkg-config libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  libglib2.0-dev libgirepository1.0-dev libusb-1.0-0-dev libudev-dev libzip-dev \
  libxml2-dev libjson-glib-dev libunwind-dev uuid-dev gobject-introspection \
  python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 python3-opencv \
  ros-humble-camera-info-manager ros-humble-camera-calibration-parsers ros-humble-cv-bridge ros-humble-image-transport
fi
"$ROOT/.venv/bin/python" "$ROOT/scripts/fetch_dependencies.py" --only gscam2 tiscamera
PATCH="$ROOT/patches/tiscamera-path-spaces.patch"
if ! git -C "$ROOT/external/tiscamera" apply --reverse --check "$PATCH" 2>/dev/null; then
 git -C "$ROOT/external/tiscamera" apply --check "$PATCH"
 git -C "$ROOT/external/tiscamera" apply "$PATCH"
fi
cmake -S "$ROOT/external/tiscamera" -B "$ROOT/external/tiscamera/build" \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT/.local" \
 -DTCAM_INSTALL_FORCE_PREFIX=ON -DTCAM_BUILD_DOCUMENTATION=OFF -DTCAM_BUILD_WITH_GUI=OFF \
 -DTCAM_ARAVIS_USB_VISION=OFF -DTCAM_BUILD_ARAVIS=OFF -DTCAM_BUILD_V4L2=ON -DTCAM_BUILD_LIBUSB=ON
cmake --build "$ROOT/external/tiscamera/build" -j"$BUILD_JOBS"
cmake --install "$ROOT/external/tiscamera/build"
[[ -e "$ROOT/ros2_ws/src/gscam2" ]] || ln -s ../../external/gscam2 "$ROOT/ros2_ws/src/gscam2"
"$ROOT/scripts/build_ros2.sh" --packages-select gscam2 --cmake-clean-cache
if [[ "$INSTALL_SYSTEM" == true ]]; then
 # Headless Jetson users need group access, without granting all users USB writes.
 cat > "$ROOT/.state/80-factory-imaging-source.rules" <<'RULES'
# Permission rule for both libusb and V4L2. No firmware changes.
SUBSYSTEM=="usb", ATTRS{idVendor}=="199e", GROUP="video", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="199e", GROUP="video", MODE="0660", TAG+="uaccess"
ACTION=="add", SUBSYSTEM=="usb", ATTR{idVendor}=="199e", TEST=="power/control", ATTR{power/control}="on"
RULES
 sudo install -m 0644 "$ROOT/.state/80-factory-imaging-source.rules" /etc/udev/rules.d/80-factory-imaging-source.rules
 sudo usermod -a -G video "$(id -un)"
 sudo udevadm control --reload-rules
 echo 'Reconnect the camera and log out/in for video group membership on headless hosts.'
fi
"$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_python.py"
printf '%s\n' 'Camera software built. Validate the attached camera with scripts/check_camera.sh; see docs/camera_calibration.md.'
