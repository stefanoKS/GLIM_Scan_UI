#!/usr/bin/env bash
# ROS, numpy, OpenCV and cv_bridge are supplied by the existing camera environment.
set -eo pipefail
source "$(dirname "$0")/env.sh"
"$ROOT/.venv/bin/python" -m pip install pyrealsense2==2.58.1.10581
"$ROOT/.venv/bin/python" -c 'import pyrealsense2, cv2, rclpy; from sensor_msgs.msg import Image, CameraInfo; from cv_bridge import CvBridge'
echo 'D405 RGB support installed. USB access requires the standard librealsense udev rules.'
