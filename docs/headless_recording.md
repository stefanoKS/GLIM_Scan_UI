# Headless raw recording

```bash
./scripts/headless_record.sh
# Optional: default countdown is 10 seconds
./scripts/headless_record.sh --countdown 5 --camera-serial SDK_SERIAL
```

Requires ROS2 Humble, `livox_ros_driver2`, `realsense2_camera`, SQLite rosbag2,
`rclpy`, `sensor_msgs`, `pyrealsense2`, PyYAML and psutil in the sourced environment.
The output filesystem needs at least 5 GiB free (or the configured minimum).
Ensure `data/sessions` resides on the NVMe before recording.

SDK enumeration selects the only D405, ignoring other models. Zero D405s or
multiple D405s without an explicit SDK serial abort. Camera YAML is never edited.
Session `sensor_info.json` saves identity and calibration match/mismatch; mismatch
warns and continues. USB descriptor serial is optional diagnostic information.

Records only `/livox/lidar`, `/livox/imu`, `/camera/image_raw`, and
`/camera/camera_info` (topic names follow repo configs) to uncompressed SQLite in
`data/sessions/YYYYMMDD_HHMMSS_headless/raw_bag`.
UI camera profiles also request 5 FPS; restart the UI backend to reload them.
D405 runs through the official C++ driver at **1280×720, 5 FPS, raw RGB**.
No UI, GLIM, preview, rectification, depth, IR, alignment, pointcloud, or filters.
No compressed image subscribers are created, so lazy image-transport encoders do
no work. Only CameraInfo is subscribed during preflight; never full RGB images.

Native driver stamps are preserved. Unlike the old Python publisher's host
receipt timestamp, the [C++ wrapper](https://github.com/realsenseai/realsense-ros)
converts SDK frame timestamps into ROS time. No online synchronization is added;
verify sensor clock alignment offline. Raw images retain lens distortion: use the
recorded CameraInfo, not the previous rectified-image intrinsics, for colorization.

Press **ENTER** or **Ctrl+C** to stop. SIGTERM also requests graceful shutdown.
The recorder sends rosbag SIGINT and waits for finalization before stopping drivers;
it never force-kills a bag. Logs are in the session's `logs` directory.

Validate on Jetson: record for 30–60 seconds, then press ENTER. Inspect the printed
absolute bag path:

```bash
source scripts/env.sh
ros2 bag info /absolute/path/to/session/raw_bag
```

Check all four topics and divide RGB message count by bag duration in seconds.
The target is close to 5 FPS while recording MID-360; no hardware performance
claim is made by local syntax/preflight checks.
