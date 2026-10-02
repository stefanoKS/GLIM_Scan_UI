# Verified upstream contracts

Inspected official source pinned in `dependencies.lock` on 2026-10-01.

- [GLIM source installation](https://koide3.github.io/glim/installation.html): GTSAM 4.3a0, gtsam_points, optional Iridescence. The scripts build into `.local`, disable native architecture flags, and default to two compiler jobs. CUDA is detected; use `USE_CUDA=OFF` for an explicit CPU fallback.
- [GLIM ROS2](https://github.com/koide3/glim_ros2): package `glim_ros`; executables `glim_rosnode` and `glim_rosbag`. Both accept ROS parameters `config_path` and `dump_path`; the bag executable also accepts `auto_quit`. No permanent `/tmp/dump` is needed.
- `glim_ros2/src/offline_viewer.cpp` implements `offline_viewer DUMP --export_path OUTPUT.ply --config_path CONFIG`. It is a real upstream interface, but **still requires an OpenGL display**. Its export path exits when finished. XYZ and available intensity are written by GLIM, using optimized submap poses.
- [Livox driver](https://github.com/Livox-SDK/livox_ros_driver2): `xfer_format=0`, `multi_topic=0` publish PointCloud2 on `/livox/lidar` and Imu on `/livox/imu`. These are source-discovered defaults, verified on the connected device, and remapped at the driver boundary from YAML. `/PointCloud2` and `/Imu` are message types in the request, not the driver's actual default topic names.
- Pinned Livox PointCloud2 has `timestamp` float64 in absolute nanoseconds (`pub_handler.cpp` assigns packet time plus point interval to the misleadingly named `offset_time`; `lddc.cpp` forwards it). This was verified by reading the recorded bag: point times are about 1.79e18 while header time is about 1.79e9 seconds. Project presets explicitly set per-point time scale `1e-9` and absolute time. IMU acceleration is in g; GLIM uses `acc_scale=9.80665`.
- [Mid-360 manufacturer manual](https://terra-1-g.djicdn.com/851d20f7b9f64838a34cd02351370894/Livox/Livox_Mid-360_User_Manual_EN.pdf), IMU data section: IMU position in LiDAR coordinates is `[11,23.29,-44.12]` mm, with aligned axes. GLIM's IMU-to-LiDAR transform therefore uses `[0.011,0.02329,-0.04412,0,0,0,1]`.
- The official driver fixes its IMU header to `livox_frame`, despite its configurable LiDAR frame. We expose this limitation and disable GLIM's same-frame IMU→LiDAR TF publication. This does not disable the numerical extrinsic used for estimation.
- We bypass upstream `build.sh` because it deletes workspace build/install trees. We prepare its ROS2 package manifest and invoke colcon with `ROS_EDITION=ROS2`, `DISTRO_ROS=humble` directly.

# Preset differences

All JSON starts from pinned upstream `glim/config/*.json`. Comments are removed by JSON5 parsing; numerical estimation defaults remain unchanged except sensor integration above. `jetson_cpu` selects upstream CPU odometry, passthrough submapping and pose graph global mapping. `jetson_gpu` and `offline_quality` select the upstream GPU defaults; the latter is an independently editable preset, not an unvalidated accuracy claim. GPU presets require a CUDA build. Viewer extensions are removed for unattended operation. File logging is disabled because all stdout/stderr is stored per session/job. `scripts/make_presets.py` regenerates presets and overwrites local preset edits; do not run it routinely.

# Loop detection

`loop_closure.enabled: false` means no extra place-recognition module. GLIM's own geometric constraints remain active; OFF does not disable all backend geometric loop constraints. `scan_context` reports UNAVAILABLE and is rejected at live launch until compatible glim_ext builds and real loop tests exist. No camera is subscribed by our packages or recorded. Upstream optional camera support is compiled off.

# Offline/live isolation verified on real data

The pinned `glim_rosbag` constructs the same GlimROS node as live mode and spins subscriptions while reading a bag. This can mix current sensor messages into old data. Our wrapper puts offline processing in its own configured ROS domain with localhost-only DDS and remaps live subscription names to a unique unused namespace. Bag topic selection still reads the original names from its snapshot; no bag is rewritten. A regression test replayed a recorded stream in the **same offline domain** while processing the bag and confirmed no timestamp-rewind errors and identical raw-file SHA-256 hashes.

Completed jobs require saved graph/trajectory artifacts and reject critical timestamp-validation messages. A zero process exit alone is not sufficient. Early invalid experimental runs are retained as failed/cancelled with explanatory metadata.
