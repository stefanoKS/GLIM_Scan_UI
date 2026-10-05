# Installation: recording Jetson and processing workstation

These instructions describe the camera-selection build `ba70f6d`. The Jetson records raw sensor data. GLIM mapping, native map tools and NKSR surfacing belong on the workstation.

## Requirements and fresh checkout

Use Ubuntu 22.04 with ROS 2 Humble installed at `/opt/ros/humble/setup.bash`. The application uses `/usr/bin/python3` (Python 3.10) and a system-site-packages venv so ROS, OpenCV and cv_bridge share compatible distribution packages. Do not use Conda Python for the dashboard or ROS processes. NKSR has its own separate environment on the workstation.

The supported Jetson image is JetPack 6 / Ubuntu 22.04 on Orin hardware. The installer rejects other Ubuntu versions; JetPack 5/7 and older Nano/TX2 configurations are not covered by these instructions. It preserves the installed OS/JetPack and existing ROS workspaces. Install ROS separately first if it is missing; see the [ROS Humble installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).

```bash
git clone https://github.com/stefanoKS/GLIM_Scan_UI.git
cd GLIM_Scan_UI
```

All commands below run from this checkout's root. Bootstrap needs network access and sudo for apt packages. Never copy `.venv`, `.local`, external build trees or `ros2_ws/build`, `install`, `log` from x86_64 to ARM64.

## Recording Jetson

```bash
BUILD_JOBS=1 scripts/bootstrap_jetson.sh --record-only
```

This installs the recording prerequisites, creates `.venv`, fetches only pinned Livox-SDK2/livox_ros_driver2 source, and builds the driver plus project monitoring/preview/bringup packages. GLIM, GTSAM, Iridescence, CUDA probing and NKSR installation are skipped. PCL remains a required Livox dependency. OpenCV and cv_bridge are included for camera support, but vendor camera SDKs/publishers are installed separately.

The installer saves `{"mode":"record_only"}` in `.state/deployment.json`. Backend controls enforce this role even if old GLIM or NKSR installations remain. Live mapping, automatic processing, native map tools, reconstruction preparation and NKSR jobs are unavailable. Project import/export never changes this policy.

No flag on ARM64 also selects recording-only. Rerunning bootstrap without a flag on Jetson does **not** enable GLIM processing.

## Processing workstation

```bash
scripts/bootstrap_jetson.sh --workstation
```

Despite the script name, this is the full workstation installer. It builds the acquisition stack and the pinned GTSAM, gtsam_points, GLIM, glim_ros2 and viewer dependencies, saving `{"mode":"workstation"}` locally. Upstreams are fetched into ignored `external/` directories using `dependencies.lock`; install trees are local to `.local/` and `ros2_ws/`. GLIM does not need a separate manual clone.

The GLIM installer probes the available GPU/CUDA toolchain. AUTO selects a CUDA build when supported and otherwise falls back to CPU. `USE_CUDA=OFF scripts/bootstrap_jetson.sh --workstation` explicitly selects CPU. NKSR is not part of bootstrap: follow [NKSR setup](nksr.md) on a compatible workstation if surfacing is needed. Native editors/export require a working server desktop/OpenGL environment.

## Camera stacks

Auto selection and Include RGB are enabled by default. Install the stack for each camera you intend to use:

```bash
# Intel RealSense D405; requires standard librealsense USB permissions.
scripts/install_d405.sh

# Imaging Source DFK 33UX287; optional alternative/fallback.
BUILD_JOBS=1 scripts/install_camera.sh
```

D405 uses `pyrealsense2==2.58.2.10647` and targets 1280×720 at 30 FPS. The installer checks SDK/ROS/OpenCV imports but does not install librealsense udev rules. DFK keeps 720×540 at 15 FPS and uses native gscam2/tiscamera, GStreamer, PyGObject and the Tcam typelib. Its installer adds its USB/video rule and video-group membership; reconnect and log out/in as instructed. Do not run the dashboard as root.

Check configured SDK/USB serials in `config/camera/d405.yaml` and the DFK serial/pipeline in `config/camera/dfk33ux287.yaml`. Presence detection is specific to the configured devices. Install both stacks if fallback between both cameras is required. A saved camera choice is a preference, not a requirement: failure to obtain either stream allows a warned LiDAR/IMU-only scan. Camera-only capture and calibration still require a usable camera.

See [camera setup and calibration](camera_calibration.md) for USB checks, factory versus measured intrinsics and camera-specific alignment files. Heavy calibration tools are installed on the workstation with `scripts/install_calibration.sh`; they are not needed for ordinary recording.

## Network and dashboard

The tracked LiDAR IP is `192.168.1.120`, ROS domain 41, point topic `/livox/lidar` and IMU topic `/livox/imu`. Assign the host Ethernet interface an address on the LiDAR subnet using OS network settings. `interface: auto` in `config/livox/mid360.yaml` selects one active physical Ethernet adapter on that subnet and discovers its host address. Wi-Fi and virtual adapters are excluded; ambiguous matches need an explicit interface in Advanced Diagnostics.

The app does not assign IP addresses, change sensor firmware or modify the default route. If the detected host address changes, active sensor acquisition is finalized before the driver is restarted. Development-PC addresses and interface names in historical reports are examples, not settings to copy.

```bash
scripts/run_system.sh
```

Open [the local dashboard](http://127.0.0.1:8080). For access from a trusted LAN, use `scripts/run_system.sh --host 0.0.0.0` and the host's actual IP on port 8080. There is no login; keep the service off the public Internet.

Source `scripts/env.sh` before manual Python/ROS commands. It selects the application venv, ROS overlay, local libraries/plugins and `.state/ros_logs`. Start/stop and recording scripts already source it.

## Updating an existing checkout

1. Finalize any capture and close the dashboard with Ctrl-C. Let managed subprocesses exit; resolve [orphan recovery](troubleshooting.md#previous-sensor-process-is-still-running) first if needed.
2. Preserve local camera serials, calibration files and network edits, then update the source using your normal Git workflow. Do not replace local changes blindly.
3. Rerun the explicit bootstrap for the host's role when dependencies/build scripts changed. Rerun `scripts/install_d405.sh` when moving from the older SDK pin to 2.58.2.10647; rebuild the DFK stack only when needed.
4. Restart and verify sensor streams before accepting a recording.

Keep `.state/camera_profile.json` and `.state/camera_enabled.json`; there is no migration requiring deletion. Old D405/DFK profile values remain preferences with fallback. Unknown/malformed profile data falls back to Auto with a warning; saving a preference replaces that file. Calibration history and completed sessions remain intact.

## Build controls

- `BUILD_JOBS` defaults by RAM: 1 below roughly 4.5 GB, 2 below roughly 20 GB, otherwise 4. Colcon builds one package at a time. `BUILD_JOBS=1` is useful on memory-constrained hosts.
- Runtime OpenMP defaults to two threads. Build scripts disable native-architecture tuning rather than copying x86 artifacts to ARM64.
- Workstation GLIM: `USE_CUDA=AUTO` probes the device/toolchain; `USE_CUDA=OFF` selects CPU; explicit `USE_CUDA=ON` fails if detection fails. `CUDA_ARCHITECTURES` is available for an intentional supported numeric override.
- Viewer support defaults ON in `install_glim.sh`. `BUILD_VIEWER=OFF` does not provide native map editing/export. Recording-only Jetson bootstrap skips GLIM entirely.
- `scripts/install_livox.sh`, `scripts/install_glim.sh` and `scripts/build_ros2.sh` can rebuild their components. Installing GLIM alone does not change a recording-only deployment policy.
- `scripts/fetch_dependencies.py` refuses to reset dirty upstream checkouts. Build/install directories are reused, not routinely deleted.

## Verification and source transfer

On the Jetson, with the dashboard stopped:

```bash
scripts/verify_jetson.sh --record-only
# With Mid-360 connected:
scripts/verify_jetson.sh --record-only --hardware
```

The verifier rejects x86_64, records the environment, checks Python/ROS imports, audits compiled architecture flags, runs tests and checks the Livox driver. In recording-only mode it does not require GLIM binaries. `--hardware` creates a new 30-second bag. The separate `--workstation` verifier option checks GLIM binaries on ARM64 processing installations; it is not needed for this recording Jetson.

For USB speed, SDK version, selection/fallback, actual geometry/FPS, CameraInfo and startup timings, follow the [four camera cases](camera_auto_resolution.md#jetson-commands). `scripts/verify_jetson.sh --record-only --camera-case=d405` runs the corresponding diagnostic after software checks.

Then record for the intended route duration on the target storage, inspect topic counts, dropped/slow streams, memory and temperature, and transfer the completed project to the workstation for processing. The latest recorded x86 software tests are in [validation](validation.md); they do not establish native Jetson performance.

`scripts/package_for_jetson.sh` creates `.state/factory_mapping-source.tar.gz` from committed source only and requires a clean Git working tree. It excludes local binaries, environments and data. Extract on Jetson, configure that host and build natively. Transfer recording projects separately through Library export/import.
