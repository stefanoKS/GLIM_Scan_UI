# Installation: Jetson and workstation

## Fresh checkout

This project builds on Ubuntu 22.04 with ROS 2 Humble. On a Jetson Orin Nano, use JetPack 6 and keep its supplied Ubuntu/ROS installation intact. On a workstation, install ROS 2 Humble using the [official Ubuntu instructions](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) first. The bootstrap expects `/opt/ros/humble/setup.bash` to exist and uses the system ROS Python packages.

Clone the private repository, then from its root run:

```bash
git clone https://github.com/stefanoKS/GLIM_Scan_UI.git
cd GLIM_Scan_UI
scripts/check_system.sh
scripts/bootstrap_jetson.sh
```

The bootstrap installs the project build prerequisites with apt, creates `.venv`, and builds the Livox SDK/driver and GLIM stack. `scripts/fetch_dependencies.py` clones the upstream projects into ignored `external/` directories at the immutable revisions recorded in `dependencies.lock`; GLIM is therefore downloaded and installed automatically by these instructions. The resulting libraries are installed under `.local/` and ROS packages under `ros2_ws/`. No prebuilt GLIM binaries are included in Git, so each machine builds native artifacts for its own architecture. Internet access and sudo for apt are required. Builds can take a long time; set `BUILD_JOBS=1` before bootstrap on a memory-constrained machine.

When bootstrap succeeds, start the UI with `scripts/run_system.sh` and open http://127.0.0.1:8080. Configure the Mid-360 network settings as described below before connecting hardware.

## Tested host

This implementation pass ran on an x86_64 PC, Ubuntu 22.04.5, kernel 6.8.0-138-generic, ROS 2 Humble, approximately 15 GiB RAM. It is not a Jetson. CUDA/nvcc and JetPack were absent. System Python is 3.10.12; interactive `python3` resolves to Conda 3.13.2. The project uses a system-site-packages Python 3.10 virtual environment so `rclpy` remains compatible. See `environment_report.md` for exact outputs and `validation.md` for actual test results.

## New Jetson Orin Nano

Preserve JetPack 6/Ubuntu 22.04/ROS Humble if installed. Copy or clone this repository (excluding `.venv`, `.local`, `external` build trees and `ros2_ws/build/install/log`; native artifacts must be rebuilt for aarch64). Do not reuse the PC's compiled libraries.

```bash
cd factory_mapping
scripts/check_system.sh
uname -m  # must be aarch64 on Jetson
scripts/bootstrap_jetson.sh
```

Bootstrap installs missing distribution prerequisites, builds official pinned source into project `.local`, and uses an isolated ROS workspace. It never upgrades the OS/JetPack or removes another workspace. It needs ordinary sudo for apt prerequisites. There was no passwordless sudo on the test PC; missing small development packages were downloaded from Ubuntu with apt and extracted inside `.local` instead. This local workaround did not replace system packages.

The normal venv setup is `/usr/bin/python3 -m venv --system-site-packages .venv`. On this PC `python3-venv` was absent, so a local virtualenv bootstrap targeted `/usr/bin/python3`. `requirements.txt` pins application packages; `requirements.lock` pins the resolved application and test dependencies for bootstrap.

Upstream `glim_ros2` declares `cv_bridge` and `image_transport` as build dependencies even when camera support is disabled. Bootstrap installs these libraries to satisfy its unmodified manifest. Camera processing is compiled OFF; no camera driver or camera input is active.

## Build controls

- Defaults: two compiler jobs and one colcon package at a time. Use `BUILD_JOBS=1` if RAM is low. Monitor temperatures and memory.
- `BUILD_WITH_MARCH_NATIVE=OFF` and `GTSAM_BUILD_WITH_MARCH_NATIVE=OFF` for all applicable builds. No x86 SIMD options are introduced.
- CUDA is selected by the installed `nvcc` when `USE_CUDA=AUTO` (default). We do not prescribe a CUDA version or desktop GPU architecture. On a CPU-only host, CUDA defaults OFF.
- If CUDA compilation fails, retain its log and rerun `USE_CUDA=OFF scripts/install_glim.sh`; choose `jetson_cpu` for runtime. Do not use a GPU preset with a CPU build.
- `BUILD_VIEWER=ON` builds Iridescence and the official PLY exporter. `BUILD_VIEWER=OFF` is available for an acquisition-only headless Jetson; export later on the workstation.
- `dependencies.lock` pins upstream SHAs and distinguishes x86 build evidence from ARM64 validation. `scripts/fetch_dependencies.py` refuses to reset dirty checkouts.
- `scripts/install_livox.sh`, `scripts/install_glim.sh`, `scripts/build_ros2.sh` can be rerun independently. Scripts preserve existing build directories.

## Network setup

Edit `config/livox/mid360.yaml` for LiDAR IP, topics and ROS domain. `interface: auto` selects the single active physical Ethernet adapter with an IPv4 address on the LiDAR subnet, excluding Wi-Fi and virtual adapters. Ambiguous matches require an explicit interface in Advanced Diagnostics. The host IP is detected; do not put it in the YAML. Assign an Ethernet address in that subnet using the OS network settings. If it changes while running, the app finalizes any active recording, restarts the sensor driver and shows the new address. The app does not change OS network configuration. Preserve Wi-Fi/default-route settings.

Source `scripts/env.sh` for any manual ROS commands. It selects the correct Python, ROS overlay, local library paths and local ROS logs.

## Jetson acceptance checklist

After a fresh native build with `BUILD_VIEWER=ON`:

```bash
scripts/verify_jetson.sh
# Only with the sensor connected and UI stopped:
scripts/verify_jetson.sh --hardware
scripts/process_bag.sh SESSION_ID --preset jetson_cpu
# For a successfully built CUDA stack:
scripts/process_bag.sh SESSION_ID --preset jetson_gpu
```

`verify_jetson.sh` deliberately rejects x86_64. It checks ROS/Python imports, tests, prohibited architecture flags and all five official GLIM binaries. Hardware validation then records a new bag. Finish with a real walking route, multiple submaps, loop/merge acceptance, segmentation save/export, a long recording and thermal/memory observation. These cannot be established by a PC build.

Use `scripts/package_for_jetson.sh` to create a source-only archive from the current commit in `.state/`. It excludes PC binaries, virtual environments and acquisition data. Extract on Jetson, edit its network/interface configuration, and run bootstrap natively. The Jetson acquires bags; it is not required to perform final heavy optimization.

The compile-job default is 1 below ~4 GiB RAM, 2 below ~19 GiB, and 4 on larger hosts, with sequential colcon packages. Override with BUILD_JOBS when appropriate. Runtime OpenMP defaults to two threads (OMP_NUM_THREADS can override); SLAM estimation parameters remain the official baselines.

CUDA setup compiles a tiny runtime probe, queries the installed GPU's compute capability, and passes the resulting SM target to CMake. This avoids compiling a desktop multi-architecture set on Jetson with older CMake. AUTO falls back to CPU if the probe is unavailable; explicit `USE_CUDA=ON` fails with a diagnostic instead. `CUDA_ARCHITECTURES` supports an intentional numeric override for controlled builds. Successful installation records capabilities in `.state/build_capabilities.json` so stale GPU library files cannot mislabel a CPU build.

## Acquisition without GLIM

A fresh Jetson can install just the official Livox driver, recording, monitoring, browser UI and preview:

```bash
scripts/bootstrap_jetson.sh --record-only
scripts/verify_jetson.sh --record-only
scripts/run_system.sh --host 0.0.0.0
```

This profile fetches only Livox-SDK2 and livox_ros_driver2, skips GLIM/GTSAM/Iridescence builds and CUDA probing, and retains the upstream PCL dependencies required by the Livox driver. It does not remove an existing GLIM installation. Choose **START SCAN** in the dashboard; it detects the record-only installation automatically. Use `scripts/verify_jetson.sh --record-only --hardware` for a sensor-connected acquisition test.

For later local GLIM processing, rerun `scripts/bootstrap_jetson.sh` without the option. Alternatively, keep Jetson acquisition-only and copy completed sessions to the PC. The full default installation continues to include all five official GLIM executables.

## Optional camera extension

See [camera acquisition and calibration](camera_calibration.md) for opt-in installation, fixed pipeline setup, one-bag RGB recording, intrinsic import, static calibration datasets and the workstation manual/NID workflow. GLIM remains LiDAR + IMU only. Camera-disabled and record-only workflows above remain supported.

## Orin Nano 8 GB with the optional camera

Use native aarch64 Ubuntu 22.04/JetPack 6 with ROS Humble. A conservative acquisition installation is:

```bash
BUILD_JOBS=1 scripts/bootstrap_jetson.sh --record-only
BUILD_JOBS=1 scripts/install_camera.sh
source scripts/env.sh
python scripts/check_camera_python.py
scripts/verify_jetson.sh --record-only
scripts/check_camera.sh
```

The Imaging Source Python interface comes from system PyGObject (`python3-gi`, `python3-gst-1.0`) and the natively built Tcam typelib, not a pip-only camera SDK. `env.sh` sets the project-local plugin/typelib paths. The venv must use `/usr/bin/python3 --system-site-packages` so ROS, cv_bridge and OpenCV share the distribution ABI. Do not copy the PC venv or use Conda Python. The installer adds the camera USB/video permission rule and video-group membership; reconnect and log out/in afterward.

Keep the configured 720×540 geometry and verify serial/pipeline against the camera attached to that Jetson. Enable camera inclusion in Settings for one bag containing LiDAR, IMU, RGB and CameraInfo. Use RECORD CAMERA for a camera-only bag. Export a completed project from Library and import it on the GLIM workstation. Native ARM64 build, sustained recording, USB throughput and thermal/memory acceptance still require the actual Jetson; PC tests cannot establish them.
