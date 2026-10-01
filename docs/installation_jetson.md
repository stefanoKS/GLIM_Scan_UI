# Installation: Jetson and workstation

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

The normal venv setup is `/usr/bin/python3 -m venv --system-site-packages .venv`. On this PC `python3-venv` was absent, so a local virtualenv bootstrap targeted `/usr/bin/python3`. `requirements.txt` pins application packages; `.state/python-freeze.txt` records the resolved PC environment.

## Build controls

- Defaults: two compiler jobs and one colcon package at a time. Use `BUILD_JOBS=1` if RAM is low. Monitor temperatures and memory.
- `BUILD_WITH_MARCH_NATIVE=OFF` and `GTSAM_BUILD_WITH_MARCH_NATIVE=OFF` for all applicable builds. No x86 SIMD options are introduced.
- CUDA is selected by the installed `nvcc` when `USE_CUDA=AUTO` (default). We do not prescribe a CUDA version or desktop GPU architecture. On a CPU-only host, CUDA defaults OFF.
- If CUDA compilation fails, retain its log and rerun `USE_CUDA=OFF scripts/install_glim.sh`; choose `jetson_cpu` for runtime. Do not use a GPU preset with a CPU build.
- `BUILD_VIEWER=ON` builds Iridescence and the official PLY exporter. `BUILD_VIEWER=OFF` is available for an acquisition-only headless Jetson; export later on the workstation.
- `dependencies.lock` pins upstream SHAs and distinguishes x86 build evidence from ARM64 validation. `scripts/fetch_dependencies.py` refuses to reset dirty checkouts.
- `scripts/install_livox.sh`, `scripts/install_glim.sh`, `scripts/build_ros2.sh` can be rerun independently. Scripts preserve existing build directories.

## Network setup

Edit `config/livox/mid360.yaml`: LiDAR IP, assigned host IP, wired interface, topics and ROS domain. The supplied host/interface are specific to the tested PC. Configure a static Ethernet address in the same subnet on Jetson using the OS network settings. The application validates assigned addresses but does not change the OS network configuration. Preserve Wi-Fi/default-route settings.

Source `scripts/env.sh` for any manual ROS commands. It selects the correct Python, ROS overlay, local library paths and local ROS logs.
