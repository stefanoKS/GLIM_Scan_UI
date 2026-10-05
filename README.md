# Factory Mapping — GLIM + Mid-360

Local ROS 2 acquisition, immutable raw bags, GLIM processing jobs and a lightweight browser interface. GLIM uses **only the Mid-360 LiDAR and its IMU**. Optional parallel RGB acquisition and manual LiDAR–camera calibration are available. The Jetson records raw sensor data only; the workstation performs mapping and surfacing.

## Install from GitHub

Use Ubuntu 22.04 with ROS 2 Humble already installed. On Jetson, use a JetPack 6 image with Ubuntu 22.04 and ROS 2 Humble; do not upgrade JetPack. For a new Ubuntu workstation, install ROS 2 Humble from the [official installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) before continuing.

Clone the private repository using your GitHub access, then run the bootstrap from its root:

```bash
git clone https://github.com/stefanoKS/GLIM_Scan_UI.git
cd GLIM_Scan_UI
scripts/check_system.sh
scripts/bootstrap_jetson.sh
```

Bootstrap installs the OS build prerequisites, creates the Python environment, downloads the exact upstream GLIM and Livox revisions pinned in `dependencies.lock`, builds and installs them into this checkout, then builds the ROS 2 workspace. GLIM is fetched automatically; do not clone or install it separately. The first build needs internet access, `sudo` for apt packages, and can take a while. Use `BUILD_JOBS=1 scripts/bootstrap_jetson.sh` on memory-constrained machines. CUDA is detected automatically and falls back to CPU when unavailable.

For the recording Jetson, use `scripts/bootstrap_jetson.sh --record-only` (also the default on ARM64). This skips GLIM/CUDA and never installs NKSR. It saves a host-local recording-only policy in `.state/deployment.json`: live mapping, automatic processing, map tools and surfacing are blocked even if older installations remain. Choose **START SCAN**, then **STOP SCAN**, wait for completion, and export the project from Library. Import it on the workstation for mapping and surfacing. Project transfers do not change either machine's deployment mode.

The supported Jetson target is **JetPack 6 / Ubuntu 22.04 / ROS 2 Humble**, on supported Orin hardware; older Nano/TX2 images and JetPack 5 or 7 are not supported by this installer. See [NVIDIA's JetPack 6 platform details](https://developer.nvidia.com/embedded/jetpack-sdk-60). Bootstrap stops on a different Ubuntu version without modifying it. Use `--workstation` only on a machine intended to process data.

RGB is enabled by default. After bootstrap, run `scripts/install_d405.sh` and configure USB access before recording with the D405, or disable RGB in dashboard Settings for LiDAR/IMU-only recording. Bootstrap supplies OpenCV and cv_bridge for the camera path. Camera SDK installation and actual USB capture must be checked on the target device.

Run `scripts/verify_jetson.sh --record-only` on the Jetson for software checks. With the dashboard stopped and sensors connected, run `scripts/verify_jetson.sh --record-only --hardware` for a 30-second finalized recording. Before deployment, also record for the intended route duration on the target storage, check for dropped messages, memory/thermal issues and disk capacity, then transfer and process that recording on the workstation. Passing workstation tests does not certify Jetson hardware throughput.

Start the dashboard after bootstrap completes:

```bash
scripts/run_system.sh
```

Open http://127.0.0.1:8080. For network and hardware setup, see [installation](docs/installation_jetson.md) and [operation](docs/operation.md).

## Quick start on this PC

```bash
scripts/run_system.sh
```

Open http://127.0.0.1:8080. For another PC on the trusted LAN, use `scripts/run_system.sh --host 0.0.0.0`, then open `http://192.168.1.135:8080`. The UI has no login; use it only on a trusted network and do not expose it to the Internet. One backend instance per repository is allowed.

Use `--mock` for development without ROS hardware. Mock sessions are marked and filtered separately; they cannot be exported as genuine maps. Stop with Ctrl-C to finalize recording and GLIM. `scripts/stop_system.sh` finalizes the active session while keeping the dashboard running.

The configured device is `192.168.1.120`, host `192.168.1.135`, interface `enp6s0`. Network edits only update project configuration; they do not change host networking or sensor firmware settings.

## Terminal-only test

With the UI stopped:

```bash
scripts/record_test.sh --name terminal_test --seconds 30
# Copy the session ID printed by the recorder:
scripts/process_bag.sh YYYYMMDD_HHMMSS_terminal_test --preset jetson_cpu
```

Each run writes a new `processing/run_NNN/` directory. Inspect `job.log`, `job.json` and `glim_dump/`. Never pass an unfinished bag to GLIM.

## Simple operator workflow

**Capture** has START/STOP SCAN, RECORD/STOP CAMERA, sensor health, elapsed time, a LiDAR view and camera preview. **Library** holds completed recordings, names/notes, transfer, processing, run history and map editing. **Calibration** provides the guided setup; **Settings → Advanced / Diagnostics** retains all engineering controls. The backend owns capture sequencing and keeps raw recording running if live GLIM fails.

See the [system analysis and verification](docs/capture_refactor_analysis.md) for the inspected modules, fixes and remaining hardware checks.

## Official GLIM editing tools

The dashboard includes launchers for **manual loop closure, map merging, plane constraints, optimization, graph recovery, MinCut/region-growing segmentation and map cleanup**. They use the installed upstream `offline_viewer` and `map_editor` on the server desktop, starting from separate working copies. The native live viewer and upstream sensor validator are also available. See [toolkit workflows](docs/glim_tools.md).

## Installation and verification

See [installation](docs/installation_jetson.md), [environment report](docs/environment_report.md), [validation results](docs/validation.md), [operation](docs/operation.md) and [upstream interfaces/preset changes](docs/upstream_interfaces.md).

```bash
source scripts/env.sh
.venv/bin/python -m pytest -q --basetemp=.state/pytest
```

No ARM64 build result is claimed from the x86_64 test machine. No AVX/native architecture options are enabled. Dependency revisions and build evidence are in `dependencies.lock`. Build/install/log trees, raw bags and map outputs are excluded from Git.

## Optional RGB camera and calibration

RGB recording is enabled by default: Start Scan records camera images and CameraInfo alongside LiDAR and IMU. Disabling RGB in Settings shows a warning and still allows LiDAR-only scans. The default RGB camera is the D405, using factory intrinsics without a printed calibration board. `scripts/install_d405.sh` adds its SDK to the existing camera environment. Camera–LiDAR mounting alignment is still required. Change the camera selection in Settings to use the preserved DFK 33UX287 profile and `scripts/install_camera.sh` stack; selection is stored locally and changing profiles restarts the camera stream. `scripts/install_calibration.sh` adds the separate workstation calibrator. Neither is required for LiDAR-only or record-only operation.

See [camera setup, calibration conventions and hardware acceptance commands](docs/camera_calibration.md). The dashboard has camera health/JPEG preview, intrinsic YAML import and a separate persistent calibration workflow: static captures → preprocessing → manual alignment → NID → result import → independent validation. SuperGlue and final colorization are not integrated.

Optional surface reconstruction: [install and run pretrained NVIDIA NKSR](docs/nksr.md).
Point preparation and mesh reconstruction are separate steps; neither changes the
existing GLIM optimized PLY export.
