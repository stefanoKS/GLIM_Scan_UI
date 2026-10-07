# GLIM Scan UI — factory recording and mapping

Record Livox Mid-360 LiDAR/IMU and optional RGB in a local browser dashboard, then process completed recordings with GLIM and optional NKSR on a workstation.

**The Jetson records only. Mapping, map editing and surface reconstruction run on the workstation.** GLIM uses only the Mid-360 LiDAR and built-in IMU; RGB is saved alongside them for separate camera/calibration work.

## Choose the installation

Run commands from the repository root. The installer targets Ubuntu 22.04 with ROS 2 Humble already installed at `/opt/ros/humble`; use system Python 3.10, not Conda, for the application environment. The supported Jetson target is Orin with JetPack 6 / Ubuntu 22.04. Other JetPack/Ubuntu combinations are not covered by this installer.

```bash
git clone https://github.com/stefanoKS/GLIM_Scan_UI.git
cd GLIM_Scan_UI
```

| Machine | Bootstrap command | Installed role |
| --- | --- | --- |
| Recording Jetson | `BUILD_JOBS=1 scripts/bootstrap_jetson.sh --record-only` | Livox driver, recorder, monitoring, preview and dashboard |
| Processing workstation | `scripts/bootstrap_jetson.sh --workstation` | Acquisition stack plus pinned GLIM dependencies and native tools |

Despite its name, `bootstrap_jetson.sh` supports both roles. Without a flag it selects recording-only on ARM64 and workstation on x86_64. Prefer the explicit commands above. Installation needs internet access and sudo for distribution packages; it builds native dependencies locally and does not upgrade JetPack or the OS.

The role is saved in `.state/deployment.json`. Recording-only mode blocks live/automatic GLIM, map tools, reconstruction preparation and NKSR even if older processing binaries remain installed. Transferring a project does not change the receiving machine's role. Workstation bootstrap probes CUDA and can use CPU GLIM when CUDA is unavailable; it does not install NKSR automatically.

See [installation and upgrades](docs/installation_jetson.md) for prerequisites, build controls, network setup and source-only transfer.

## Add the cameras you use

RGB is **requested by default**, with **Auto — D405 preferred** selection. Vendor camera stacks are installed separately:

| Camera | Target stream | Setup |
| --- | --- | --- |
| Intel RealSense D405 | 1280×720 RGB at 30 FPS | `scripts/install_d405.sh` — pins `pyrealsense2==2.58.2.10647`; standard librealsense USB permissions are also required |
| Imaging Source DFK 33UX287 | 720×540 RGB at 15 FPS | `BUILD_JOBS=1 scripts/install_camera.sh` — gscam2/tiscamera stack |

Install both stacks if you want either camera to be usable. Match device serials in `config/camera/` to your hardware. Bootstrap supplies OpenCV/cv_bridge prerequisites; the D405 installer does not install USB rules. D405 publishes rectified images with factory intrinsics; DFK uses its configured lens calibration. Both need their own camera–LiDAR mounting alignment when that alignment is used.

Auto tries D405, then DFK. An explicit D405/DFK choice is a preference and allows the other camera as fallback. USB detection alone is insufficient: the publisher must deliver fresh images and CameraInfo at the expected geometry and measured rate.

- **START SCAN:** if neither camera works, records LiDAR/IMU only, displays a warning, and records why RGB was unavailable. Include RGB stays enabled for the next scan.
- **RECORD CAMERA / calibration:** require a working camera and fail clearly if none is usable.
- **During acquisition:** the camera cannot switch. The snapshot records the actual camera and its calibration paths; calibration datasets remain bound to their original camera.

See [camera setup and calibration](docs/camera_calibration.md) and [selection, D405 performance and hardware diagnostics](docs/camera_auto_resolution.md). A 30 FPS target is not a claim of measured Jetson throughput.

## Start and record

Give the host Ethernet interface an address on the LiDAR subnet using the OS network settings. The tracked LiDAR IP is `192.168.1.120`; `interface: auto` discovers one suitable physical Ethernet interface. The host IP and interface name are not fixed to the development PC. The app does not change OS networking.

```bash
scripts/run_system.sh
```

Open [the dashboard](http://127.0.0.1:8080). For another browser on a trusted LAN, run `scripts/run_system.sh --host 0.0.0.0` and open port 8080 on the recording host's actual IP. There is no login; do not expose the dashboard to the Internet. Only one backend per checkout is allowed.

1. Check **Capture** sensor status. Preview startup does not record a bag.
2. Choose **START SCAN**. The app creates a session, checks sensors, resolves RGB if requested, snapshots acquisition settings and starts one raw bag.
3. Choose **STOP SCAN** and wait for finalization before disconnecting sensors or powering down.
4. In **Library**, rename/add notes and **Export selected**. Import the `.fmproject.zip` on the workstation to process, export maps or reconstruct surfaces.

A recording-only Jetson never automatically processes the scan. On a workstation, live mapping defaults off and automatic processing defaults on when GLIM is available; Settings controls these preferences. **RECORD CAMERA** makes a separate camera-only bag without requiring LiDAR/IMU.

**“Ready for next scan” is a capture-state label, not a guarantee that Start is enabled.** If it is accompanied by “A previous sensor process is still running,” recovery is required. “Previous scan: recorded” describes the prior finalized recording. See [process recovery](docs/troubleshooting.md#previous-sensor-process-is-still-running).

Stop the server with Ctrl-C and allow shutdown to complete. `scripts/stop_system.sh` asks a running backend to finalize the current session; it leaves the dashboard and preview sensors running. For hardware-free development, use `scripts/run_system.sh --mock` after installing the base environment; mock data is clearly marked and cannot be exported as genuine maps.

## Record on Jetson, process on the workstation

With the dashboard stopped, a terminal recording uses the same acquisition checks:

```bash
scripts/record_test.sh --name terminal_test --seconds 30
```

Transfer the entire completed session, including its raw bag, metadata, `active_config.json` and `config_snapshot/`. Prefer Library project export/import, which verifies hashes. Keep the original until the transfer is verified. Dedicated calibration datasets are transferred separately.

On the **workstation**, with its dashboard stopped, use the printed/imported session ID:

```bash
scripts/process_bag.sh SESSION_ID --preset jetson_cpu
```

`jetson_cpu` is the CPU GLIM preset's historical name; it does not enable processing on a recording-only Jetson. Each processing attempt creates a new `processing/run_NNN/`. Raw bags remain unchanged. Library also provides processing and the upstream native editing tools; native windows open on the workstation's desktop, not inside a remote browser.

Optional [NKSR surface reconstruction](docs/nksr.md) uses a separate workstation environment. Preparing point input and reconstructing a mesh are separate steps from GLIM optimized PLY export.

Optional [camera RGB colorization](docs/colorization.md) produces the master colored point cloud and can copy its color onto the GLIM PLY and the NKSR mesh. Automatic target selection is bound to the colorized processing run and refuses to guess when the lineage is ambiguous.

## Verify this build

The latest recorded implementation run for the camera-selection build (`ba70f6d`, 2026-10-05) reported **178 passed, 1 skipped, 1 warning** on x86_64. The skip was opt-in NKSR inference; native RealSense projection tests ran without physical cameras. JavaScript checks also passed. See [validation history and remaining checks](docs/validation.md). These results do not certify Jetson hardware or sustained 30 FPS recording.

To rerun software tests:

```bash
source scripts/env.sh
.venv/bin/python -m pytest -q --basetemp=.state/pytest
```

On the Jetson, with the dashboard stopped:

```bash
scripts/verify_jetson.sh --record-only
# Mid-360 attached: create and finalize a 30-second recording.
scripts/verify_jetson.sh --record-only --hardware
```

Run the [four camera hardware cases](docs/camera_auto_resolution.md#jetson-commands), then a full-duration route on target storage. Check delivered FPS, bag topic counts, USB stability, free space, memory and temperature, and process the transferred recording on the workstation.

## Documentation

| Guide | Contents |
| --- | --- |
| [Installation](docs/installation_jetson.md) | Jetson/workstation roles, native builds, upgrades and networking |
| [Operation](docs/operation.md) | Capture, shutdown, transfer and workstation processing |
| [Troubleshooting](docs/troubleshooting.md) | Orphan recovery, camera fallback, recording and processing faults |
| [Camera setup](docs/camera_calibration.md) | D405/DFK installation, intrinsics and alignment datasets |
| [Camera selection](docs/camera_auto_resolution.md) | State machine, performance tests and hardware commands |
| [GLIM tools](docs/glim_tools.md) | Native editing, merging and cleanup on the workstation |
| [Editing and surfacing](docs/editing_and_surfacing.md) | End-to-end saved-map cleanup, export, preparation and NKSR mesh workflow |
| [NKSR](docs/nksr.md) | Workstation surface preparation, reconstruction and validation |
| [Colorization](docs/colorization.md) | RGB point cloud, run lineage, bounded observations and color transfer |
| [Architecture](docs/architecture.md) | Process ownership, snapshots, camera isolation and recovery |
| [Validation](docs/validation.md) | Dated evidence and hardware limitations |
| [Upstream contracts](docs/upstream_interfaces.md) | Pinned driver/GLIM interfaces and preset behavior |

`dependencies.lock` pins upstream source; `requirements.lock` pins the app/test Python environment. Build artifacts, local preferences, acquisition data and generated maps are excluded from Git. [Original requirements](docs/requirements.md), [early capture analysis](docs/capture_refactor_analysis.md) and the [environment report](docs/environment_report.md) are historical records, not fresh-install instructions.
