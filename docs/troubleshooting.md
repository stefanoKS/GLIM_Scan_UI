# Troubleshooting

Use **Settings → Advanced / Diagnostics**, `scripts/diagnose.sh` and the relevant process/session logs. Current sensor readiness and the previous capture result are separate. Preserve failed/interrupted raw bags and their logs while investigating.

## Previous sensor process is still running

This Capture message means ProcessManager found a surviving process group from an earlier backend instance. It verifies saved PID/creation-time information and refuses automatic adoption or duplicate publishers. An ordinary preview process owned by the current backend is not an orphan.

- **Ready for next scan** is the high-level capture-state label. A recovery condition can still disable Start.
- **A previous sensor process is still running** means current process ownership needs recovery.
- **Previous scan: recorded** describes the earlier finalized recording. The orphan warning alone does not invalidate it; inspect that session's metadata/logs if its result is uncertain.

Recovery:

1. Open Advanced / Diagnostics and identify the role marked `orphaned`, its PID and command. Saved records are `.state/process_*.json`. Do not delete those files or `.state` to hide a live process.
2. If an older dashboard/terminal still owns the process, use that instance to stop capture and exit gracefully. Avoid launching another recorder.
3. Otherwise close the current dashboard backend with Ctrl-C, allowing its owned processes to finish. Inspect the reported process and group in a terminal. Replace the placeholders with the reported numeric values:

   ```bash
   ps -p PID -o pid,pgid,lstart,args
   ps -eo pid,pgid,args
   ```

   Confirm the command, start time, checkout and all group members belong to this application's abandoned sensor/recorder. A PID may have been reused; if identity is uncertain, do not signal it.
4. For a confirmed abandoned group, request graceful shutdown of that **specific** group:

   ```bash
   kill -INT -- -PGID
   ```

   The leading minus targets the verified process group. Wait for the recorder to finalize; shutdown may take time. If it does not exit, retain logs and investigate the affected role rather than broadly killing ROS/Python processes. A forced termination can leave an incomplete bag.
5. Verify the group has exited, then restart `scripts/run_system.sh`. Startup removes records for processes that are no longer alive and rechecks any survivors. Reinspect sensor readiness before starting a new scan.

`scripts/stop_system.sh` only asks a running backend to finalize its current session. It does not terminate the server/preview sensors or take ownership of orphaned processes. Normal full shutdown is Ctrl-C in the backend terminal after capture stops.

## RGB unavailable or wrong preferred camera

Auto tries D405 then DFK; either explicit choice allows the other supported camera as fallback. Old `.state/camera_profile.json` values remain valid preferences. A malformed/stale profile uses Auto with a warning; save a preference in Settings to replace it. Deleting local state is not required.

Check `camera_selection` in `/api/status` or Advanced / Diagnostics: it distinguishes each camera's USB detection, dependency check, real stream health, active profile and fallback reason. USB presence does not prove image delivery.

- **Not detected:** check USB power/cable and configured SDK/USB serials in `config/camera/`. Detection is specific to those configured devices.
- **D405 dependency unavailable:** run `scripts/install_d405.sh` in the checkout's environment; expected pin is `pyrealsense2==2.58.2.10647`. Check standard librealsense USB permissions separately; the installer does not add udev rules.
- **DFK dependency unavailable:** use `scripts/install_camera.sh`; source `scripts/env.sh` for local GStreamer/Tcam paths, check permissions and reconnect after group/rule changes.
- **Publisher exited / no image messages:** inspect `.state/camera.log` and `.state/camera_monitor.log`. D405 logs resolve, pipeline.start, calibration, map creation and first-frame timing; a missing stage helps locate startup failure.
- **Wrong geometry, low rate or missing CameraInfo:** inspect actual stream samples and camera configuration. D405 targets 1280×720 at 30 FPS; DFK targets 720×540 at 15 FPS. Health uses measured delivery, not the requested rate. A preview failure alone does not prove recording has stopped.

Run the [four camera diagnostics](camera_auto_resolution.md#jetson-commands) with the dashboard stopped. Their reports include USB speed, SDK version, actual selection/FPS and fallback reasons. A normal START SCAN may continue as LiDAR/IMU-only with explicit metadata; RECORD CAMERA and calibration require a working camera. Cameras cannot switch during acquisition. If a calibration dataset's original camera is unavailable, reconnect it or create a new dataset for a different camera.

## Mid-360 and ROS

- **Ethernet missing / host IP missing:** assign a host address on the LiDAR subnet using OS network settings. Auto selection needs exactly one active matching physical Ethernet adapter; select an explicit interface when ambiguous. The app does not configure the OS.
- **Unreachable:** check sensor power, cable, subnet and firewall. ICMP reachability alone does not prove ROS message delivery.
- **Driver stopped/failed:** inspect `.state/driver.log` or the session's `logs/driver.log`. **Topic missing:** check domain, remapping and PointCloud2 `xfer_format=0`. **Topic exists/no messages:** check UDP destination, device state and sensor power. Expected defaults are 10 Hz point clouds and 200 Hz IMU.
- **ROS Python import error:** source `scripts/env.sh`; use the project's `.venv/bin/python` based on `/usr/bin/python3`. Do not pip-install a replacement rclpy or use Conda Python for acquisition.

For manual topic inspection, use the acquisition domain from `config/livox/mid360.yaml` (41 by default). Sourcing the environment sets paths but does not set this domain for your terminal:

```bash
source scripts/env.sh
export ROS_DOMAIN_ID=41
ros2 topic list -t
ros2 topic hz /livox/lidar
```

## Recording and storage

- **Missing bag metadata / failed finalization:** preserve `raw_bag/`, metadata and `logs/recording.log`. A file's presence alone does not establish a complete recording: finalization checks duration and message counts on requested topics. Any reindex/recovery should be an explicit operation on a backup.
- **RGB was requested but absent:** inspect `metadata.json` → `camera_selection` and the acquisition's topic list. A deliberate preflight fallback records only LiDAR/IMU. `active_config.json` is the preflight snapshot; final metadata establishes `rgb_recorded` and recorded FPS.
- **Disk low:** the default minimum is 5 GB. The backend starts a graceful stop at the configured threshold. Move verified completed projects before another run; full-resolution raw RGB can consume storage quickly.
- **Driver shutdown crash or hang after unplugging:** these were observed in earlier PC runs of the pinned Livox driver. Preserve logs and verify the next start. The manager tracks the whole process group and escalates only after its graceful timeout; forced shutdown is reported. Stop capture before disconnecting where possible.

## Workstation processing

- **Recording-only host:** mapping/surfacing is deliberately blocked by `.state/deployment.json`, even if old binaries exist. Transfer the project to the workstation. Installing GLIM alone or changing its preset does not change the host role.
- **GPU preset unavailable:** CUDA presets require an actual CUDA-enabled GLIM build. Use `jetson_cpu` or `pc_dense` on a CPU GLIM workstation; `jetson_cpu` is a preset name, not permission to process on the recording Jetson.
- **No valid GLIM output:** inspect `processing/run_NNN/job.log` and `job.json`. Empty input, timestamp units, extrinsics or missing libraries can cause failure. A zero exit code alone is insufficient without graph/trajectory artifacts.
- **Offline timestamp rewinds with live sensors:** use the project wrapper, which separates the offline ROS domain and remaps live subscriptions. Direct unisolated upstream `glim_rosbag` can receive live messages while reading a bag.
- **Exporter/OpenGL error / editor not visible remotely:** native GLIM tools run on the workstation's display, not in the browser. Use a local desktop/OpenGL environment and a viewer-enabled build.
- **NKSR unavailable:** see [NKSR setup and diagnostics](nksr.md). It uses a separate workstation environment; ordinary optimized PLY export does not depend on NKSR.
