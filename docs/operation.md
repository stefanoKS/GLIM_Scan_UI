# Operation

1. Power the Mid-360 and connect Ethernet. Host and sensor must share a subnet. Connect the optional DFK33UX287 over USB3 (720×540).
2. Start `scripts/run_system.sh` and open http://127.0.0.1:8080. LiDAR and the configured camera start automatically for live preview; this does not create a session or record a bag.
3. In **Settings**, choose whether to include the camera. Live mapping is optional and off by default; automatic processing is on when GLIM is installed. The application selects CPU/GPU from the local installation.
4. Press **START SCAN**. The backend creates a dated session, starts the required sensors, verifies LiDAR/IMU and enabled camera streams, then starts one raw rosbag. There is no need to create a session or start individual processes. Optional live GLIM starts after recording; its failure does not stop the bag.
5. Walk slowly, keep the sensor unobstructed, revisit locations and return near the start. Prefer overlapping zones over one enormous recording.
6. Press **STOP SCAN**. Wait through Saving recording and, when available, Processing scan. Do not power down while finalization is running. A record-only Jetson saves the scan without launching GLIM.
7. Open **Library** to rename the scan, add notes, process/reprocess, inspect run history, export projects/maps, or use **Edit Map**, **Clean Map**, and **Merge Maps**. Native tools open on the server desktop. Edit Map includes manual loop closure, constraints and optimization; Clean Map exposes upstream segmentation/cleanup; Merge Maps starts with separate copies of the selected completed maps.
8. **RECORD CAMERA** records only RGB and CameraInfo in a separate rosbag, independently of the scan's camera preference. **STOP CAMERA** finalizes it. It does not need LiDAR/IMU or run GLIM. Concurrent scan and standalone camera recording are intentionally mutually exclusive; enable the camera in Settings for a combined scan.
9. **Calibration** guides camera intrinsics and camera–LiDAR alignment. Low-level process controls, presets, topics, frames, network and timestamp diagnostics are under **Settings → Advanced / Diagnostics**.

Health indicators distinguish a connected device from a healthy publishing stream. Storage shows free space. The elapsed timer runs while recording. Errors retain the raw files and appear on Capture; use Library and Advanced Diagnostics for recovery. A disconnected required sensor prevents capture from claiming success.

The live browser preview is raw LiDAR in its sensor frame, not a moving optimized world map. A preview is intentionally downsampled; archived bags remain full data. GLIM itself applies its normal estimation filtering, so the optimized export is not a replacement for the raw bag.

To move a mapping project to a workstation, stop its session and any processing, select it in **Library**, and choose **Export selected**. The downloaded `.fmproject.zip` contains the entire session directory: raw bag, GLIM dump and processing runs, map exports, logs, `metadata.json`, `active_config.json`, and `config_snapshot/` (including the intrinsics/extrinsics and GLIM presets captured for that session). On a server running this application, choose **Project ZIP** and **Import project**. The manifest verifies file sizes and SHA-256 hashes before importing into `data/sessions/`; an existing session ID is never overwritten. Importing does not change the workstation's live camera, LiDAR, ROS domain, or network settings. The target still needs compatible ROS 2/rosbag storage and GLIM installed to process a raw bag. Archives are not a way to move binaries between architectures. Dedicated calibration datasets under `data/calibrations/` are not part of a mapping session project; copy those separately if their history is needed.

Alternatively, copy a **completed entire session directory** to the workstation's `data/sessions/` manually. Preserve `metadata.json`, `active_config.json`, and `config_snapshot/`. The application never mutates a bag during processing. No raw-bag deletion endpoint exists. Delete Derived Run only removes the selected generated run after processing has stopped.

Terminal manual commands:

```bash
source scripts/env.sh
scripts/record_test.sh --name manual_zone --seconds 30
scripts/process_bag.sh SESSION_ID --preset jetson_cpu
ros2 bag info "data/sessions/SESSION_ID/raw_bag"
ros2 run glim_ros offline_viewer "$PWD/data/sessions/SESSION_ID/processing/run_001/glim_dump" \
  --config_path "$PWD/data/sessions/SESSION_ID/processing/run_001/config" \
  --export_path "$PWD/data/sessions/SESSION_ID/exports/map.ply"
```

For exact GLIM commands and dependencies see `upstream_interfaces.md` and `dependencies.lock`. Keep the UI stopped while using terminal commands that manage the same device.

## Jetson record-only operation

Without GLIM installed, **START SCAN** automatically uses record-only operation. It retains full PointCloud2 + IMU data (plus camera when enabled), checks streams and free space, and finalizes the bag on **STOP SCAN**. On a machine with GLIM, disable both live mapping and automatic processing in Settings for the same acquisition-only workflow. Sensors remain available for preview between captures and stop when the backend closes.

The dashboard and CLI do not require a token. For terminal control of an already running backend:

```bash
scripts/fm.py create jetson_zone
scripts/fm.py action session_record_start --session SESSION_ID
scripts/fm.py action session_stop
```

After shutdown completes, copy the entire session to the workstation. Select it there and process/export with GLIM. Keep the original Jetson copy until the copied bag has been verified.

## Optional camera extension

See [camera acquisition and calibration](camera_calibration.md) for opt-in installation, fixed pipeline setup, one-bag RGB recording, intrinsic import, static calibration datasets and the workstation manual/NID workflow. GLIM remains LiDAR + IMU only. Camera-disabled and record-only workflows above remain supported.

## Level the live LiDAR view

Under Calibration, keep the mounted sensor stationary and click **Orient LiDAR View** once. The app immediately uses the latest fresh, stable IMU window already collected in the background; there is no two-second wait after clicking. Motion, stale or missing measurements are rejected immediately. If the sensor was stopped, the app starts it; retry once healthy IMU data is available. The saved roll/pitch correction applies only to the live browser cloud; raw bags, GLIM inputs, extrinsics, optimized map previews and exports are unchanged. No point preprocessing is added. Gravity cannot determine heading. Reorient after changing the mounting angle; **Reset orientation** restores the original display. The setting is machine-local in `.state/view_orientation.json`.

Ethernet defaults to automatic selection of a single active physical adapter on the configured LiDAR subnet. The app does not assign IP addresses or change OS networking. Explicit selection remains available under Advanced Diagnostics when multiple wired adapters match.

## PC dense mapping

Settings → Capture preferences → Mapping quality selects the preset used by Start Scan for automatic processing (and live mapping if enabled). Save preferences also selects that preset for Library → Process / Reprocess; Advanced / Diagnostics retains the per-job selector. Existing processing runs and exports are not rewritten.

`PC dense · CPU · 5 cm detail` targets the current Ryzen 5 5600G / 16 GB workstation. It uses six CPU threads, a 20,000-point preprocessing target, 0.05 m minimum point separation within submap voxels, at most 500 points per voxel and 200,000 points per submap. These are sampling settings, not guaranteed accuracy or uniform spacing. CPU odometry and pose-graph mapping remain compatible with the installed CPU-only GLIM build. The RTX 5060 is present, but CUDA presets require a CUDA-enabled GLIM installation.

Leave live mapping off and automatic processing on for this PC. Dense processing uses more memory and time; long routes still need memory/performance testing. Raw bags are unchanged. Browser previews remain limited to 50,000 points; use the exported PLY/native viewer to evaluate full map detail.

### Prepare surface reconstruction

Select a scan in Library → Surface Reconstruction, choose its `traj_lidar.txt`,
and click **Prepare Reconstruction**. **Advanced → Voxel size** defaults to
**1.0 cm** and accepts 0.2–20 cm; the browser converts centimeters to meters.
The backend API accepts `voxel_size_m` (default `0.01`; `0` disables sampling).
Preparation requires the original ROS bag and the trajectory from the same scan.
For merged maps, use each original bag with its corresponding transformed trajectory.
Preparation does not require NKSR or Open3D, and does not generate a mesh itself.

From a shell with the project ROS environment:

```bash
source scripts/env.sh
python tools/glim_nksr_prepare.py \
  --bag data/sessions/SESSION/raw_bag \
  --trajectory data/sessions/SESSION/processing/run_001/glim_dump/traj_lidar.txt \
  --output-dir data/sessions/SESSION/reconstruction/run_debug \
  --voxel-size 0.01
```

CLI voxel units are **meters**: `0.01` = 1 cm. Use `--voxel-size 0` for
full-density NKSR input, or `--save-full-density` to additionally write a full
validation PLY. Use a new output directory for each preparation.

Per-point trajectory interpolation and trajectory-range filtering precede global
world-space voxel selection (`floor(point_world / voxel_size_m)`). NumPy selects
the first actual observation in each occupied voxel, in acquisition order, and
applies those same indices to every measurement attribute. No sensor origins are
averaged independently. The selection is deterministic for identical inputs.

Outputs under `reconstruction/run_*/` are:

- `input/nksr_input.npz`: sampled points, paired sensor origins, intensity and timestamps.
- `validation/reconstructed_from_bag.ply`: the same sampled geometry sent to NKSR.
- `validation/comparison.json`: `voxel_size_m`, `points_before_voxel`,
  `points_after_voxel`, and `voxel_reduction_ratio` (after / before).
- `validation/reconstructed_from_bag_full.ply`: only when explicitly requested.

The UI displays raw and sampled point counts, progress, and preparation logs.
Compare spatial agreement with GLIM's optimized export; its processed submaps
naturally have a different point count. No equality check is applied and the
metadata does not claim a spatial validation has been performed. Existing
`exports/run_*.ply` and GLIM export commands are unchanged.
