# Operation

1. Power the Mid-360 and connect Ethernet. Ensure host and sensor addresses share a subnet.
2. Start `scripts/run_system.sh` and open http://127.0.0.1:8080.
3. Start Mid-360. Verify **both** LiDAR and IMU rates, latest timestamps, network state and driver status. A process alone is not a connection test.
4. Create a named session and enter notes. Start Record-only Session automatically starts the sensor and records LiDAR + IMU without launching GLIM. Use Start Recording + GLIM for live mapping as well. The separate Start Recording button remains available after manually starting the sensor. CPU is the preset for this test PC.
5. Walk slowly, avoid violent rotations, keep the sensor unobstructed, revisit locations and return near the start. Prefer several overlapping factory zones over a single enormous recording.
6. Stop Session. Wait for rosbag metadata and GLIM dump finalization before powering off. Stopping recording separately leaves live GLIM active until Stop GLIM/Stop Session.
7. Select a session, choose a preset, and Process/Reprocess. Watch the processing log. Each attempt gets a separate run directory and state.
8. Inspect trajectory statistics and exported map. These describe the reconstruction and do not establish metrological accuracy.
9. Export to PLY using upstream GLIM. The exporter needs an OpenGL display even when invoked by the backend; use a desktop workstation for headless Jetson sessions. PCD conversion preserves all exported points and intensity when present. Exported PLY can be previewed in the browser.

The live browser preview is raw LiDAR in its sensor frame, not a moving optimized world map. A preview is intentionally downsampled; archived bags remain full data. GLIM itself applies its normal estimation filtering, so the optimized export is not a replacement for the raw bag.

Copy a **completed entire session directory** to the workstation's `data/sessions/` to process locally. Preserve `metadata.json`, `active_config.json`, and `config_snapshot/`. The same ROS distribution/storage plugin must be able to read its raw bag. The application never mutates a bag during processing. No raw-bag deletion endpoint exists. Delete Derived Run only removes the selected generated run after processing has stopped.

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

Record-only ignores the GLIM preset, does not require CUDA or installed GLIM binaries, and retains full PointCloud2 + IMU data for later processing. It still verifies both message streams and free disk space before recording. If live GLIM is already running, stop it before choosing record-only. Stop Session finalizes the bag and metadata; Stop Mid-360 then shuts down the sensor driver and preview.

The dashboard and CLI do not require a token. For terminal control of an already running backend:

```bash
scripts/fm.py create jetson_zone
scripts/fm.py action session_record_start --session SESSION_ID
scripts/fm.py action session_stop
```

After shutdown completes, copy the entire session to the workstation. Select it there and process/export with GLIM. Keep the original Jetson copy until the copied bag has been verified.
