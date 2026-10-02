# Architecture

```
Mid-360 -> official Livox driver -> PointCloud2 + Imu
                                  |             |
                                  +-- rosbag2 --+ -> immutable raw_bag
                                  +-- GLIM live -> session glim_dump
raw_bag -> upstream glim_rosbag -> processing/run_NNN/glim_dump
GLIM dump -> upstream offline_viewer exporter -> PLY -> optional PCD
PointCloud2 -> visualization-only voxel/cap -> binary WebSocket -> browser
```

FastAPI runs independently from sensor subprocesses. ROS imports remain in monitoring/preview subprocesses, allowing the UI to survive driver and GLIM crashes. All commands are constructed as argument arrays from fixed operations, with no shell expansion. A single event-loop lock serializes control mutations. Long-running work uses subprocess watchers and disk operations use worker threads where necessary.

`factory_mapping_monitor` measures actual message arrival, graph presence, timestamp and point rate. Network reachability and driver state are separate signals. `factory_mapping_preview` applies a voxel filter and hard point cap to a copy used only for display. There is no invented optimized-map ROS topic. The initial live viewer is explicitly a **raw sensor-frame cloud**. Exported PLY previews show the optimized result.

Sessions snapshot configuration at recording start. ROS bag creation owns `raw_bag/`. Live dumps and repeated offline jobs use independent paths. Metadata includes environment/revisions, sensor configuration, duration, rate observations and explicit mock flags. Stop order is recorder → metadata finalization → GLIM SIGINT/dump → log collection → session finalization. SIGTERM and ultimately SIGKILL are timeout escalation paths and are reported as forced; they are never marked clean output.

Backend restart marks unfinished sessions/jobs interrupted. PID records include creation time; stale identities are discarded. A matching surviving orphan is surfaced and blocks a duplicate role, rather than signalling an unknown process. The operator must inspect and stop that PID. Linux flock prevents two backends. The same lock is used by terminal test commands.

A bounded latest-frame file is shared between preview and HTTP server; slow WebSocket consumers cannot grow an unbounded queue. FMPC v1 has `<4sId` magic/count/seconds followed by little-endian float32 XYZI. Preview max defaults to 50,000 at 3 Hz and 0.1 m voxels. The raw bag and archival GLIM export never use these preview parameters.

Optional future plugins belong after stable acquisition. ScanContext is currently UNAVAILABLE when selected. `false` disables extra place recognition, not GLIM's existing geometric constraints. No camera runtime dependency is enabled.

## Optional parallel camera acquisition

`camera_config.py` validates trusted local camera YAML. `commands.camera` writes fixed gscam2 ROS parameters; camera, camera_monitor and camera_preview are separately managed subprocesses. FastAPI never imports rclpy/cv_bridge. Camera health uses its own atomic state file and JPEG preview uses its own endpoint, separate from the cloud WebSocket. The recorder includes image and CameraInfo topics only when enabled; GLIM RGB subscriptions are assigned an unused name for camera-enabled sessions.

`calibration_data.py` owns intrinsic/transform validation; `calibration.py` owns persistent independent calibration datasets and fixed manual tool commands. Calibration bags are hashed, originals are never passed to preprocess, and every tool stage copies the prior derived output. Imported results retain intrinsic hashes, transform convention and time offset for a future independent projection/colorization subsystem.
