# Optional RGB acquisition and LiDAR–camera calibration

The Mid-360 LiDAR + IMU remain GLIM's only inputs. Camera RGB and CameraInfo are parallel acquisition streams in the same raw rosbag. Camera-disabled operation retains the original two-topic recording, mock mode, native GLIM toolkit and record-only Jetson option. No GLIM core code is modified. No final colorizer, mesh or texture processing is included.

## Three independent calibrations

- **Intrinsics** describe pixel projection: image geometry, fx/fy/cx/cy, distortion and projection model. Repeat after lens changes, significant focus changes, or resolution/crop changes.
- **Extrinsics** describe the rigid camera-to-LiDAR transform. The stored convention is **p_lidar = T_lidar_camera × p_camera**, with `[x,y,z,qx,qy,qz,qw]`. Repeat after either mount moves or the camera/lens changes. A future projector transforms LiDAR points into camera coordinates using the inverse; the stored transform is never silently inverted or renamed.
- **Time offset** is a separate scalar. `t_lidar = t_camera + time_offset_sec`. Initially zero, not estimated. It is stored with configurations and calibration results; it does not rewrite raw timestamps or change GLIM timing.

Camera is initially free-running. gscam2's `use_gst_timestamps=true` uses GStreamer PTS with its ROS clock adjustment; this is not a claim of hardware exposure synchronization. Check header timestamps, delay, jitter and rewinds on the real hardware. Hardware triggering and automatic offset estimation are future work.

## Optional installation

First install the existing acquisition stack as described in installation_jetson.md. Camera and calibration packages are never pulled by default or record-only bootstrap.

On the camera acquisition host:

```bash
scripts/install_camera.sh
source scripts/env.sh
scripts/check_camera.sh
```

This optionally installs system GStreamer, cv_bridge and OpenCV dependencies, fetches pinned gscam2/tiscamera sources and builds into this project. It does not install pip opencv-python. tiscamera uses a project-local prefix including its static data, V4L2/libusb support, and no GigE backend or GUI build. The script does not install global udev rules, alter firmware, enable triggers or modify networking. Follow the vendor's USB permissions/udev instructions if the camera is inaccessible; project-local udev files alone are not activated by the OS. Do not run the dashboard as root.

The vendor has placed tiscamera in maintenance mode and announced EOL in 2029. This implementation uses the requested tiscamera interface and keeps the publisher boundary replaceable. See the [vendor repository](https://github.com/TheImagingSource/tiscamera).

On a workstation with the existing GLIM dependencies:

```bash
scripts/install_calibration.sh
source scripts/env.sh
ros2 pkg executables direct_visual_lidar_calibration
```

The optional calibrator build needs Ceres, OpenCV, PCL, GTSAM and Iridescence. No SuperGlue package, weights or matching command is installed/invoked by these wrappers. Upstream may ship its unused matching script as part of its package; the project API exposes only preprocess, initial_guess_manual and calibrate. Their source revisions are pinned in dependencies.lock and explicitly marked **source-inspected, not build/hardware validated**. These installers require sudo for apt prerequisites. Native ARM64 builds, gscam2 Humble compatibility and the calibrator's compatibility with this exact dependency stack must still be verified.

## Physical DFK configuration

`config/camera/dfk33ux287.yaml` contains the model/name/serial, frame and topics, width/height, configured and expected FPS, timestamp preference, preview bounds, calibration paths, optional-camera policy and offset. The supplied 720×540 / 15 Hz values are requested configuration defaults, **not verified DFK 33UX287 capabilities**.

The pipeline is intentionally null and `pipeline_validated=false`. Never mark it validated based on a generic example. With the DFK physically attached by USB3:

```bash
source scripts/env.sh
tcam-ctrl --list
# Replace CAMERA_SERIAL with the listed serial:
tcam-ctrl --caps CAMERA_SERIAL
gst-inspect-1.0 tcambin
scripts/check_camera.sh CAMERA_SERIAL
```

Inspect supported formats, dimensions and rates. Validate a source/conversion pipeline that produces the requested raw RGB geometry for gscam2 (which supplies its own sink). Store the exact validated pipeline in **local trusted YAML**, along with matching serial, geometry, FPS and `pipeline_validated: true`. No HTTP endpoint accepts or modifies pipelines or arbitrary command strings. The gscam2 ROS parameters are written as YAML, avoiding shell parsing. The configured `serial_number` is used for device diagnostics; the validated pipeline must select that same serial. If unset, recorded serial metadata remains null rather than claiming an identification.

Set `camera.enabled: true` in config/system.yaml and restart the backend. Configuration changes while running require a restart; stop recording first. Start Camera in the dashboard. Check Image FPS, geometry, frame ID, age and timestamp diagnostics. Missing intrinsics are allowed for preview/testing; CameraInfo will be reported missing or invalid, not calibrated. Preview has a separate subprocess and endpoint, so decoding/JPEG failures do not stop recording.

Hardware acceptance commands after the camera is publishing:

```bash
source scripts/env.sh
ros2 topic info /camera/image_raw -v
ros2 topic hz /camera/image_raw
ros2 topic echo /camera/camera_info --once
ros2 topic echo /camera/image_raw --once --field header
scripts/fm.py status
```

Use configured topic names if changed. `camera.required_for_mapping: false` leaves LiDAR-only startup possible even when RGB is unhealthy. Enabled camera topics are still included in the bag; post-run metadata exposes missing/zero image counts. When true, combined session startup starts the camera and waits for healthy frames; it blocks on missing frames or incorrect geometry/rate. Stopping Camera stops only camera-related roles, leaving GLIM and mapping recording alone. Stop a calibration capture before stopping either sensor.

## Intrinsics import

No fake matrices are shipped. The canonical intrinsic YAML is initially a missing-calibration marker. Import a measured standard ROS camera calibration YAML through the camera panel or replace the configured local file before starting the camera. The UI import archives previous files and invalidates the active extrinsic association; historical datasets and session snapshots remain intact.

Supported models are plumb_bob with five distortion terms, equidistant/fisheye with four, and omnidir with four plus a measured `xi` extension. Intrinsic files must contain K, R, P, distortion, image_width and image_height. Geometry must match configured acquisition. Omnidir's xi is passed explicitly to preprocessing because standard CameraInfo does not carry it. Unknown models are rejected rather than guessed.

For obtaining an intrinsic file, ROS's optional `camera_calibration` package can be used separately. With a measured checkerboard, supply its actual internal-corner count and square size, for example the command pattern below (replace all placeholders):

```bash
ros2 run camera_calibration cameracalibrator --size COLSxROWS --square METRES \
  --ros-args -r image:=/camera/image_raw -r camera:=/camera
```

Save/export its YAML and import it here. Interactive intrinsic calibration is not a backend dependency or a ChArUco implementation. Revalidate resolution, focus and lens before collecting extrinsics.

## Calibration datasets and state

The separate dashboard panel persists datasets under `data/calibrations/cal_ID/`. They never appear as mapping sessions:

```
metadata.json
active_config.json
config_snapshot/
captures/capture_001/{metadata.json,config_snapshot/,raw_bag/,capture.log}
captures/capture_002/...
jobs/job_ID/{job.json,input_bags/,work/,tool.log}
result/{calib.json,lidar_camera.yaml}
validation/{inputs.json,review.json}
```

The states are CREATED → CAPTURING → CAPTURED → PREPROCESSED → INITIALIZED → CALIBRATED → IMPORTED → VALIDATED. Capture failure returns the dataset to its preceding stable state and retains the rejected raw capture. Backend restart marks unfinished captures/jobs interrupted. Later tool failures retain the previous successful stage, allowing a retry in a new job directory. Captures cannot be appended once preprocessing succeeds; create a new dataset instead.

1. Enable and start camera/LiDAR. Import valid intrinsics before creating a dataset. Match the fixed camera configuration throughout a dataset.
2. Create a calibration dataset. Capture several static views with textured surfaces and varied depths visible to both sensors. Keep the rigid rig and scene stationary during each short capture. Use at least several seconds to get representative rates.
3. Stop each capture. The wrapper checks PointCloud2, Image and CameraInfo counts/types and image rate, and hashes finalized bag files. IMU is not required for these calibration captures.
4. Preprocess. The wrapper copies each accepted bag into a job's input directory, passes explicit topics/intrinsics/intensity, and leaves originals untouched. Static integration is used for Mid-360; dynamic integration and automatic topic guessing are not enabled.
5. Manual alignment opens on the **server desktop**, not in a remote browser. Use upstream's correspondence/alignment tools, explicitly save the initial guess, then close normally. A successful process exit without saved `init_T_lidar_camera` does not advance the state.
6. NID fine calibration runs from a copy of the manual stage. The upstream viewer needs an available OpenGL DISPLAY. A browser on another PC does not stream this native window.
7. Stop acquisition and Import result. The importer validates the seven finite transform values, unit quaternion, camera model, and exact intrinsic/distortion parameters against the dataset snapshot. It preserves the original calib.json, archives the prior canonical YAML, and writes `T_lidar_camera` without inversion. Imported results are **calibrated=true, validated=false**.
8. Independently evaluate held-out overlays and reprojection residuals. A validation interface is stored in validation/inputs.json containing projection, distortion, geometry, transform, offset and source datasets. There is no overlay renderer or colorizer in this change. Record the external validation evidence in the panel only after conducting those checks. VALIDATED is an explicit operator attestation; optimization alone never sets it.

Copy an entire completed calibration directory between Jetson and workstation, preserving snapshots and hashes. On the workstation, select matching local camera configuration/intrinsics before import. The data paths used by jobs are relative to the dataset; no raw bag rewrite is needed. Manual and optimizer work directories are distinct so successful outputs are never overwritten by retries.

## Fixed API surface

- Existing `/api/action`: `camera_start`, `camera_stop` (no arbitrary config/commands).
- GET `/api/camera/preview`: bounded, fresh JPEG; 404 when unavailable; no-store.
- POST `/api/camera/intrinsics`: bounded ROS YAML content, validated with safe YAML parsing; no arbitrary path.
- GET/POST `/api/calibrations`; GET `/api/calibrations/{id}`.
- POST `/api/calibrations/{id}/action`: capture_start, capture_stop, preprocess, initial_guess_manual, calibrate, import, validate, cancel.
- GET `/api/calibrations/{id}/logs/{job_id}`: bounded tool log.

All mutations use the existing serialized service lock, process groups and fixed argv constructors. The existing no-login local-network policy remains unchanged. Camera/calibration raw data and calibration history are excluded from Git.

## Sources inspected

- [gscam2 source and parameters](https://github.com/clydemcqueen/gscam2)
- [tiscamera tutorial](https://www.theimagingsource.com/en-us/documentation/tiscamera/tutorial.html)
- [Koide calibrator programs](https://koide3.github.io/direct_visual_lidar_calibration/programs/)
- [ROS camera calibration](https://docs.ros.org/en/ros2_packages/humble/api/camera_calibration/)

Pinned source inspections confirmed `gscam_main`, relative image_raw/camera_info names, use_gst_timestamps, camera_info_url; explicit preprocess topic options; results.init_T_lidar_camera and results.T_lidar_camera; graphical manual/NID workflows. Source inspection and mock tests are not physical camera or numerical calibration validation.
