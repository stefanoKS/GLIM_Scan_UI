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

This optionally installs system GStreamer, cv_bridge and OpenCV dependencies, fetches pinned gscam2/tiscamera sources and builds into this project. It does not install pip opencv-python. tiscamera uses a project-local prefix including its static data, V4L2/libusb support, and no GigE backend or GUI build. The normal installer installs a USB/video udev rule and adds the current user to the video group; reconnect and log out/in afterward. `--skip-system` uses existing prerequisites/permissions. It does not alter firmware, enable triggers or modify networking. Do not run the dashboard as root.

The vendor has placed tiscamera in maintenance mode and announced EOL in 2029. This implementation uses the requested tiscamera interface and keeps the publisher boundary replaceable. See the [vendor repository](https://github.com/TheImagingSource/tiscamera).

On a workstation with the existing GLIM dependencies:

```bash
scripts/install_calibration.sh
source scripts/env.sh
ros2 pkg executables direct_visual_lidar_calibration
```

The optional calibrator build needs Ceres, OpenCV, PCL, GTSAM and Iridescence. No SuperGlue package, weights or matching command is installed/invoked by these wrappers. Upstream may ship its unused matching script as part of its package; the project API exposes only preprocess, initial_guess_manual and calibrate. Their source revisions are pinned in dependencies.lock. The calibration package was built and its three exposed commands passed `--help` launch checks on the x86_64 Humble PC; numerical calibration on physical reference views is still unvalidated. These installers require sudo for apt prerequisites. The acquisition stack (tiscamera, gscam2, GObject Python, cv_bridge) has been built and tested on the x86 Humble PC. Native ARM64 builds and physical calibration accuracy still require separate acceptance.

### Ceres / GTSAM build compatibility

The installer applies `patches/direct-visual-lidar-calibration-native-compat.patch` to the pinned calibrator. Ubuntu 22.04's Ceres 2.0 does not provide `ceres/manifold.h`; the patch uses its `LocalParameterization` API with the same Sophus SE(3) update and Jacobian. Ceres 2.1+ retains the upstream manifold implementation. The patch also follows GTSAM's shared-pointer type to accommodate the pinned GTSAM 4.3 build.

If a previous installation failed at `ceres/manifold.h`, rerun `scripts/install_calibration.sh --skip-system` when the apt prerequisites are already installed. The patch is applied only once; conflicting source edits stop the installer instead of being overwritten. The normal installer remains appropriate for a fresh host. Unused GLIM CMake options and the ament header-install warning are not the missing-header failure.

## Physical DFK configuration

`config/camera/dfk33ux287.yaml` contains the model/name/serial, frame and topics, width/height, configured and expected FPS, timestamp preference, preview bounds, calibration paths, optional-camera policy and offset. The configured DFK33UX287 uses **720×540 at 15 Hz**. These settings were verified on the attached PC camera in the earlier acquisition acceptance; recheck the actual Jetson device and USB connection.

The tracked pipeline was verified with the PC camera and preserves the user's current camera configuration and measured intrinsics. Verify a different device or changed pipeline before marking it validated. With the DFK physically attached by USB3:

```bash
source scripts/env.sh
tcam-ctrl --list
# Replace CAMERA_SERIAL with the listed serial:
tcam-ctrl --caps CAMERA_SERIAL
gst-inspect-1.0 tcambin
scripts/check_camera.sh CAMERA_SERIAL
```

Inspect supported formats, dimensions and rates. Validate a source/conversion pipeline that produces the requested raw RGB geometry for gscam2 (which supplies its own sink). Store the exact validated pipeline in **local trusted YAML**, along with matching serial, geometry, FPS and `pipeline_validated: true`. No HTTP endpoint accepts or modifies pipelines or arbitrary command strings. The gscam2 ROS parameters are written as YAML, avoiding shell parsing. The configured `serial_number` is used for device diagnostics; the validated pipeline must select that same serial. If unset, recorded serial metadata remains null rather than claiming an identification.

Enable **Include camera in scans** in Settings. START SCAN starts the camera automatically and requires its image and CameraInfo streams before the combined bag starts. RECORD CAMERA works independently of that preference. Engineering status and low-level camera controls remain in Advanced Diagnostics. Missing intrinsic calibration is allowed for acquisition; it is not labelled calibrated. Preview has separate processes/endpoints, so a preview failure does not stop raw recording.

Hardware acceptance commands after the camera is publishing:

```bash
source scripts/env.sh
ros2 topic info /camera/image_raw -v
ros2 topic hz /camera/image_raw
ros2 topic echo /camera/camera_info --once
ros2 topic echo /camera/image_raw --once --field header
scripts/fm.py status
```

Use configured topic names if changed. A camera-enabled scan requires healthy images and CameraInfo; disable camera inclusion for LiDAR-only acquisition. `required_for_mapping` still controls the legacy GLIM health check, but does not permit a one-click combined capture to silently omit requested images. Advanced Stop Camera retains its legacy independent behavior; normal capture uses STOP SCAN/STOP CAMERA to finalize recording.

## Intrinsics import

No fake matrices are shipped. The current canonical intrinsic YAML contains the user's measured calibration; a fresh or reset configuration may contain a missing-calibration marker. Import a measured standard ROS camera calibration YAML through the camera panel or replace the configured local file before starting the camera. The UI import archives previous files and invalidates the active extrinsic association; historical datasets and session snapshots remain intact.

Supported models are plumb_bob with five distortion terms, equidistant/fisheye with four, and omnidir with four plus a measured `xi` extension. Intrinsic files must contain K, R, P, distortion, image_width and image_height. Geometry must match configured acquisition. Omnidir's xi is passed explicitly to preprocessing because standard CameraInfo does not carry it. Unknown models are rejected rather than guessed.

For browser-based camera intrinsics, open **Calibrate camera intrinsics** from Calibration (or `/intrinsics.html`). Download the 24 × 16 ChArUco board with `DICT_4X4_250`, 25 mm squares and 20 mm markers. Print at 600 × 400 mm and measure a square with a ruler; printing at "fit to page" without matching the physical size produces an incorrect calibration. With the camera running at its configured resolution, capture at least four distinct views with the board at different angles and positions, then choose **Calibrate & save**. The server detects corners in full-resolution frames, rejects duplicate poses and high reprojection error, and saves a ROS `plumb_bob` calibration at the configured `intrinsics_file`. The running camera is restarted to publish the new CameraInfo. Previous intrinsic files are archived, and any existing LiDAR–camera extrinsic is marked unvalidated because it was tied to the old intrinsic parameters. **Delete intrinsics** archives and replaces the active file with a missing-calibration marker, then restarts the camera. Neither change is allowed during recording or calibration jobs. Recheck intrinsics after changing lens, focus, crop, or resolution; camera intrinsics do not estimate LiDAR–camera extrinsics or timing offset.

If calibration reports high reprojection error, the solver's RMS residual exceeds 12 pixels, or its error/matrix/distortion is nonfinite. The rejected result is not saved. RMS values near this permissive ceiling indicate a poor fit; treat them as low-confidence and recapture or independently verify before precision use. Reset views and recapture at least four sharp images of a rigid, flat board: keep focus fixed, avoid glare and motion blur, vary tilt and distance, and spread the board across the image including its edges. Do not use four nearly identical front-facing views. Four is a minimum, not a guarantee of accuracy; add more varied views if needed. Verify the printed geometry matches the downloaded board and leave camera resolution/crop unchanged throughout capture.

Alternatively, ROS's optional `camera_calibration` package can generate an intrinsic file separately. With a measured checkerboard, supply its actual internal-corner count and square size, for example the command pattern below (replace all placeholders):

```bash
ros2 run camera_calibration cameracalibrator --size COLSxROWS --square METRES \
  --ros-args -r image:=/camera/image_raw -r camera:=/camera
```

Save/export its YAML and import it here. Revalidate resolution, focus and lens before collecting extrinsics.

## Calibration datasets and state

The Calibration wizard persists datasets under `data/calibrations/cal_ID/`. They never appear as mapping sessions:

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

1. Enable camera inclusion in Settings and calibrate/import valid intrinsics. Choose New alignment in Calibration. Match the fixed camera configuration throughout a dataset; Record reference view automatically starts/checks sensors.
2. Create a calibration dataset. Capture several static views with textured surfaces and varied depths visible to both sensors. Keep the rigid rig and scene stationary during each short capture. Use at least several seconds to get representative rates.
3. Stop each capture. The wrapper checks that PointCloud2, Image and CameraInfo topics contain correctly typed messages, and hashes finalized bag files. Calibration reference captures do not require mapping-rate LiDAR or camera streams; keep the scene and rig static during each capture. IMU is not required for these calibration captures.
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
