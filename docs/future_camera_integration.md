# Future camera integration — intentionally inactive

Phase 1 contains no active camera source, camera recording or visual SLAM. `camera.enabled` must remain false.

Future hardware: Imaging Source DFK global-shutter camera, externally triggered by Jetson hardware timing. Preserve original frame timestamps and trigger provenance alongside raw images. Do not substitute receipt time for capture time.

Planned interfaces:

- `CameraFrame(timestamp, clock_domain, trigger_id, image_path, intrinsics_id)`.
- `Calibration(intrinsics, distortion, T_lidar_camera, time_offset, provenance)`; FAST-Calib2 may supply extrinsics once its compatibility is checked.
- Offline colorization: hardware timestamp → calibrated camera poses from optimized LiDAR trajectory → occlusion-aware RGB projection → XYZRGB.
- Visual place recognition: ORB/DBoW → candidate pairs → LiDAR geometric verification → a supported GLIM pose-graph extension. Candidates alone never create trusted constraints.
- Reconstruction: optimized poses and RGB → TSDF/VDBFusion mesh. Optional Gaussian splatting uses calibrated images and optimized camera poses as a separate derived product.

Keep raw acquisition immutable and version each calibration/colorization/reconstruction job independently. Camera synchronization, calibration and accuracy validation require their own milestone.
