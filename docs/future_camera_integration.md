# Camera roadmap and remaining integration

This file originally described camera work as entirely inactive. That is historical: the current build supports D405/DFK RGB recording, automatic camera resolution, camera-only bags, profile-specific intrinsics and camera–LiDAR calibration datasets. See [current camera setup](camera_calibration.md) and [selection/verification](camera_auto_resolution.md). Include RGB now defaults on; GLIM still consumes only Mid-360 LiDAR/IMU.

Optional workstation [NKSR surface reconstruction](nksr.md) is also implemented as a separate derived geometry stage. The Jetson remains recording-only.

## Still outside the current acquisition pipeline

- Hardware exposure triggering/synchronization and automatic camera time-offset estimation. Current receipt/driver timestamps must not be described as synchronized exposure times.
- Occlusion-aware offline RGB projection/colorization and texture generation using optimized LiDAR poses and independently validated camera alignment.
- Visual place recognition with LiDAR geometric verification before adding trusted pose-graph constraints. Camera imagery does not currently drive GLIM.
- RGB-based reconstruction and Gaussian-splat visualization as separate derived products.

Potential future interfaces should preserve timestamp clock domain, trigger provenance, camera identity, calibration hashes, transform convention and time offset. Raw recordings and completed calibration datasets must remain immutable; each colorization/reconstruction job should have its own versioned output. Existing recording-only deployment and camera fallback/locking must remain intact.
