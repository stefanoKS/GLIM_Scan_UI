# Portable RGB acquisition and Jetson verification

The Jetson remains a recording host; mapping and surfacing run on the workstation.
The camera-selection build is committed as `ba70f6d` (developed against `a46730e`) and preserves its recording-only policy,
project transfer, DFK stack and existing GLIM pipeline. RGB is never passed to GLIM.

## Root causes

The saved profile was treated as mandatory, only that camera was detected, and its
failure blocked the entire scan. Camera monitoring loaded the preference again in
its subprocess, so selecting a fallback also requires passing the resolved camera
to the monitor and preview. The D405 generated 921,600 individual Python-to-SDK
projection calls before publishing the first 1280×720 image.

## Selection state machine

1. Read the preference: `auto`, `d405` or `dfk33ux287`. Old JSON values stay valid.
   Malformed or unknown preferences use Auto with a logged/status warning, leaving
   the file intact until the operator explicitly saves a preference.
2. Detect both configured USB devices independently. Presence is not readiness.
3. Try D405 then DFK for Auto/D405 preference; DFK then D405 for DFK preference.
   A calibration capture only tries the camera frozen into its dataset.
4. Reuse a healthy candidate if appropriate; otherwise stop preview, monitor and
   publisher through ProcessManager, then remove stale health/image state. An
   orphan whose identity cannot safely be adopted blocks acquisition for recovery.
5. Check dependencies (six-second total candidate budget), start the publisher,
   monitor and optional preview, then allow six seconds for fresh frames, measured
   rate within the existing ±30% health tolerance, expected dimensions/frame ID,
   and fresh CameraInfo. A publisher exit fails immediately. No SDK extraction or
   USB enumeration alone qualifies a candidate. Process cleanup completes before
   another candidate is allowed to publish on the shared topics.
6. On failure, log the reason, clean up all candidate processes, and try the next.
   On success, snapshot the resolved profile, calibration files, stream dimensions,
   measured preflight FPS and preference/fallback provenance before rosbag starts.
7. If neither works: Start Scan records only LiDAR/IMU, adds a prominent warning,
   and disables camera topics only in the acquisition copy. Camera-only recording
   and calibration fail clearly. Global Include RGB remains unchanged.
8. During recording/calibration the camera is locked. No fallback or profile edit
   can swap it. Calibration datasets remain bound to their camera across captures.
9. At stop, bag message counts establish `rgb_recorded` and measured recorded FPS
   in session metadata. `active_config.json` remains the immutable preflight snapshot
   (`rgb_recorded: false` there means not yet established, not a final result).

`/api/status.camera_selection` separates preferred profile, active publisher/profile,
fallback reason, USB detection, dependency results and stream health per candidate.
A generation token prevents a previous monitor's health file from validating the
next publisher. Historical capture errors appear separately from current camera
readiness in the UI. Detailed reasons are in Advanced / Diagnostics.

## D405 performance and image semantics

The pin is `pyrealsense2==2.58.2.10647`. The CPython 3.10 ARM64 wheel was downloaded
successfully; the same pinned version was installed for native projection tests
on this x86_64 workstation. See the [SDK release](https://github.com/realsenseai/librealsense/releases/tag/v2.58.2).

No-distortion images bypass remapping. Ordinary Brown and inverse Brown use
vectorized float32 coordinate maps. Inverse Brown reproduces the SDK's separate
projection formula, not OpenCV's ordinary Brown coefficients. Maps are created
once per publisher and converted to OpenCV fixed-point maps for native remapping.
Two OpenCV worker threads bound CPU oversubscription. Published CameraInfo continues
to describe rectified RGB with zero distortion; original factory coefficients and
serial stay in the calibration file. Calibration is saved after pipeline.start,
before the first image; there is no second extraction subprocess on the normal path.

The numerical test checks representative center, edge and corner pixels against
`rs2_project_point_to_pixel` from the pinned SDK, with 0.0003-pixel tolerance for
float32 rounding, below OpenCV's 1/32-pixel interpolation quantization. Existing
inverse-model deprojection round-trip checks also pass.

Synthetic **x86_64 only** measurements at 1280×720: old map construction 1.397 s;
vectorized map median 0.0234 s (10 runs); native remap median 0.00157 s (100 frames).
These are not Jetson measurements or ROS/USB throughput claims. D405 now requests
30 FPS; readiness uses measured delivery. DFK remains 15 FPS. The publisher logs
resolve, pipeline.start, calibration, map creation and first-frame timings, SDK
version, resolution and total time to the first published image.

## Jetson commands

Use the supported JetPack 6 / Ubuntu 22.04 / ROS 2 Humble environment. Stop the
dashboard before these commands; diagnostics acquire the same backend lock and
will refuse to interfere with an active application. Configure the actual device
serials in the trusted camera YAML files and install USB permissions/vendor stacks.

```bash
source scripts/env.sh
scripts/install_d405.sh
scripts/verify_jetson.sh --record-only
```

Connect only the named cameras before each command. These tests temporarily use
Auto in memory and do not overwrite saved preferences:

```bash
# D405 only: expect D405, 1280x720, measured rate near 30 FPS, CameraInfo present.
.venv/bin/python scripts/diagnose_camera.py --case d405 --seconds 30

# DFK only: expect DFK, 720x540, measured rate near 15 FPS, D405 fallback reason.
.venv/bin/python scripts/diagnose_camera.py --case dfk --seconds 30

# Both: expect D405 under Auto.
.venv/bin/python scripts/diagnose_camera.py --case both --seconds 30

# Neither: expect no active camera and explicit unavailability reasons.
.venv/bin/python scripts/diagnose_camera.py --case neither --seconds 30
```

The JSON reports in `.state/camera_diagnostic_CASE.json` include architecture,
Ubuntu/Jetson release, installed SDK version, both USB detections and negotiated
speeds, preference, actual camera configuration, measured stream samples,
CameraInfo, fallback reasons and the publisher's timing logs. `passed` checks the
expected USB set and camera selection plus final stream health. Inspect all samples
for stability; it is not a sustained-recording certification. The script also runs
on x86_64 but reports that architecture without claiming Jetson verification.
`verify_jetson.sh --camera-case=d405` can run the same probe after its ARM64 checks.

Repeat D405-only with `--preference dfk33ux287`, and DFK-only with `--preference d405`
to verify saved-preference fallback. For each physical case, attach Mid-360 and run
`scripts/record_test.sh --name camera_acceptance --seconds 30`, then inspect the
session metadata and bag topic counts. With neither camera, Start Scan must succeed
with LiDAR/IMU only; Record Camera and New alignment must fail. Repeat at the intended
route duration on target storage while observing disk use, frame/message counts,
USB stability, temperature and memory. Transfer that completed project and process
it on the workstation. Full-resolution uncompressed RGB needs substantial storage;
check actual disk throughput under concurrent LiDAR recording.

## Changed files and tests

- Configuration: `config/system.yaml`, `config/camera/d405.yaml`.
- Backend: `config.py`, new `camera_selection.py`, `service.py`, `camera.py`,
  `realsense_camera.py`, `ros_nodes.py`, `api.py`, `calibration.py` under
  `ui/backend/factory_mapping/`.
- UI: `ui/frontend/index.html`, `camera.js`, `capture.js`, new `camera-status.js`.
- Installation/diagnostics: `scripts/install_d405.sh`, `scripts/verify_jetson.sh`,
  new `scripts/diagnose_camera.py`.
- Tests: new `tests/test_camera_selection.py`, new `tests/test_camera_status.mjs`,
  updated `tests/test_d405.py`, `tests/test_camera.py`, `tests/test_calibration.py`.
- Documentation: `README.md`, `docs/camera_calibration.md`, this document.

Tests cover candidate order, legacy/Auto/bad preference persistence, dependency and
stream failures, immediate exit, cleanup, LiDAR-only fallback, required camera
failure, actual-profile snapshots, recording/calibration locks, separate calibration
paths, stale monitor generations, status polling, measured 30 FPS/staleness, native
projection agreement, and separation of previous errors from live UI readiness.
Ordinary tests need no physical camera; native numerical tests skip only if the
optional RealSense SDK is absent. Jetson hardware remains unverified here.

## Validation results (2026-10-05, x86_64)

Executed the requested command:

```bash
source scripts/env.sh
.venv/bin/python -m pytest -q --basetemp=.state/pytest
```

Result: **178 passed, 1 skipped, 1 warning in 88.06 seconds**. The skip is the
opt-in real NKSR inference test (`RUN_NKSR_INTEGRATION`); the warning is Starlette's
AnyIO BlockingPortal alias deprecation. The RealSense projection tests ran against
2.58.2.10647, without physical devices.

`tests/test_camera_status.mjs` passed using the bundled Node runtime. Every
`ui/frontend/*.js` file passed module syntax checks. Python compilation, shell
syntax checks for the changed shell scripts and `git diff --check` passed.
No Jetson hardware recording or sustained 30 FPS result is claimed.
