# Historical capture workflow analysis and verification

This is evidence from an earlier refactor, not the latest build report. Camera selection, RGB fallback, host deployment enforcement and UI recovery messages have since changed; use [operation](operation.md), [camera selection](camera_auto_resolution.md) and [validation](validation.md) for current behavior/results. Historical test counts and hardware availability below refer only to that pass.

Reviewed against the user's committed camera/import-export changes at `194bbfb`. Source, tests, configuration, installation scripts and operator documentation were inspected before implementation. Changes retain the existing backend/session/process architecture and the user's measured camera intrinsics, pipeline and network configuration. All acceptance artifacts are under the repository's ignored `.state/` directory.

## Workflow ownership

| Files | Responsibility / change |
| --- | --- |
| `ui/backend/factory_mapping/capture.py` (new) | Persistent high-level capture state, semantic actions, host capabilities, asynchronous start/stop, interruption recovery and best-effort live GLIM |
| `ui/backend/factory_mapping/service.py` | Existing sensor/recorder/GLIM orchestration; reused for camera-only acquisition, recorder readiness, required-topic finalization and operation exclusion |
| `ui/backend/factory_mapping/api.py` | Semantic capture/settings/rename endpoints, calibration wizard adapter, compatibility guards and transfer coordination |
| `ui/backend/factory_mapping/commands.py` | Existing fixed ROS commands; camera-only topic selection added; original rosbag/GLIM command paths retained |
| `ui/backend/factory_mapping/storage.py` | Existing SHA-256 project transfer; additional configuration/bag-path validation and imported job recovery |
| `ui/backend/factory_mapping/processes.py` | Existing process groups, shutdown escalation and orphan detection; reused without rewrite |
| `ui/backend/factory_mapping/calibration.py`, `calibration_data.py` | Existing detailed dataset/tool states, measured intrinsic/extrinsic validation and immutable calibration bags; reused |
| `ui/backend/factory_mapping/camera_intrinsics.py` | Compatibility with distribution OpenCV 4.5 ChArUco corner access; actionable error without optional camera dependencies |
| `ui/frontend/index.html`, `style.css`, `capture.js` (new) | Capture / Library / Calibration / Settings navigation; simple capture controls and PiP; low-level controls retained in Advanced Diagnostics |
| `ui/frontend/app.js`, `camera.js`, `intrinsics.html`, `intrinsics.js` | Existing cloud viewer, library, transfer and native tools retained; semantic actions, post-capture names/notes and guided calibration added |
| `scripts/sensor_cli.py` | Terminal recording now uses the same service preparation/finalization, including the enabled camera |

## Findings addressed

- Legacy combined startup stopped rosbag if live GLIM failed. Semantic and compatibility startup now retain raw acquisition; runtime estimator failure is reported without stopping the recorder.
- Recording could be announced before the child initialized, allowing an immediate Stop to require forced termination. Start now waits for the recorder's output/readiness before SCANNING.
- A metadata file alone was insufficient evidence of a complete requested capture. Finalization now checks positive duration and messages on every requested stream, retaining incomplete raw files with a failed outcome.
- Terminal recording included camera topic names without preparing the camera. It now shares the service's acquisition startup.
- Standalone camera capture needed a separate acquisition snapshot and preview handling. It no longer requires LiDAR or a globally enabled scan-camera preference and is excluded from GLIM processing.
- Normal users previously coordinated many independent process controls. The controller owns sequencing and the UI submits semantic actions only. Advanced endpoints remain available with capture/job exclusion guards.
- Calibration preparation could start or stop sensors before rejecting conflicting work. Wizard checks now precede hardware changes.
- Project upload could write data during capture; exports lacked staging-space checks. Capture exclusion and upload checks now apply before/during transfer, and export checks available staging space.
- Archive hashes did not validate bag-internal paths or acquisition schema. Both are checked before publishing an imported session; running transferred jobs are marked interrupted. Edited-map paths are resolved on the receiving host.
- Distribution OpenCV 4.5 uses `chessboardCorners` instead of the newer getter. Both interfaces are supported. Camera-specific solver tests no longer prevent a camera-free record-only install from collecting the rest of its tests.
- Final startup found abandoned driver/preview processes from the earlier hardware run. Their saved identities and command paths were verified before cleanup; no recorder was active. Capture now explains when recovery or another job blocks starting.
- Existing documentation still described a null camera pipeline, missing intrinsics, manual session setup and uninstalled PC camera software. The guides now describe the current configuration, one-click flow and native Jetson installation boundary.

## Verification

- Regression suite: 82 tests passed in the final full regression run (40.07 seconds). This covers preflight failure, duplicate start, immediate stop, live GLIM startup/runtime failure, recorder failure, shutdown/restart, camera-only mode, record-only host, transfer/reprocessing, legacy APIs, camera and calibration behavior. Logs: `.state/capture-full-tests.log` and `.state/capture-transfer-tests.log`.
- Browser mock acceptance: START SCAN → SCANNING → STOP SCAN → finalized/automatically processed Library run; post-capture rename/notes; RECORD/STOP CAMERA with preview while scan RGB is disabled; Calibration guidance and collapsed Advanced Diagnostics. Mock data is visibly labelled and isolated in `.state/capture-ui-review/`.
- Real archived session `20261002_142915_combined_rgb_reliable_acceptance`: exported a **1,153,084,719-byte** project, imported into `.state/refactor-acceptance/`, processed with native CPU GLIM (exit 0, no timestamp validation errors), and exported a **36,338-byte PLY**. Both source and imported raw bag hashes remained unchanged. Evidence: `.state/refactor-acceptance/acceptance.json` and `.state/refactor-acceptance.log`.
- Native `offline_viewer` (Edit Map / Merge Maps host) and `map_editor` (Clean Map / segmentation) launched against independent copies of the imported result. Evidence: `.state/refactor-acceptance/native-tools.json`. This checks launch/integration, not manual geometric accuracy or a completed multi-map merge.
- Imaging Source Python check passed: PyGObject 3.42.1, Tcam 1.0, GStreamer 1.20.3, system OpenCV 4.5.4 and ROS cv_bridge. JavaScript/Python/shell syntax checks and `git diff --check` passed.
- Portability audit: 190 build flag files, no prohibited flags; the two tiscamera Intel-only SSE4.1 targets are architecture-dispatched upstream. `arm64_build_verified` remains false.

## Remaining hardware and product boundaries

Sensors are disconnected by the user's confirmation. No fresh live capture is claimed in this pass. Reconnect Mid-360 and DFK33UX287 for the combined 720×540 capture, disconnect/reconnect and sustained-write checks. Replay of a short static recording is not walking-route, loop-closure or factory accuracy acceptance.

An actual Orin Nano 8 GB is still required to verify native ARM64 builds, USB throughput, CUDA/CPU performance and thermal/memory behavior. The installer retains system Python/ROS ABI, builds native dependencies, and supports acquisition without GLIM; PC binaries/venvs must not be copied to Jetson. Use the commands in `installation_jetson.md` and `camera_calibration.md`.

The optional `direct_visual_lidar_calibration` native package and numerical extrinsic results are not certified by these checks. The wizard preserves manual initial alignment and independent validation; it cannot manufacture either. Those steps require the optional workstation installation, a usable server desktop and physical reference captures. Automatic ScanContext loop detection remains unavailable; manual loop closure and native segmentation/editing remain supported. Final RGB colorization, meshing and automatic time-offset estimation remain outside the existing implementation.
