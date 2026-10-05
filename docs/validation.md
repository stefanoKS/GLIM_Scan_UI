# Validation results and history

## Latest recorded software validation — 2026-10-05

The camera-selection implementation now committed as `ba70f6d` was tested on **x86_64**, not Jetson. The prescribed run (`source scripts/env.sh` followed by `.venv/bin/python -m pytest -q --basetemp=.state/pytest`) reported **178 passed, 1 skipped, 1 warning in 88.06 seconds**. The skip is opt-in real NKSR inference; the warning is Starlette's AnyIO BlockingPortal alias deprecation. JavaScript behavior/module syntax, Python compilation, changed shell-script syntax and diff whitespace checks passed.

Coverage includes Auto/preferred camera ordering, dependency/stream failures, cleanup, LiDAR-only fallback, required-camera failures, persisted preferences, actual-profile snapshots, recording/calibration locks, stale monitor generations, measured FPS and previous-error/current-status separation. Native SDK projection tests ran against `pyrealsense2==2.58.2.10647` without physical cameras; its CPython 3.10 ARM64 wheel was downloaded successfully.

A synthetic x86_64 benchmark reduced 1280×720 inverse-Brown map creation from 1.397 seconds to a 0.0234-second median. This is not a Jetson measurement or proof of end-to-end 30 FPS recording. Full details are in [camera selection validation](camera_auto_resolution.md#validation-results-2026-10-05-x86_64).

Remaining target acceptance: native ARM64 installation, the [four physical camera cases](camera_auto_resolution.md#jetson-commands), sustained RGB/LiDAR recording, USB and storage throughput, thermal/memory behavior and successful workstation processing of transferred bags. Jetson does not need CUDA/GLIM/NKSR acceptance for its recording role. Geometry/loop-closure/calibration accuracy requires independent workstation/hardware evaluation.

The sections below preserve **historical** measurements and limitations as recorded. Their older test counts, host addresses and feature descriptions are not the current build's defaults. The documentation refresh did not rerun hardware tests.

## Historical implementation — 2026-10-01 / 2026-10-02

## What was actually tested

Host: Ubuntu 22.04.5 x86_64 PC, Linux 6.8.0-138-generic, ROS 2 Humble, system Python 3.10.12, approximately 15 GiB RAM. CUDA and JetPack are absent. This is **not a Jetson test result**. The Mid-360 was connected at 192.168.1.120 through enp6s0 (host 192.168.1.135) on October 1. It was disconnected on October 2; subsequent tests used recorded bags.

| Check | Result |
|---|---|
| Official SDK + ROS driver source build | Passed locally; driver 1.2.8 / SDK2 1.5.2 |
| GTSAM 4.3a0, gtsam_points 1.2.2, GLIM 1.2.2, glim_ros2 1.2.2, Iridescence 1.0.3 | CPU source builds passed |
| First terminal recording | 29.720 s, 295 PointCloud2 messages, 5,945 Imu messages; 149.1 MiB bag |
| Real browser recording + live GLIM | 43.107 s bag, 431 PointCloud2, 8,622 Imu; clean bag and live dump finalization |
| Live binary browser preview | Visually checked real sensor cloud, roughly 7,500 displayed voxel-filtered points at 3 Hz |
| Offline GLIM | Both real bags processed; valid graph and trajectories saved in independent runs |
| Isolation regression | Replayed recorded data in the offline domain while processing; no timestamp validation errors; raw bag hashes identical |
| Official PLY export | Passed through upstream offline_viewer CLI and through browser action; XYZ + intensity present |
| PCD conversion | Passed on first exported map; all exported points retained |
| Native map_editor | Launched with a copied real dump; log confirmed one submap loaded |
| Native offline_viewer | Launched with a copied real dump; interactive viewer initialized |
| Project tests | 20 passed (configuration, metadata, process crashes/cancellation/descendants, mock lifecycle, API restrictions, binary WebSockets, editing copies, timestamp convention, offline isolation command contract) |
| Architecture flag audit | 157 compiled flags.make files checked; no AVX/SSE/native-x86 flags introduced |
| ARM64 Python dependency availability | All pinned packages and dependencies downloaded as Python 3.10 ARM64/universal wheels; execution still requires Jetson |
| Optimized browser preview, sensor disconnected | Accepted 2,366-point export displayed correctly |
| ARM64 / Jetson / CUDA execution | **Not yet tested**; native build and acceptance scripts supplied |

## Saved results

- `data/sessions/20261001_171141_mid360_first_test/processing/run_002/` is the accepted first terminal result. Run 001 was cancelled/rejected during timestamp diagnosis.
- `data/sessions/20261001_172816_browser_live_test/glim_dump/` is the accepted live result.
- `data/sessions/20261001_172816_browser_live_test/processing/run_002/` is the accepted isolated offline result. Run 001 is marked failed because live topics contaminated its input; its exports are retained under that failed run, not advertised as accepted maps.
- The accepted second export contains 2,366 points with intensity. The first contains 2,250. These are GLIM's filtered optimized outputs from short mostly stationary tests; the full raw bags retain the original sensor data.
- The second accepted trajectory spans about 40 s, with 0.141 m accumulated estimated movement and 0.015 m start/end displacement. These numbers **do not measure mapping accuracy**.

Raw second-bag hashes after replay/offline processing:

```
raw_bag_0.db3  7d4aaece4dc525897aaf1a865518f2c70990a11bd706be8c4bfdc16d13562ac5
metadata.yaml df3bd76ab607b0e4c44bf974eb3e704370615344a5f2ae6537e00609ae382ad9
```

## Corrections made from evidence

1. Pinned Livox PointCloud2 timestamps are **absolute nanoseconds**, despite an internal field named offset_time. Config uses `perpoint_relative_time=false`, scale `1e-9`. No bag/source rewrite was needed.
2. Upstream glim_rosbag also spins live subscriptions. Offline jobs now use another ROS domain and unique remapped live inputs. Direct bag filtering keeps original names.
3. The official driver reported a shutdown bus error on the PC and later hung after sensor disconnection. The manager now tracks entire process groups, waits for graceful completion, and records forced escalation when needed; a wrapper exit cannot abandon its child. The upstream shutdown behavior still deserves a separate fix/long-run test.
4. Local GLFW dependencies needed library/include paths because this PC had no passwordless sudo. No GLIM/Livox algorithm patch and no ARM64-specific source patch were applied.

## Limits and remaining acceptance work

- Native Jetson Orin Nano acquisition build, memory/thermal soak and sustained recording still require the actual target. CUDA/GLIM checks belong to processing-host acceptance, not the recording Jetson. Source-only transfer and `verify_jetson.sh` are provided; the verifier refuses x86_64.
- The native toolkit is built and launch-checked. A walked multi-submap route is needed to validate meaningful manual loop constraints, cross-zone merges and edited-map save/export. These interactions are not falsely reported as completed by the launch smoke tests.
- Native GLIM editors require a server desktop/OpenGL display. The browser can launch/track them but does not stream their windows.
- Raw live preview is not an accumulated optimized world map. Exported PLY has a separate optimized preview.
- ScanContext remains unavailable: optional source inspected, not built/validated; upstream declares a noncommercial dependency. At this early milestone camera/visual-loop/reconstruction/splatting stages were inactive. Parallel camera recording and optional NKSR have since been added; visual-loop and splatting remain outside the current pipeline.
- LiDAR serial was not available through the current wrapper and is null in metadata. Network diagnostics distinguish assigned-host-address, reachability and actual message health; they are not a packet-capture analyzer.

Detailed build/run logs remain under `.state/`, and sensor/job logs under each session. Dependencies are pinned in `dependencies.lock`; Python resolution is pinned in `requirements.lock`.

## Auth removal and record-only review — 2026-10-02

Reviewed commit `55d2566` and retained its no-login HTTP, WebSocket, browser and CLI behavior. Same-origin HTTP/WebSocket rejection remains tested. No token is created for fresh test environments.

Added an explicit **Start Record-only Session** action that starts the driver and recorder without invoking GLIM or checking its preset. Tests cover missing GLIM binaries, an irrelevant CUDA preset, duplicate-start protection, finalized metadata, and the absence of a GLIM process. A missing GLIM installation is rejected before a combined mapping action starts the driver or creates a bag.

The browser review used an isolated mock repository under `.state/record-only-browser-review/`, with no GLIM installation. It opened without login, created a session, started record-only with the CUDA preset selected, kept GLIM stopped, and finalized the session as recorded with `glim_live=false` and `bag_finalized=true`. This was a mock lifecycle check; the disconnected sensor was not contacted and production sessions were not changed.

`bootstrap_jetson.sh --record-only` skips GLIM and CUDA setup; `verify_jetson.sh --record-only` checks acquisition prerequisites without requiring GLIM. Selective fetching of the two official Livox repositories was exercised on this PC. The full fresh ARM64 installation and live record-only hardware acceptance remain to be run on Jetson.

## Semantic capture and simplified UI regression

See [capture system analysis](capture_refactor_analysis.md) for the earlier module review, capture failure tests, browser acceptance, real 1.15 GB project export/import → native GLIM → PLY replay, raw-file hash verification and remaining disconnected-sensor/Jetson checks. The user's current camera calibration/configuration was preserved.

## Ethernet auto-detection and display orientation

89 regression tests passed (`.state/orientation-final-tests.log`), including Ethernet exclusion/ambiguity, level/inverted/tilted gravity, motion/stale rejection, and persisted orientation/reset without configuration changes. On the connected PC, auto-detection selected enp6s0 / 192.168.1.135; Mid-360 streamed about 10 Hz and IMU about 200 Hz. Browser Orient succeeded with real IMU measurements; Reset was also verified and left at original orientation. Evidence is in `.state/ethernet-orientation-acceptance.json`. No recording or GLIM job was started for this check.

## Optional calibration native build — 2026-10-03

Built and installed `direct_visual_lidar_calibration` on the x86_64 Humble PC with the system Ceres 2.0 and pinned GTSAM 4.3. The tracked native compatibility patch preserves Sophus right-multiplicative SE(3) updates through the older Ceres API and uses GTSAM's pointer type for the continuous-time ICP/GICP factors. Patch application and repeat-install detection were checked against the pinned source. A native finite-difference check passed for the SE(3) Jacobian at five poses, along with unit-quaternion, zero-update and Ceres parameter-block checks.

ROS lists the installed package executables; `preprocess`, `initial_guess_manual` and `calibrate` passed `--help` launch checks. The 69 camera, calibration, capture and API tests passed across the suite run and the corrected-test rerun. One stale API test was updated to account for automatic live-preview startup while still verifying that rejected mapping requests start no capture processes. No calibration timing, live-preview behavior, camera configuration or measured intrinsics were changed. Physical camera–LiDAR calibration accuracy, the newer Ceres branch and ARM64 builds remain unvalidated.

## PC dense preset — 2026-10-03

Detected Ryzen 5 5600G (6 cores / 12 threads), 16 GB RAM and RTX 5060 (8 GB VRAM); installed GLIM reports CUDA disabled. Added `pc_dense` with CPU modules and selectable, persisted mapping quality. Forty existing capture/core/API tests and three new preset-selection, CPU-compatibility and invalid-preset tests passed.

Reprocessed the real bag from `20261003_140604_Scan_2026-10-03_14_06_04` in `.state/pc-dense-validation`, preserving the session's existing outputs. Native processing completed successfully. The single submap retained 9,205 points versus 2,102 in the baseline result (approximately 4.4×). This short-bag check establishes compatibility and increased retained detail, not long-route performance or improved geometric accuracy.
