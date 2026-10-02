# Implementation and validation — 2026-10-01 / 2026-10-02

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

- Native Jetson Orin Nano build, CUDA test, memory/thermal soak and sustained acquisition still require the actual target. Source-only transfer and `verify_jetson.sh` are provided; the verifier refuses x86_64.
- The native toolkit is built and launch-checked. A walked multi-submap route is needed to validate meaningful manual loop constraints, cross-zone merges and edited-map save/export. These interactions are not falsely reported as completed by the launch smoke tests.
- Native GLIM editors require a server desktop/OpenGL display. The browser can launch/track them but does not stream their windows.
- Raw live preview is not an accumulated optimized world map. Exported PLY has a separate optimized preview.
- ScanContext remains unavailable: optional source inspected, not built/validated; upstream declares a noncommercial dependency. Camera/visual-loop/reconstruction/splatting stages remain deliberately inactive.
- LiDAR serial was not available through the current wrapper and is null in metadata. Network diagnostics distinguish assigned-host-address, reachability and actual message health; they are not a packet-capture analyzer.

Detailed build/run logs remain under `.state/`, and sensor/job logs under each session. Dependencies are pinned in `dependencies.lock`; Python resolution is pinned in `requirements.lock`.

## Auth removal and record-only review — 2026-10-02

Reviewed commit `55d2566` and retained its no-login HTTP, WebSocket, browser and CLI behavior. Same-origin HTTP/WebSocket rejection remains tested. No token is created for fresh test environments.

Added an explicit **Start Record-only Session** action that starts the driver and recorder without invoking GLIM or checking its preset. Tests cover missing GLIM binaries, an irrelevant CUDA preset, duplicate-start protection, finalized metadata, and the absence of a GLIM process. A missing GLIM installation is rejected before a combined mapping action starts the driver or creates a bag.

The browser review used an isolated mock repository under `.state/record-only-browser-review/`, with no GLIM installation. It opened without login, created a session, started record-only with the CUDA preset selected, kept GLIM stopped, and finalized the session as recorded with `glim_live=false` and `bag_finalized=true`. This was a mock lifecycle check; the disconnected sensor was not contacted and production sessions were not changed.

`bootstrap_jetson.sh --record-only` skips GLIM and CUDA setup; `verify_jetson.sh --record-only` checks acquisition prerequisites without requiring GLIM. Selective fetching of the two official Livox repositories was exercised on this PC. The full fresh ARM64 installation and live record-only hardware acceptance remain to be run on Jetson.
