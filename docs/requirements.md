You are building a new production-oriented repository for factory 3D mapping.

The target hardware is an **NVIDIA Jetson Orin Nano, ARM64/aarch64**, with a **Livox Mid-360**. The first milestone must use **LiDAR + the Mid-360 built-in IMU only**. Do NOT add the camera to the active SLAM pipeline yet.

The long-term project will later add an Imaging Source DFK camera for RGB colorization, visual loop detection, LiDAR-camera calibration, surface reconstruction, and Gaussian-splat visualization. Architect the repository so those can be added later, but do not let them complicate Phase 1.

Use the official repositories whenever possible:
- `koide3/glim`
- `koide3/glim_ros2`
- `Livox-SDK/livox_ros_driver2`
- later, optionally `koide3/glim_ext` for ScanContext/loop detection

Do not substitute LIO-Livox, FAST-LIO2, FAST-LIVO2, or another SLAM backend for GLIM in Phase 1.

The desired initial pipeline is:

Mid-360
→ PointCloud2 + IMU
→ ROS2
→ rosbag2 recording
→ GLIM LiDAR-inertial SLAM
→ GLIM dump / optimized map
→ point-cloud export
→ browser-based user interface

The system must support both:

LIVE MODE:
Mid-360 → ROS2 → GLIM → live map preview

and:

OFFLINE MODE:
Mid-360 → rosbag2 → copy bag to workstation or process locally → GLIM rosbag processing → optimized result

The Jetson will primarily be the acquisition machine. It must not be assumed that the Jetson will perform the final heavy processing.

## 1. Inspect the machine before installing anything

At the beginning, inspect and record:

`uname -a`
`uname -m`
`lsb_release -a`
`cat /etc/nv_tegra_release`
`nvcc --version`
installed CUDA version
available RAM
available disk
Jetson model
ROS distribution
Python version
colcon version
network interfaces

Confirm that this is `aarch64`.

Do not assume x86_64.

Create a human-readable environment report at:

`docs/environment_report.md`

If this Jetson is running JetPack 6 / Ubuntu 22.04 / ROS2 Humble, use that environment.

If ROS2 is already installed, preserve the existing installation.

Do not perform destructive OS upgrades.

## 2. ARM64 / Jetson build requirements

GLIM and all dependencies must be compiled in a way compatible with ARM64.

In particular:

- DO NOT enable x86 AVX flags.
- Use `BUILD_WITH_MARCH_NATIVE=OFF` by default.
- Do not introduce `-mavx`, `-mavx2`, SSE-specific flags, or other x86-only compile options.
- Detect CUDA instead of hard-coding a desktop CUDA version.
- Prefer CUDA acceleration if the existing Jetson CUDA toolchain supports it.
- Provide a CPU fallback if CUDA GLIM cannot be built.
- Avoid excessive parallel compilation that causes OOM on Orin Nano.
- Default build parallelism to something conservative such as 2–4 jobs based on detected RAM.

Follow current upstream GLIM source-build requirements. Inspect the current official documentation rather than assuming dependency versions.

Pin known-working Git SHAs after successful builds so later installations are reproducible.

Store dependency revisions in:

`dependencies.lock`

or an equivalent machine-readable manifest.

## 3. Repository layout

Create a clean repository approximately like:

factory_mapping/
    README.md
    LICENSE
    .gitignore

    docs/
        architecture.md
        installation_jetson.md
        operation.md
        troubleshooting.md
        environment_report.md
        future_camera_integration.md

    config/
        system.yaml
        livox/
        glim/
        ui/

    scripts/
        check_system.sh
        bootstrap_jetson.sh
        install_livox.sh
        install_glim.sh
        build_ros2.sh
        run_system.sh
        stop_system.sh
        diagnose.sh

    ros2_ws/
        src/
            factory_mapping_bringup/
            factory_mapping_monitor/
            factory_mapping_preview/

    ui/
        backend/
        frontend/

    data/
        sessions/

    external/
        README.md

Do not commit build/, install/, log/, rosbag data, GLIM dumps, or large point clouds.

Use either a `.repos` file with `vcs import` or reproducible installation scripts for external repositories.

## 4. Livox Mid-360 integration

Use official `livox_ros_driver2`.

Support at minimum:

LiDAR PointCloud2
IMU

Prefer standard ROS2 messages at the boundary of our own packages.

The actual topic names must be discovered from the installed official driver and exposed through configuration rather than hard-coded throughout the source.

Create:

`config/livox/mid360.yaml`

containing at minimum configurable:

LiDAR IP
Jetson host IP
PointCloud2 topic
IMU topic
frame IDs
publish rate/settings that the Livox driver exposes
ROS_DOMAIN_ID

The UI must later allow the important network values to be edited safely.

Provide diagnostics that clearly distinguish:

- Ethernet interface missing
- Mid-360 unreachable
- Livox driver not running
- LiDAR topic exists but has no messages
- IMU topic exists but has no messages
- packet/message rate abnormal

Do not silently report "connected" merely because a ROS process exists.

## 5. GLIM integration

Use GLIM as the Phase 1 SLAM backend.

Support:

A. live `glim_rosnode` processing

B. direct/offline rosbag processing using the upstream GLIM rosbag workflow where available

Do not rewrite GLIM itself unless absolutely necessary.

Keep our GLIM configuration separate from the upstream repository.

Create project-local presets such as:

`config/glim/jetson_gpu/`
`config/glim/jetson_cpu/`
`config/glim/offline_quality/`

Start from official GLIM presets and document every modification.

Important: for the first milestone, do not aggressively tune parameters without evidence.

Expose common settings in the UI/config but keep advanced GLIM JSON files editable on disk.

Ensure closing/stopping GLIM gracefully saves its dump.

Create session-specific GLIM output directories instead of relying permanently on `/tmp/dump`.

If upstream GLIM requires `/tmp/dump`, safely copy/move the completed dump into the session directory afterward.

## 6. Loop closure strategy

Phase 1 must first establish stable GLIM LiDAR+IMU mapping.

Make loop closure modular.

Create configuration:

`loop_closure.enabled`

Initially allow:

`false`

and later:

`scan_context`

Do not make Phase 1 depend on an experimental visual loop-closure module.

After core GLIM works, investigate official `glim_ext` ScanContext support.

If `glim_ext` is sufficiently compatible with the current GLIM version, add it as an optional feature.

Do not patch GLIM heavily just to force ScanContext to work.

The UI should clearly display whether loop detection is:

OFF
SCANCONTEXT
UNAVAILABLE

Later camera/DBoW loop detection will be a different mode. Do not implement it now.

## 7. Recording workflow

Every mapping run must create a session.

Directory format:

`data/sessions/YYYYMMDD_HHMMSS_<session_name>/`

Inside it save approximately:

metadata.json
config_snapshot/
raw_bag/
glim_dump/
exports/
logs/

`metadata.json` should include:

session name
start time
end time
duration
Jetson model
OS
JetPack version
ROS version
GLIM git SHA
Livox driver git SHA
repository git SHA
LiDAR serial number if available
LiDAR IP
ROS topic names
average LiDAR rate
average IMU rate
disk usage
whether GLIM was used live
whether loop closure was enabled
notes entered by user

When recording starts, snapshot all active configuration into the session.

The original rosbag must never be modified by offline processing.

## 8. User interface

Build a lightweight **local web UI**.

Do NOT use Electron.

The UI must run comfortably on Jetson Orin Nano and be accessible from another PC on the same network through a browser.

Recommended architecture:

FastAPI backend
+
minimal HTML/CSS/JavaScript frontend
+
Three.js for point-cloud preview

Avoid a large frontend framework unless there is a compelling reason.

If Three.js is used, package it locally. The mapping UI should not require internet access at runtime.

The FastAPI environment must coexist correctly with ROS2 Humble Python packages on the Jetson. If a virtual environment is used, account for `rclpy` and system ROS packages, for example with an appropriate system-site-packages strategy. Do not break the ROS environment.

The main dashboard should show:

SYSTEM
Jetson model
CPU usage
RAM usage
GPU usage if available
Jetson temperature
disk free
network interface/IP
ROS_DOMAIN_ID

MID-360
connection state
LiDAR topic
LiDAR message rate
point rate if obtainable
IMU rate
latest timestamp

GLIM
stopped / starting / mapping / error
CPU or CUDA mode
current map/session
loop closure mode
runtime
latest error

RECORDING
recording state
elapsed time
bag size
session name
free disk estimate

Provide primary controls:

Start Mid-360
Stop Mid-360

Start Recording
Stop Recording

Start GLIM Live
Stop GLIM

Start Mapping Session
Stop Mapping Session

Process Existing Bag

Open Session

Export Map

System Diagnostics

Do not make the backend execute arbitrary user-provided shell commands.

All process commands should come from a controlled process manager.

## 9. Live 3D preview

Implement a lightweight point-cloud preview suitable for the Jetson.

Do NOT attempt to send the complete Mid-360 cloud at full rate to the browser.

Create a ROS2 preview node:

`factory_mapping_preview`

that subscribes to the selected PointCloud2 topic.

Downsample aggressively for visualization only.

Make configurable:

preview update rate: default 2–5 Hz
maximum displayed points: default around 30k–100k depending on performance
voxel size

Send preview data to the browser efficiently.

Prefer binary WebSocket payloads over huge JSON point arrays.

The browser viewer should support:

orbit
pan
zoom
reset view
point-size adjustment
color by intensity initially

The preview cloud is for visualization only and must never be used as the stored mapping data.

If obtaining the live optimized GLIM map is significantly more complicated than obtaining raw LiDAR, first display raw LiDAR in the viewer and document that limitation.

Do not invent fake GLIM map topics.

Inspect upstream GLIM interfaces first.

## 10. Session browser

The UI should have a session page displaying:

session name
date/time
duration
bag size
GLIM processing state
export state
notes

Provide actions:

Process with GLIM
Reprocess with another GLIM config
Open point-cloud preview
Export
Delete generated derived data

Protect the raw rosbag from accidental deletion.

Deleting the original raw bag should require a separate explicit confirmation mechanism later; it is not required in the first version.

## 11. Offline GLIM processing

Implement a job abstraction so an existing session can be processed after acquisition.

Example:

Raw bag
→ GLIM offline
→ saved GLIM dump
→ optimized map
→ export

The UI must stream processing logs.

Provide:

running
completed
failed
cancelled

states.

Do not block the FastAPI event loop during GLIM processing.

Use subprocess/job management correctly.

Store stdout/stderr per job.

The same bag should be reprocessable with multiple GLIM configurations without overwriting previous results.

For example:

processing/
    run_001/
    run_002/

## 12. Point-cloud export

Investigate the current official GLIM mechanism for exporting the optimized map.

Do NOT invent an export CLI that does not exist.

If GLIM already provides an official exporter/viewer API, use it.

Target formats, in priority order:

PLY
PCD

Later we will add LAS/LAZ and Houdini `.bgeo.sc` conversion.

At minimum preserve:

XYZ
intensity if available

Do not downsample the archival/raw source data just because the browser preview is downsampled.

## 13. Map quality tools

Create basic non-metrology diagnostics.

At minimum support:

trajectory duration
trajectory length if obtainable
start/end displacement
bounding box
number of points
map/session size

Later we will add quantitative control-point accuracy.

Do not claim sub-centimeter accuracy from SLAM statistics alone.

## 14. Camera future-proofing

Create interfaces/config placeholders but NO active camera dependency yet.

Document the future pipeline in:

`docs/future_camera_integration.md`

Future hardware:

Imaging Source DFK camera
global shutter
externally triggered from Jetson hardware timing

Future pipeline:

DFK
→ hardware timestamp
→ camera intrinsics
→ FAST-Calib2 LiDAR-camera extrinsics
→ RGB projection onto final optimized LiDAR map
→ XYZRGB

Later visual SLAM functionality:

camera
→ ORB/DBoW visual place recognition
→ visual loop candidate
→ LiDAR geometric verification
→ GLIM pose graph

Later visualization:

optimized LiDAR poses
+
RGB camera frames
→ TSDF/VDBFusion mesh

and optionally:

camera images
+
optimized camera poses
→ Gaussian Splatting

Do not implement those now.

## 15. Reliability requirements

The UI backend must survive an individual ROS/GLIM subprocess crash.

Never require restarting the whole UI merely because GLIM stopped.

Prevent multiple incompatible instances of the same process.

On application startup, detect stale PID/state files.

Handle SIGINT/SIGTERM correctly.

Stopping a mapping session should perform graceful shutdown in this order where appropriate:

stop recording
finish rosbag metadata
request GLIM graceful shutdown
wait for dump generation
collect logs/configuration
finalize session metadata

Do not use `kill -9` except as a final forced-shutdown path.

## 16. Logging

Use structured application logs.

Each session gets its own logs.

UI should display recent errors.

Make the complete logs downloadable/viewable locally.

Important errors should explain the likely corrective action rather than only exposing stack traces.

## 17. Mock / development mode

Create a mode that lets the UI be developed without a physical Mid-360.

It should be possible to:

start the UI
browse sessions
test process state transitions
test WebSockets
test mock point data

without requiring the LiDAR.

Do not mix mock data into production mode.

## 18. Tests

Add practical tests for:

configuration loading
session creation
session metadata
process manager
process cancellation
system status parsing
WebSocket preview encoding if applicable

Tests must run on ARM64.

Avoid x86-specific test assumptions.

## 19. Documentation

README must contain a concise quick start.

`installation_jetson.md` must document the exact tested Jetson environment.

`operation.md` must explain this workflow:

Power Mid-360
→ connect Ethernet
→ start application
→ verify LiDAR and IMU health
→ create session
→ start recording
→ optionally start GLIM
→ walk mapping route
→ return near starting point
→ stop session
→ process/reprocess bag
→ inspect result
→ export point cloud

Also document recommended scanning behavior:

move slowly
avoid violent rotations
revisit locations
close loops
keep the Mid-360 unobstructed
record multiple overlapping factory zones rather than one enormous recording

## 20. Implementation order

Work incrementally and keep the repository runnable after each major step.

Implement in this order:

1. Inspect Jetson and create environment report.
2. Create repository skeleton.
3. Get official Livox driver working with Mid-360.
4. Verify PointCloud2 + IMU topics and rates.
5. Implement rosbag session recording.
6. Build/install GLIM safely on aarch64.
7. Process a recorded Mid-360 bag through GLIM from the CLI.
8. Save GLIM outputs into a session.
9. Implement backend process manager.
10. Implement basic web dashboard.
11. Add session creation/start/stop from UI.
12. Add LiDAR/IMU health monitoring.
13. Add lightweight point-cloud browser preview.
14. Add GLIM live controls.
15. Add offline GLIM job processing.
16. Add export workflow.
17. Only after the above is stable, investigate optional ScanContext loop closure.

Do not skip directly to UI polish before proving Livox → rosbag → GLIM works from the terminal.

## 21. Important behavior while working

Inspect upstream repository APIs and current documentation before writing wrappers around them.

Do not fabricate topic names, GLIM command names, parameters, or export commands.

Where upstream behavior differs from this specification, adapt our wrapper and document the difference.

If a dependency does not support ARM64, investigate the actual cause and find an ARM-compatible build path rather than quietly substituting an unrelated SLAM system.

Do not remove or overwrite existing ROS workspaces on the Jetson.

Do not modify unrelated user projects.

Use non-destructive installation procedures.

After every major working milestone, commit changes with a descriptive Git commit.

## 22. First execution goal

Before doing any advanced work, get this minimal end-to-end test functioning:

Mid-360
→ official livox_ros_driver2
→ `/PointCloud2` + `/Imu`
→ rosbag2
→ GLIM offline processing
→ saved GLIM result

Then launch the browser UI and make it able to:

show Mid-360 health
create a named mapping session
start/stop rosbag recording
show recording duration/size
select a recorded session
run GLIM processing
show the processing log/result

Point-cloud browser visualization can come immediately after that.

At the end of the first implementation pass, provide:

- what was successfully implemented
- exact Jetson environment detected
- dependency versions and Git SHAs
- exact commands to start the UI
- exact commands for a manual terminal-only test
- known limitations
- remaining work
- any ARM64-specific patches that were required

Start by inspecting the machine and the existing directory. Then proceed with the implementation. Do not only give me a plan: create the repository and implement as much of this as possible.