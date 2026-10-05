# GLIM toolkit

The workstation GLIM build includes the five official `glim_ros2` executables below. These controls are blocked on the recording-only Jetson, even if an older installation left binaries behind. Transfer completed recordings to the workstation first:

| Executable | Project access | Purpose |
|---|---|---|
| glim_rosnode | Start GLIM Live / Start GLIM + desktop viewer | LiDAR-inertial mapping, optional native live visualization |
| glim_rosbag | Process/Reprocess | Direct bag processing with isolated ROS inputs |
| offline_viewer | Manual loops / merge / optimize | Graph editing, session merging, refinement and PLY export |
| map_editor | Object segmentation / cleanup | Point selection, segmentation, annotation and removal |
| validator_node | Start upstream validator | Upstream timestamp and sensor-data checks |

These are the actual official programs, not placeholder browser reimplementations. Native tools open on the **server's desktop**. A browser on another PC does not receive the native window; use a desktop session on the server or process the copied session on that PC. The processing workstation needs a working display/OpenGL environment for native tools and export. The recording Jetson does not run GLIM.

## Non-destructive workspaces

Select a completed session/run. The launcher copies its dump to `edits/edit_ID/map_01/`. Additional merge inputs are copied as map_02, map_03, etc. The browser displays those exact paths. Original dumps and raw bags are untouched by the launcher.

**Save explicitly to `edits/edit_ID/saved_map/` in the native application.** Closing a native window does not imply saving. After closing, Export saved edited map converts this directory to a new PLY. The source copies remain available. Do not manually select original acquisition directories as save destinations in native file dialogs.

Viewer copies use the official CPU global-mapping configuration and 1280×720 window size, supporting machines without CUDA. This changes only the working copy. Editing a GPU-origin dump on a CPU-only workstation still needs target acceptance testing. GPU builds are still available for large offline jobs.

## Graph editing and merging

The [official offline-viewer guide](https://koide3.github.io/glim/quickstart.html) describes manual loop constraints and Plane-BA. Select two submap spheres as loop endpoints, align their clouds, inspect the registration, then add the constraint. For a flat surface, the point context menu offers a plane constraint; choose a suitable support region before adding it. Optimization and factor visualization help inspect the result.

For multiple maps, follow the [official merging workflow](https://koide3.github.io/glim/merge.html). Start with the first copied dump and enable optimization when prompted. Load another copy through File → Open Additional Map. Use Merge sessions, select indoor/outdoor registration settings, establish coarse alignment, refine it, and add a factor only after visual inspection. Find overlapping submaps and Optimize refine the combined graph. Recover graph is available when the viewer reports missing graph connections; repair and save each source separately before merging.

The short stationary acquisition tests contain only one submap. They prove the pipeline, not a manual-loop or multi-zone alignment result. Proper loop acceptance testing requires a walked route with multiple overlapping submaps.

## Segmentation and cleanup

The [official map-editor guide](https://koide3.github.io/glim/edit.html) covers MinCut object segmentation, RegionGrowing plane segmentation, Gizmo selection, and radius/outlier tools. Start from a seed point, adjust the corresponding geometric parameters, inspect the selected set, then remove or annotate it. Save to the displayed derived output folder. These are geometric selection tools, not an automatic semantic object classifier. The editor keeps submap poses fixed; change poses in offline_viewer first.

## Terminal use

Keep the web backend stopped when using the managed terminal launchers:

```bash
scripts/open_glim_tool.sh offline_viewer SESSION_ID run_002
scripts/open_glim_tool.sh offline_viewer SESSION_A run_001 --add-map SESSION_B run_003
scripts/open_glim_tool.sh map_editor SESSION_ID run_002
```

For the direct upstream diagnostic on a workstation, use the acquisition ROS domain (41 by default; adjust to `config/livox/mid360.yaml`):

```bash
source scripts/env.sh
export ROS_DOMAIN_ID=41
ros2 run glim_ros validator_node --ros-args -r imu:=/livox/imu -r points:=/livox/lidar
```

The dashboard remaps those diagnostic names from project configuration automatically.

## Extensions

`glim_ext` is a separate optional ecosystem, not required by these core tools. Its pinned source was inspected. ScanContext declares a CC BY-NC-SA 4.0 dependency in upstream README, so it remains disabled/unvalidated for this production-oriented factory repository. The native manual-loop tools above do not require ScanContext. GLIM camera/DBoW input and alternative odometry backends remain inactive; the separate RGB recorder is supported. IMU prediction/validation, gravity, velocity suppression, GNSS and deskewing extensions need individual compatibility and use-case tests before enabling them. Their presence in an upstream repository is not evidence that all are appropriate for this sensor or Jetson.
