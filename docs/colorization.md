# Camera → LiDAR colorization

Offline RGB colorization of a finalized session. The master colored point cloud is
the color authority; GLIM and NKSR outputs only receive a copy of its color.

```bash
source scripts/env.sh
SESSION="$ROOT/data/sessions/<session>"
python3 tools/glim_colorize.py --session "$SESSION" --validation-frames 10
python3 tools/glim_colorize.py --session "$SESSION" --transfer-glim --glim-ply auto
python3 tools/glim_colorize.py --session "$SESSION" --transfer-nksr --nksr-mesh auto
```

Everything is written to a fresh `session/colorization/run_<id>/` directory.
Recorded bags, GLIM dumps, exports, prepared input and NKSR meshes are never
modified or overwritten.

## Pipeline

| Stage | Detail |
| --- | --- |
| Raw observations | Livox points with their own per-point timestamps |
| Trajectory | Existing GLIM optimized `traj_lidar.txt` interpolation (GLIM's own SLERP) |
| World points | Raw LiDAR observations transformed into world space |
| Image time | Recorded `Image.header.stamp`, never arrival time, frame index or configured FPS |
| Clock | `t_lidar = t_camera + time_offset_sec` from the validated session extrinsic |
| Association | `|point_t - t_lidar| <= max_time_delta` (default 0.15 s) |
| Projection | World → camera with the session's rectified intrinsics; rectified images are not undistorted again |
| Visibility | One depth buffer per image over **all** projection chunks; a second pass tests every chunk against it |
| Sampling | Bilinear RGB at the floating projected coordinates |
| Fusion | Per-channel median of up to `max_color_observations_per_voxel` observations per voxel |
| Output | Standard binary PLY (`x y z red green blue [intensity]`), NPZ metadata, statistics, validation overlays |

Different raw measurements inside one final voxel may have different camera
visibility. Each of them can contribute its own valid RGB observation; a voxel is
colored when at least one accepted observation exists.

## Border quantization

Floating projected coordinates are converted to integer pixels by exactly one
helper and one convention (`floor`, clipped into the image) used both to build the
depth buffer and to test a point against it. A sample at `u = width - 0.2` is
inside the frame but must never produce the index `width`. Bilinear RGB sampling
keeps using the floating coordinates.

## Bounded observation storage

Memory scales with `colored_voxels × max_color_observations_per_voxel`, not with
the total number of valid projections. Selection is deterministic: the first
accepted observations per voxel in recorded frame order, no sampling. Rejected
overflow is reported as `observations_dropped_due_to_per_voxel_limit`.

## Run lineage

Automatic resolution never picks "the newest file".

* The colorization trajectory determines the GLIM processing run
  (`processing/run_00N/glim_dump/traj_lidar.txt` → `run_00N`).
* `--glim-ply auto` accepts only exports named `<run_id>_<hex>.ply` for that run.
* `--nksr-mesh auto` accepts only reconstruction runs whose `job.json` references
  that same trajectory and whose mesh job completed.
* Zero candidates, several candidates, or a trajectory outside `processing/run_*`
  (for example a saved map-editor cleanup) all fail with a message that asks for
  an explicit path. Colors are never guessed across runs.

`metadata.json` records `source_session`, `source_bag`, `source_processing_run`,
`source_trajectory`, `source_glim_ply`, `source_reconstruction_run`,
`source_nksr_mesh`, `camera_calibration`, `image_topic` and `camera_info_topic`.

Before writing a colored target, a loose sanity guard rejects geometry that
cannot belong to the colorized cloud: disjoint bounding boxes, or fewer than 1 %
of sampled target vertices having a colored point inside the transfer radius.
`--skip-transfer-sanity` disables it; use it only for a deliberately different
target.

## Color transfer

`scipy.spatial.cKDTree` is required and is installed from the project lock file.
There is no quadratic fallback: if SciPy is missing, transfer fails with an
explicit message and recording, GLIM processing and NKSR are unaffected.

* GLIM: `glim_colored.ply` preserves every original attribute bit for bit and only
  adds `red`, `green`, `blue`. Only master points with valid color participate.
* NKSR: `nksr_colored.ply` preserves vertex positions and face indices exactly.
* Weighting is deterministic: `confidence / distance²` over the `k` neighbors
  inside the radius. Nothing beyond `--transfer-radius` is extrapolated.
* Defaults: `--transfer-radius 0.025`, `--transfer-k 5`. A large radius can mix
  color across two sides of a thin wall or panel, because the transfer is
  Euclidean and not normal-aware.

## Outputs

| File | Content |
| --- | --- |
| `output/colored_points.ply` | Portable RGB PLY; uncolored points use the neutral fallback (128 128 128) |
| `output/colored_points.npz` | `points`, `Cd` (NaN when uncolored), `color_confidence`, `color_count`, intensity, timestamps |
| `output/colorization_stats.json` | Counts, coverage and the effective parameters |
| `output/glim_colored.ply`, `output/nksr_colored.ply` | Optional transferred color |
| `output/*_transfer_stats.json` | Transfer coverage and sanity results |
| `validation/frame_*_overlay.jpg` | Depth-colored LiDAR projections spread over the whole scan |
| `metadata.json`, `calibration_snapshot.yaml` | Provenance, lineage and the calibration actually used |

`color_confidence` is observation-support confidence only: the number of retained
valid camera observations per voxel, capped at three. It is not a photometric
quality measure, and uncolored points keep a count and confidence of zero. The
neutral PLY fallback is display-only and must not be read as measured RGB.

## Validation overlays

`--validation-frames N` writes depth-colored (JET) LiDAR projections to the
recorded images, selected evenly from the first to the last eligible frame.
Each entry in `metadata.validation` records the file, frame index, camera
timestamp, interpolation timestamp, image topic and the offset used. Overlays
reveal extrinsic misalignment, a wrong transform direction, a wrong time offset,
incorrect focal length or principal point, and trajectory mismatch. They contain
no sampled camera RGB.

`python -m factory_mapping.colorization --session "$SESSION" --frames 8` remains
the standalone projection-validation entry point described in
[camera calibration](camera_calibration.md).

## Limitations

* Occlusion uses one depth buffer per image plus a tolerance
  (`occlusion_base_tolerance + occlusion_range_scale × depth`); it is a
  visibility filter, not a full geometric occlusion model.
* `--depth-edge-rejection` (off by default) additionally rejects samples whose
  pixel neighbourhood shows a depth discontinuity, which reduces color bleeding
  along pipes, silhouettes and wall edges. It is a heuristic.
* Voxel size, time window, occlusion tolerances and observation limits are
  user-configurable algorithm defaults, not calibration constants.
* No Houdini or BGEO export is part of this pipeline; the standard RGB PLY is the
  portable output.
