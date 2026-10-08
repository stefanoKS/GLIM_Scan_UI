# VDBFusion surface reconstruction

[VDBFusion](https://github.com/PRBonn/vdbfusion) is the fast TSDF engine: it streams
raw LiDAR observations into a sparse OpenVDB TSDF and extracts one fused world-space
triangle mesh. It is an **additional** engine. NKSR is untouched and remains fully
selectable, and a run prepared for one engine can never be reconstructed by the
other.

```bash
source scripts/env.sh
./scripts/setup_vdbfusion.sh            # isolated environment, explicit install
./scripts/check_vdbfusion.sh            # real native integration + extraction test
```

## Reconstruction algorithm selection

**Reconstruction algorithm** is the first control under Surface Reconstruction.

| Option | Notes |
| --- | --- |
| **VDBFusion — Fast TSDF** | Default for new jobs when it is installed |
| **NKSR — Neural Reconstruction** | All existing mode/device/chunk controls are unchanged |

* Jobs, API requests and history entries without an algorithm field stay **NKSR**.
* An unavailable VDBFusion shows its installation status and blocks preparation. The
  dashboard never substitutes NKSR for a failed or unavailable VDBFusion request.
* Switching the selector does not delete, hide or overwrite older jobs. Every job
  records its engine in `job.json`, and the prepared-input list labels each run with
  the engine it was prepared for.

## Isolated installation

```bash
./scripts/setup_vdbfusion.sh
```

* Creates `~/.cache/factory-mapping/vdbfusion-env` (override with
  `VDBFUSION_PREFIX`, `VDBFUSION_CACHE`, `VDBFUSION_BASE_PYTHON`,
  `VDBFUSION_VERSION`) and records the interpreter in `.state/vdbfusion_python.txt`.
* Base interpreter is `/usr/bin/python3.10`, matching the workstation's ROS Python.
* Upstream `vdbfusion==0.1.6` wheels are published for Python 3.6–3.10 only, so the
  install refuses any other interpreter version instead of building from source.
* NumPy is pinned to the project's locked `1.26.4`. The upstream wheel is built
  against the NumPy 1.x ABI and segfaults against NumPy 2.x.
* The upstream wheel statically links OpenVDB and TBB, so it cannot conflict with the
  ROS Humble system TBB and needs no `apt` changes.
* Nothing is written into the NKSR environment, the application virtualenv, CUDA,
  Torch or the ROS installation. Supported on Linux x86_64 only, and the install
  refuses a `record_only` deployment, so a Jetson recording installation never gains
  these dependencies.

The readiness check is a real primitive test, not an import check: it constructs a
small TSDF, integrates synthetic observations, extracts a triangle mesh and validates
finite vertices with in-range triangle indices. Results are written to
`.state/vdbfusion_health.json`.

```bash
VDBFUSION_PYTHON=/path/to/python ./scripts/check_vdbfusion.sh
```

## Settings

The selected algorithm controls which settings are visible.

| Setting | Default | Notes |
| --- | --- | --- |
| TSDF voxel size | `0.02` m | Validated finite, `0.001`–`1.0` m |
| TSDF truncation distance | `0.06` m | Independently adjustable, must be at least the voxel size |
| Space carving | off | Left off by default, especially for edited maps |
| Mesh output | single mesh | Or separate meshes, or both |
| Region of interest | off | Optional world-space box in metres |

| Preset | Voxel | Truncation |
| --- | --- | --- |
| FAST | 20 mm | 60 mm |
| DETAILED | 10 mm | 30 mm |
| EXPERIMENTAL | 5 mm | 15 mm |

The EXPERIMENTAL preset warns about memory and runtime. Measured on a real
10 m × 10 m × 6 m region of the factory session (see
[measured results](#measured-workstation-results)):

| Preset | Triangles | Peak RSS | Output | Total elapsed |
| --- | --- | --- | --- | --- |
| 20 mm | 3,989,195 | 628 MiB | 80.2 MiB | 48.6 s |
| 10 mm | 11,952,395 | 1.99 GiB | 252.5 MiB | 63.6 s |
| 5 mm | 17,460,382 | 4.41 GiB | 408.9 MiB | 80.5 s |

**TSDF voxel size is not the NKSR preparation voxel.** Preparation sampling is an
NKSR control in centimetres; VDBFusion reads the raw bag directly and never applies
NKSR's internal coordinate scaling. The two settings are shown in separate blocks and
are never combined.

## Data flow

```text
ROS raw bag
  -> bounded observation batches
  -> GLIM trajectory interpolation (the proven traj_lidar.txt path)
  -> optional verified-edit classification
  -> motion-aware sensor-origin grouping
  -> native VDBFusion integration (one fused sparse TSDF)
  -> one mesh extraction
  -> optional removed-region triangle masking
  -> spatial mesh partitioning, when separate output files are requested
```

## Motion-aware sensor origins

VDBFusion takes **one** sensor origin per `integrate()` call, so a moving Mid-360
recording is never integrated with a constant origin.

* Every observation is transformed with its own per-point timestamp through the
  existing GLIM SLERP interpolation, exactly as the NKSR preparation does.
* Observations are grouped into maximal contiguous runs whose origins stay within a
  budget distance of the group representative.
* The representative is the **actual interpolated trajectory origin** of the group's
  first observation. No origin is ever fabricated, and `[0, 0, 0]` is never assumed.
* The default budget is one TSDF voxel, and the realized maximum approximation error
  is reported in `output/vdbfusion_metadata.json`
  (`integration.max_origin_error_m` against `integration.origin_error_budget_m`).
* Grouping never discards an observation: the groups partition the batch exactly.
* Observations outside the trajectory's time range are counted
  (`integration.outside_trajectory`) and dropped explicitly. Timestamps are never
  clamped to the first or last pose.

## Memory behaviour

* The recording is streamed: bounded batches (default 2,000,000 observations, maximum
  20,000,000) are read, filtered, grouped and handed to the native integrator one at
  a time. No whole-cloud `concatenate`, no global `unique`, no intermediate PLY and no
  100M-point NumPy array exists anywhere in the pipeline.
* VDBFusion preparation writes **no** prepared point cloud. It validates the sources
  and reports a resource preflight to `input/vdbfusion_prepare.json`.
* Resource preflight reports free RAM, free disk, requested voxel size, the sampled
  scan extent and an order-of-magnitude TSDF footprint estimate. The estimate is
  capped by the observation count, so it is realistic rather than a whole-box volume.
* **Streaming does not bound the TSDF itself.** A large fine-resolution sparse volume
  still grows with the touched surface. A soft memory budget (default 24 GiB, or
  `--memory-budget-gib`) fails the run cleanly with `MEMORY_BUDGET_EXCEEDED`; the
  requested resolution is never silently reduced.

## Run, cancel and job isolation

* The worker runs as a managed independent process (`vdbfusion`), so a native memory
  fault cannot take down the web backend.
* Progress is written to `reconstruction/run_ID/vdbfusion_progress.json`, logs to
  `reconstruction/run_ID/job.log`.
* Cancellation is cooperative: the signal sets a flag that is checked at every
  bounded batch boundary, so the native integrator is never abandoned mid-call. A
  second signal unwinds immediately.
* Every mesh attempt archives a previous `output/`, `mesh_job.json` and progress file
  into `attempts/previous_<id>/`. A validated mesh is never overwritten by a retry.
* VDBFusion writes `output/vdbfusion_metadata.json`. NKSR metadata is never written
  for a VDBFusion job, and the two engines use separate managed processes, so neither
  can overwrite the other.

## Output files

| File | Content |
| --- | --- |
| `input/vdbfusion_prepare.json` | Preparation: validated sources, sampled scan summary, preflight, TSDF settings |
| `output/mesh.ply` | Binary little-endian triangle PLY, XYZ in **world metres**, indexed faces, no local origin shift |
| `output/mesh_chunks/chunk_NNNN.ply`, `chunks.json` | Separate meshes: spatial cells of the same single fused mesh |
| `output/vdbfusion_metadata.json` | Engine identity, settings, timings, integration and edit-filter statistics, mesh validation |
| `vdbfusion_progress.json`, `job.log` | Live stage, messages and full log |

Separate meshes reuse the existing validated spatial partitioner on the **one** fused
mesh. Independent TSDF tiles are never reconstructed just to create separate files, so
no extra tile-boundary seam is introduced. If full-mesh extraction exceeds memory the
run reports that limitation instead of quietly producing disconnected tiles.

## Saved edited geometry

The existing workflow is unchanged, with VDBFusion added as the engine:

```text
GLIM processing -> Clean Map -> Save As edits/edit_ID/saved_map/ -> close editor
  -> Export saved edited map -> verify export ready
  -> select VDBFusion -> Use saved edited geometry -> select the matching saved edit
  -> Reconstruct mesh
```

* Verification reuses the existing lineage: `edit_ID`, saved-map fingerprint,
  trajectory fingerprint, verified edited PLY export, export state and session
  identity. A trajectory from one edit is never combined with geometry from another,
  and `edits/edit_ID/map_01/traj_lidar.txt` is never substituted for the saved one.
* The saved cleanup itself is never mutated or regenerated.
* NKSR still uses its proximity tolerance. **VDBFusion does not use that tolerance.**
  It classifies observations against the verified kept **and removed** submap geometry
  and derives its own association radius from the measured reference sampling spacing
  (`sampling_resolution_m × association_spacing_multiplier`, default `4`). The exact
  radius is reported in the metadata and in the dashboard.
* Filtering mode is reported honestly. Exact per-observation edit transfer is **not**
  available, because GLIM stores submaps as voxel samples and cannot trace a saved
  point back to the raw observations that produced it. The strategy is therefore
  labelled `validated_approximate` everywhere and the limitation is written into the
  metadata, not only into documentation.
* Observation filtering alone does not guarantee that an implicit surface respects
  every deletion, so triangles whose centroids fall inside a removed region are masked
  after extraction. On the real factory ROI this removed 30,291 triangles that the
  observation filter could not have removed. Masking is on by default and reported as
  `extraction.masked_triangles`.

Limits, stated plainly:

* Removed/supported observations without kept geometry within the association radius
  (`unsupported`) are **excluded** by default. On the real session this is about 13 %
  of observations inside a fully retained ROI, because GLIM's own range gating and
  0.1 m submap sampling discard raw returns. Including them
  (`unsupported_observations = include`) is available but is an explicitly labelled
  experimental override that can recreate removed geometry.
* A documented boundary uncertainty band equal to the reported association radius
  surrounds every edit boundary. Coverage and exclusion claims are always measured
  outside that band.

### Measured edit behaviour

On the real saved cleanup `edit_ef16d03e7f86` (57 submaps, 2,806,885 pre-edit
submap samples, 2,442,751 retained, measured sampling resolution 0.0385 m, derived
association radius 0.1542 m):

| Measurement | Value |
| --- | --- |
| Raw observations whose nearest pre-edit sample is removed | 12.77 % |
| Of those, retained by the filter | **0.00 %** (complete exclusion) |
| Raw observations whose nearest pre-edit sample is kept | 87.23 % |
| Of those, retained | **92.18 %** |
| Legacy fixed 5 cm tolerance retention on the same observations | 57.33 % |
| 10 m ROI, edited mode | 1,717,122 before filter → 1,488,392 retained (86.68 %) |
| Triangles masked inside removed regions | 30,291 |

The 57.33 % figure reproduces the earlier factory run's `filter_retention`, which is
why the previous output looked patchy: a 5 cm tolerance is below the 0.1 m sampling
resolution of the reference geometry, so retained-surface observations were dropped
systematically. It does not indicate that the discarded observations were wrong.

Synthetic acceptance results
(`tests/test_vdbfusion_edit.py`, spatial occupancy at the TSDF resolution):

| Target | Result |
| --- | --- |
| ≥ 95 % intended retained-surface coverage | 0.99–1.00 (wall, thin retained strip, second object) |
| ≥ 98 % removed-region exclusion, outside the band | 1.00 |
| No large artificial holes in retained planar surfaces | ≥ 0.95 with no band allowance, 1.00 with the band |
| No bridging across a deliberately deleted gap | 0 triangles in the deleted connector |

## Validation commands

```bash
source scripts/env.sh

# Real native runtime check (integration + extraction + mesh validation)
./scripts/check_vdbfusion.sh

# Staged benchmark: synthetic, then a real subset, then a real ROI, then full
./scripts/benchmark_vdbfusion.sh --stage A
./scripts/benchmark_vdbfusion.sh --stage B --bag "$SESSION/raw_bag" \
    --trajectory "$SESSION/processing/run_001/glim_dump/traj_lidar.txt" --subset-seconds 20
./scripts/benchmark_vdbfusion.sh --stage C --bag "$SESSION/raw_bag" \
    --trajectory "$SESSION/processing/run_001/glim_dump/traj_lidar.txt" \
    --roi-min -41.34 13.76 0.0 --roi-max -31.34 23.76 6.0 --presets
./scripts/benchmark_vdbfusion.sh --stage D --bag "$SESSION/raw_bag" \
    --trajectory "$SESSION/processing/run_001/glim_dump/traj_lidar.txt" --allow-full

# Engine, edit and orchestration tests (native tests skip without an installation)
.venv/bin/python -m pytest tests/test_vdbfusion.py tests/test_vdbfusion_edit.py \
    tests/test_vdbfusion_jobs.py -q
```

Stage D processes the whole recording and refuses to run without `--allow-full`.
Benchmark output includes per-phase timings (bag read, decode, trajectory
interpolation, edit filtering, origin grouping, TSDF integration, mesh extraction,
mesh validation), peak RSS, mesh counts and output size. Every figure is measured by
that run.

## Measured workstation results

Session `20261007_130338_Scan_2026-10-07_13_03_38`, Livox Mid-360, 5,316 LiDAR
frames, 105,802,081 raw world-space observations, `run_001` trajectory
(5,304 poses, 566.8 s, 603.6 m).

Stage B — first 20 s of the trajectory, 20 mm TSDF:

| Measurement | Value |
| --- | --- |
| Integrated observations | 3,835,427 (102,226,525 outside the trajectory range) |
| Triangles / vertices | 15,155,404 / 12,449,424 |
| TSDF integration | 2.89 s |
| Peak RSS | 3,223 MiB |
| Total elapsed | 45.40 s |
| Output | 330.4 MiB |

Stage C — 10 m × 10 m × 6 m ROI, 20 mm TSDF:

| Measurement | Value |
| --- | --- |
| Integrated observations | 1,717,122 (104,084,959 outside the ROI) |
| Triangles / vertices | 3,989,195 / 2,685,109 |
| Bag read / decode | 0.83 s / 3.99 s |
| Trajectory interpolation | 25.62 s |
| Origin grouping | 0.30 s |
| TSDF integration | 0.88 s (8,219 origin groups) |
| Mesh extraction | 1.77 s |
| Mesh validation (write + re-read) | 4.78 s |
| Peak RSS | 628 MiB |
| Total elapsed | 48.6 s |
| Max origin approximation error | 20.00 mm of the 20.00 mm budget |

Read these figures with their limits in mind:

* **Trajectory interpolation dominates**, not the TSDF. Per-point GLIM SLERP
  interpolation of 105.8 M observations costs about 4 M points/s and is paid for the
  whole recording even when an ROI keeps only 1.7 M observations. Profile the input
  pipeline before optimising the native integration.
* Integration cost scales with voxel resolution (0.88 s at 20 mm, 1.32 s at 10 mm,
  1.97 s at 5 mm on the same ROI).
* Mesh validation includes writing and re-reading the PLY, so it scales with the
  output size.
* No verified speedup against NKSR is claimed. The completed NKSR runs on this session
  used different prepared inputs and spatial extents (for example `run_e3b10a239f68`
  reconstructed 58,823,438 prepared points in Low RAM mode over 6,509 s), so no
  apples-to-apples measurement exists here. Compare engines only on the same bag,
  trajectory and region.
* The 5 mm result is a triangle count and a memory figure, not an accuracy claim.
  Geometric accuracy at 5 mm has not been validated against an independent reference.

## Limitations

* The TSDF surface is an implicit reconstruction from observations only. There is no
  normal estimation, no learned prior and no hole filling, so occlusion shadows and
  sparse regions remain open. NKSR remains the choice for smooth, watertight surfaces.
* Separate mesh output is a spatial split of one fused mesh. It reduces per-file size,
  not peak memory during extraction.
* Space carving is off by default and can delete thin structures. Keep it off for
  edited maps.
* The edited-mode association is approximate and never claims exact transfer. Read the
  reported band before trusting sub-voxel edit boundaries.
* VDBFusion is not installed on the Jetson, and the install refuses a recording-only
  deployment.
* `extract_vdb_grids`, `prune` and `update_tsdf` from the upstream Python bindings are
  not exposed by this integration; saving the TSDF volume is not supported.
