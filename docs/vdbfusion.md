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
* `batch_points` is a **hard** bound on every batch, not an average. A single LiDAR
  frame larger than the bound is split across several batches in recorded order:
  observations are never dropped, reordered or duplicated at a slice boundary, and the
  trailing partial batch is still emitted. `bag_scan`/integration metadata reports
  `batches`, `split_frames` and `max_batch_points` so the bound can be checked after a
  run.
* VDBFusion preparation writes **no** prepared point cloud. It validates the sources
  and reports a resource preflight to `input/vdbfusion_prepare.json`.
* Resource preflight reports free RAM, free disk, requested voxel size, the sampled
  scan extent and a **component-wise** footprint estimate: TSDF (primary estimate plus
  the dense-band and truncation-influence upper bounds), edited-geometry reference
  index, bounded input batches, mesh extraction, masking and PLY output plus its
  transient copy. The primary TSDF estimate is the observation count times the
  **measured** voxels-per-observation from the sampled bag frames, or the observation
  count itself when no sample is available. Every component is reported with the
  heuristic behind it; the total carries an explicit `1.3x` safety margin.
* The truncation-influence bound is reported separately and is never hidden inside a
  single number: one observation can influence `(2·trunc/voxel + 1)³` voxels
  (343 at 20 mm voxels with 60 mm truncation), so a surface estimate of one voxel per
  observation is only sane because a surface band is one voxel thick. The union bound
  warns when the worst case would exceed the usable memory.
* One byte-valued limit is derived from the request by
  `resolve_memory_budget_bytes`: the UI/API control `memory_budget_gib` (or an explicit
  `memory_budget_bytes`, or the default 24 GiB) is normalised once and used by the
  preflight, the runtime soft check and the memory supervisor, so all three agree. A
  malformed budget is an error rather than a silent fall back to the default, which could
  only widen the limit.
* The preflight also plans **disk**: the published set (merged PLY, the chunk set with its
  documented cell-boundary duplication factor, or both), the previous set that is kept as a
  backup until the completion marker is written, and a reserve of 5 % of the predicted set or
  2 GiB, whichever is larger. The measured retry history under `attempts/` is reported
  separately because it is kept, never pruned. A shortfall is refused with advice (free
  space, export one mode instead of both, or use an ROI); the requested resolution is never
  changed automatically.
* That disk figure is deliberately conservative and its uncertainty is documented: it uses 32
  bytes per triangle where real 20 mm and 10 mm ROI output measured 21.0 and 21.7, and a 1.1x
  chunk factor where the real 20 mm edited ROI export measured 1.002 (80,346,854 bytes of
  chunk files against 80,217,785 bytes for the same mesh merged, 8 cells, 3,824,200
  triangles). On that run the prediction was 1.68x the 153.1 MiB actually written. Treat it as
  a guard against a clearly too-small disk, not as a prediction.
* A run is **refused before it starts** with `RESOURCE_PREFLIGHT_FAILED` plus
  mitigation advice when the estimated peak cannot fit inside
  `min(configured budget, available RAM − reserve)`. The comparison is arithmetic on byte
  values, so zero usable headroom fails closed instead of passing a positive estimate. The
  requested resolution is still never silently reduced: the message tells the user which
  setting to change and asks for a re-prepare; when free RAM does not even cover the
  reserve it says so explicitly.
* **Streaming does not bound the TSDF itself.** A large fine-resolution sparse volume
  still grows with the touched surface. A soft memory budget (default 24 GiB, or
  `--memory-budget-gib`) also fails a run that grows past it *while* integrating, with
  `MEMORY_BUDGET_EXCEEDED` and the requested resolution untouched.
* While the worker runs, the backend samples the **worker process tree** RSS every
  2 seconds and records the sampled peak, the stage at the peak, samples and elapsed
  time per stage and any pressure event in `mesh_job.json → memory`. The limit it samples
  against is the same normalised byte value the worker enforces. Sampling happens
  in the backend so it keeps working while the worker is inside a native call. At 95 %
  of the budget the supervisor requests cooperative cancellation once and records the
  event; it never kills the process itself and never touches previous outputs. Every
  figure is labelled as a *sampled* peak, i.e. a lower bound on the true peak.
* What the sampled figures do **not** establish: the sampler observes the process tree every
  2 seconds from outside, so a spike between two samples, or memory held inside the native
  library between allocations, is invisible; the peak is a lower bound, not an instrumented
  measurement. The per-stage elapsed times are derived from sample counts, not from stage
  timestamps, so they are approximate to the sampling interval. The runtime budget check
  inside the worker remains the authority on exceeding the limit, and the preflight remains
  the authority on whether the run should start at all.

## Run, cancel and job isolation

* The worker runs as a managed independent process (`vdbfusion`), so a native memory
  fault cannot take down the web backend.
* Progress is written to `reconstruction/run_ID/vdbfusion_progress.json`, logs to
  `reconstruction/run_ID/job.log`.
* Cancellation is cooperative: the signal sets a flag that is checked at every
  bounded boundary **including the mesh scan, the triangle probes, immediately after the
  native extractor returns, and before the outputs are published**, so the native
  integrator is never abandoned mid-call but a long extraction, masking or writing pass
  still stops at the next checkpoint.
* `SIGTERM` is handled like `SIGINT`, because it is the escalation step of the managed stop
  ladder: the worker stops at the next checkpoint and reports `CANCELLED` instead of dying
  with no state, and the parent still escalates to `SIGKILL` if a checkpoint is too far
  away. A Python signal handler only runs when the interpreter regains control, so a signal
  that arrives during a native call is honoured *after* that call returns - that deferral is
  measured by `tests/test_vdbfusion_cancellation.py`, not assumed away: on a 7.1 million
  triangle mesh the worker was signalled inside `EXTRACTING_MESH` and reported `CANCELLED`
  2.4 s later (13.4 s before the post-extraction checkpoint was added, because it first
  wrote a 135 MiB staged PLY).
* A second signal unwinds immediately. A user-requested stop is recorded as `CANCELLED`,
  never as an engine failure.
* Cancellation escalates with a bounded grace period: `SIGINT` (cooperative), then
  `SIGTERM`, then `SIGKILL` on the worker's process group, so no orphan process tree is
  left behind.
* Every mesh attempt archives a previous `output/`, `mesh_job.json`, progress file and
  staging directory into `attempts/previous_<id>/`. A validated mesh is never
  overwritten by a retry, and a stale progress file can never be mistaken for the
  current run's.
* Meshes are written into `output/staging/`, validated there, and only then published. A
  partial or invalid mesh is therefore never present at the published path, and a failed
  attempt leaves the previous output untouched. The whole staged set is validated before
  anything moves: the merged PLY is re-read with the independent validator, and the chunk
  set through the shared chunk-manifest validator.
* Publishing never unlinks a previous valid artifact before its replacement exists: a file
  is swapped with a single atomic rename, and a directory is moved aside before the new one
  is moved in and restored if the swap fails. If a restore cannot be completed the error
  names the exact backup path instead of losing the artifact silently. A cancelled or
  failed run removes its staging area, so no partial mesh is left on disk either.
* Publication order is: validate the staged set, move every existing artifact aside to a
  backup, move each staged artifact into place, **write the completion manifest, and only then
  delete the backups and the staging area**. The manifest is the commit point, so a failure
  while writing it - including a full disk - rolls the whole publication back: the previous
  artifacts are restored, anything this attempt placed is removed, and the staged set is kept
  for a retry. Multi-artifact atomicity is *not* claimed: between the renames a reader can see
  a mixed set, and a hard kill in that window leaves backups on disk next to a stale manifest.
* A **completion manifest** (`output/publish_manifest.json`) records the attempt id, the mode,
  the published artifacts with their sizes and (up to 256 MiB) a SHA-256, and the transaction
  limitation above. `validate_completed` re-reads it and refuses a set whose manifest is
  missing, records different artifacts than the mode requires, records an artifact that is
  missing or has a different size, belongs to a different attempt, or sits next to a leftover
  `*.previous-*` backup from an interrupted publish. A partially published set therefore can
  never be reported as `COMPLETED`.
* VDBFusion writes `output/vdbfusion_metadata.json`. NKSR metadata is never written
  for a VDBFusion job, and the two engines use separate managed processes, so neither
  can overwrite the other.

## Output files

| File | Content |
| --- | --- |
| `input/vdbfusion_prepare.json` | Preparation: validated sources, sampled scan summary, preflight, TSDF settings |
| `output/mesh.ply` | Binary little-endian triangle PLY, XYZ in **world metres**, indexed faces, no local origin shift |
| `output/mesh_chunks/chunk_NNNN.ply`, `chunks.json` | Separate meshes: spatial cells of the same single fused mesh |
| `output/staging/` | Staging area: the mesh is written and validated here, then published by rename (removed after a successful publish and after a cancellation) |
| `output/publish_manifest.json` | Completion manifest: mode, published artifacts with sizes and SHA-256, and the documented cross-artifact transaction limitation |
| `output/vdbfusion_metadata.json` | Engine identity, settings, timings, integration and edit-filter statistics, mesh validation, geometry audit |
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
* The saved geometry is validated as an **exact multiset subset** of the pre-edit
  geometry, submap by submap. Both point blocks must be a whole number of float32 rows
  and finite, both `data.txt` files must declare the same submap `id` as the directory
  and a rigid `T_world_origin`, and the two frames must match exactly. A saved point the
  pre-edit submap never contained, or a saved multiplicity above the original
  multiplicity, raises `INVALID_SAVED_SUBMAP` naming the edit id, the submap id, the
  mismatch type, the number of invalid saved points and the corrective action. Nothing
  is reinterpreted as "removed" and nothing is guessed. (Measured on the real cleanup
  `edit_ef16d03e7f86`: all 57 submap pairs are byte-identical in `data.txt`, every saved
  submap is a multiplicity-valid subset, no submap contains a duplicate row, and saved
  points keep the original order.)
* The **saved** submap frame is read and compared, not ignored: a cleanup saved in a
  different coordinate frame is rejected instead of producing silently wrong world
  geometry.
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
  every deletion, and a centroid test alone misses a large triangle that crosses a
  deleted strip while its centroid sits in kept geometry. Masking is therefore staged
  and conservative:

  1. triangles whose **centroid** is dominated by removed geometry are dropped (this
     keeps the established edit-filter semantics, unsupported centroids included);
  2. the survivors are probed at all **three vertices and all three edge midpoints**,
     and a triangle touched by removed geometry is dropped;
  3. survivors whose edges are longer than the measured reference sampling resolution
     get **interior samples** along those edges, at most 8 per edge; hits on that cap
     are reported in `extraction.long_edge_cap_hits`.

  One hit removes the whole triangle, because gluing deleted geometry back into the
  mesh is exactly what this filter exists to prevent. Masking is on by default and
  reports `triangles_before`, `triangles_removed_by_centroid`, `triangles_removed_by_probe`,
  `triangles_dropped_unsupported`, `masked_triangles` and the probe counts.
* Classification reports a fourth class. Coordinates whose nearest kept and nearest
  removed samples are both inside the association radius and within the documented
  boundary band are labelled **AMBIGUOUS** (`classify()['ambiguous']`,
  `class_of()`). They are kept unless removed geometry dominates, and the count is
  reported as `ambiguous_triangles`, so the boundary band is visible instead of being
  presented as a confident classification. The band defaults to the measured reference
  sampling resolution and never widens the association radius.

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

## Prepared settings, staleness and source immutability

Preparation pins a canonical identity in `input/vdbfusion_prepare.json → identity`:

* **source**: bag path, file count, byte total and `metadata.yaml` size/mtime, topic,
  trajectory path/size/SHA-256, and the edited workspace plus edit id when filtering.
  The raw bag is read-only after capture and can be many gigabytes, so it is pinned by
  its byte total and metadata timestamp rather than a full hash; the small trajectory
  text file is hashed in full.
* **semantic settings**: preset, voxel size, truncation, space carving, origin error
  budget, ROI, unsupported policy, triangle masking, boundary margin, association
  spacing multiplier and the edited-geometry source. Their canonical form has a stable
  SHA-256 `semantic_fingerprint`.

Reconstruction re-resolves the requested settings, compares them with the prepared
identity, re-verifies the source identity and re-runs the resource preflight with the
settings the run will actually execute:

* a semantic difference is refused with `PREPARED_SETTINGS_STALE`, listing the
  differing keys with their prepared and requested values, and asks for a re-prepare.
  The requested values are never substituted silently, and the prepared ones never win
  silently either - the run is refused.
* a changed source is refused with `PREPARED_SOURCE_CHANGED` naming the changed fields.
* a resource change since preparation (or a finer resolution requested at mesh time) is
  refused by the preflight before any native work starts.
* **execution-only** settings - `batch_points`, memory budget, mesh output mode and
  chunk size - may change at mesh time without re-preparing, because they change how the
  run executes, not what it produces. The prepared semantic settings are what the worker
  runs with, and `mesh_job.json` records both the effective settings and the requested
  ones.

A run whose metadata is unreadable still reads as its historical engine (NKSR), so old
runs keep working; a VDBFusion run is never executed by the NKSR path and a missing
VDBFusion installation is always a clear error, never a silent fallback.

## Mesh geometry audit

`validate_and_report` refuses hard corruption (non-finite vertices, non-integer or
out-of-range indices, empty meshes) and additionally reports an independent
`geometry_audit` block: degenerate/zero-area triangles and their ratio, total surface
area, median and longest edge, extreme-edge count, orientation conflicts measured from
directed edge usage, connected-component count and the largest component's share, and
whether the mesh stays inside the observed extent (plus two truncation bands).

The audit **reports and never deletes geometry**: disconnected components and long
triangles are legitimate in factory scans, so the numbers are written to the metadata
and left for a human to judge. Chunk manifests are validated separately by the existing
chunk validator.

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

Stage C re-measured after the production-hardening audit (same session, same ROI, same
command shape, `scripts/benchmark_vdbfusion.sh`, one run at a time):

| Measurement | 20 mm plain | 20 mm edited | 10 mm plain | 10 mm edited |
| --- | --- | --- | --- | --- |
| Integrated observations | 1,717,122 | 1,488,392 | 1,717,122 | 1,488,392 |
| Batches / origin groups | 54 / 8,232 | 54 / 8,128 | 54 / 13,250 | 54 / 12,928 |
| Bag read + decode | 4.7 s | 4.6 s | 4.6 s | 4.6 s |
| Trajectory interpolation | 25.5 s | 25.0 s | 25.5 s | 25.1 s |
| Edit filter | — | 2.51 s | — | 2.55 s |
| TSDF integration | 0.92 s | 0.79 s | 2.27 s | 2.01 s |
| Mesh extraction | 1.73 s | 1.71 s | 10.95 s | 10.71 s |
| Triangle masking | — | 16.8 s (30,300 masked) | — | 53.9 s (66,768 masked) |
| Mesh validation (audit + write + re-read) | 7.1 s | 7.0 s | 26.4 s | 26.0 s |
| Triangles / vertices | 3,989,261 / 2,685,098 | 3,824,200 / 2,541,917 | 17,440,326 / 12,608,994 | 17,064,642 / 12,260,343 |
| Output | 80.2 MiB | 76.5 MiB | 360.5 MiB | 351.9 MiB |
| Process peak RSS | 1,066 MiB | 1,263 MiB | 2,706 MiB | 2,891 MiB |
| Total elapsed | 50.1 s | 71.3 s | 80.4 s | 141.0 s |
| Estimated peak / usable (preflight) | 8.02 / 24.0 GiB | 8.44 / 24.0 GiB | 6.34 / 24.0 GiB | 6.77 / 24.0 GiB |
| Max origin error vs budget | 20.00 mm / 20.00 mm | 20.00 mm / 20.00 mm | 10.00 mm / 10.00 mm | 10.00 mm / 10.00 mm |

The edit filter retained 86.68 % of the ROI observations in both resolutions
(1,717,122 → 1,488,392), with 228,730 observations dropped as unsupported and **0**
dropped as removed-dominated, which matches the earlier measurement.

End-to-end publication check on the real 20 mm edited ROI, output mode `both`
(`tools/vdbfusion_worker.py` directly, one run):

| Measurement | Value |
| --- | --- |
| Observations / triangles | 1,488,392 / 3,824,200 (30,300 masked) |
| Merged PLY | 80,217,785 bytes (20.98 bytes per triangle) |
| Chunk set | 80,346,854 bytes in 8 cells (21.01 bytes per triangle) |
| Manifest | records both artifacts, sizes match what `validate_completed` re-measures |
| Attempt id | manifest and metadata agree, so only this run's set can validate |
| Disk prediction vs actual | 256.7 MiB predicted, 153.1 MiB written (1.68x conservative) |

Mesh geometry audit on those four saved meshes (10 mm/20 mm, plain/edited):

| Measurement | 20 mm plain | 20 mm edited | 10 mm plain | 10 mm edited |
| --- | --- | --- | --- | --- |
| Degenerate triangles | 20 (5.2e-6) | 20 (5.2e-6) | 233 (1.34e-5) | 237 (1.39e-5) |
| Median edge | 20.16 mm | 20.16 mm | 10.05 mm | 10.05 mm |
| Longest edge | 34.63 mm | 34.63 mm | 17.32 mm | 17.32 mm |
| Extreme edges (> 0.48 m) | 0 | 0 | 0 | 0 |
| Orientation conflicts (sampled) | 0 | 0 | 0 | 0 |
| Surface area | 501.9 m² | 501.1 m² | 573.4 m² | 562.1 m² |
| Outside the observed ROI | 0.00 m | 0.00 m | 0.00 m | 0.00 m |

Read those audit figures with their limits in mind:

* All four meshes exceed the exact-topology face limit, so the topology and edge statistics
  are measured on a bounded, deterministic sample and labelled `topology_scope: sampled` /
  `edge_statistics_scope: sampled` in the metadata. The orientation-conflict count is
  therefore a **lower bound**, and no connected-component count is claimed for them:
  sampling arbitrary faces cuts edge adjacency, so a component count on the sample would be
  a fragmentation artefact rather than a property of the mesh (for reference, on a
  contiguous 1,000,000-face slice of the same mesh the exact scope reports 14,759 components
  with a largest-component share of 38.5 %).
* The audit itself costs 2.2–4.6 s on these meshes. The validation row is larger because it
  also writes and independently re-reads the PLY.
* The earlier stage C figures in the table above were measured before the audit existed and
  before `batch_points` became a hard bound; the extra 66 triangles at 20 mm and the higher
  peak RSS are the audit's working set, not a change in the TSDF or in the extracted
  surface. Integration peak RSS, observation counts, edit-filter retention and the origin
  error bound are unchanged.

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
