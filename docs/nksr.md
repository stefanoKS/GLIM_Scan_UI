# Optional pretrained NKSR surface reconstruction

Run this workflow on the **processing workstation** after transferring a completed Jetson project. Recording-only deployment blocks preparation, NKSR checks and mesh jobs in the backend even if an NKSR interpreter exists. The Jetson does not perform surfacing. See [installation roles](installation_jetson.md) and the [current validation summary](validation.md); test counts later in this document are historical NKSR-specific evidence.

**Prepare Reconstruction** creates paired world-space point measurements.
**Reconstruct Mesh** runs NVIDIA's pretrained kitchen-sink (`ks`) NKSR network,
builds its implicit surface, and extracts a triangle mesh. No training is required.
The regular **Export optimized PLY** control still uses GLIM's original exporter;
it never invokes NKSR and does not require the NKSR environment.

The raw bag, GLIM map/edits, trajectory, optimized exports, and prepared NPZ are
read-only inputs to mesh inference. Results live in the existing singular
`session/reconstruction/run_<id>/` tree:

```text
job.json                       # preparation state (PREPARED is not mesh completion)
job.log                        # preparation + worker stdout/stderr
input/nksr_input.npz
validation/reconstructed_from_bag.ply
validation/comparison.json
mesh_job.json                   # independent mesh state and settings
nksr_progress.json              # latest worker stage
output/mesh.ply                 # binary little-endian vertices AND triangle faces (unless output mode is Separate)
output/nksr_metadata.json
output/mesh_chunks/chunks.json  # Full/Chunked separate output manifest
output/mesh_chunks/chunk_0000.ply
output/tiles.json               # Low RAM tile manifest
output/tiles/tile_000001/mesh.ply
attempts/previous_<id>/         # previous mesh outputs / state, retained on retries
```

Mesh completion requires a successful worker exit plus an independently parsed,
nonempty triangle PLY with finite vertices and valid face indices. Vertex-only
clouds are rejected. In the Separate and Both output modes every exported cell is
parsed and checked against its manifest as well. Mesh-bound differences are recorded
as PASS or WARNING; point-count differences from the optimized GLIM export are
expected. No ICP or coordinate alignment is applied. A WARNING preserves the mesh
for inspection. A worker exit code of 0 is never sufficient on its own.

## Isolated installation

The verified upstream source is [NVIDIA NKSR](https://github.com/nv-tlabs/NKSR),
commit `e40336845e67761343a756788e5a98b827d4a143` (package 1.0.3).
The helper uses its `environment.yml`, `requirements.txt`, and package build.
It requires x86_64 Linux, Conda, a compatible NVIDIA driver, and a CUDA 12.8
compiler. It creates Python 3.10 and torch 2.7.0+cu128 in a dedicated environment.
No packages are installed into ROS Humble, system Python, or the app's `.venv`.

```bash
# Run from the workstation repository root.
./scripts/setup_nksr.sh
./scripts/check_nksr.sh
```

Source and environment default to `~/.cache/factory-mapping/NKSR` and
`~/.cache/factory-mapping/nksr-env`. `NKSR_CACHE` / `NKSR_PREFIX` can override them.
Use a path without spaces for the compiler environment; application and data paths
containing spaces are supported through argument arrays. Build concurrency defaults
to `MAX_JOBS=1` to fit a 16 GB workstation.

The current upstream requirements mistakenly reference torch-scatter wheels for
PyTorch 2.8 while pinning torch 2.7. The helper preinstalls the matching official
PyG torch 2.7/CUDA 12.8 wheel before installing the remaining upstream requirements.
NKSR also imports `pycg.vis`, so the helper installs Open3D as an upstream
import dependency omitted by PyCG's default extras. Preparation and our PLY writer
do not use Open3D. Conda CUDA target headers are added through `CPATH`; NVIDIA
source is not patched.
Re-running setup reuses the environment and matching installed NKSR build. To
update upstream, review the new environment, API and chunk dispatch, update the
verified commit in the helper, and rebuild explicitly. Nothing installs at web
application startup.

The helper records the interpreter in `.state/nksr_python.txt`. An explicit
`NKSR_PYTHON=/absolute/path/to/env/bin/python` overrides it. Check NKSR starts a
separate process that imports torch/NKSR, loads the real pretrained `ks` checkpoint,
reconstructs a small sensor-oriented sphere, extracts its dual mesh, and validates
its triangle faces. Import alone never reports Ready. Check results include Python,
NKSR, torch, CUDA, GPU, source commit, checkpoint, mesh counts, and timestamps.
They are cached for at most one day; inference checks its runtime again on each run.
Use `NKSR_DEVICE=cpu ./scripts/check_nksr.sh` to test the CPU path explicitly.

The first check/run downloads the pretrained checkpoint through NKSR's official
loader into torch's normal hub checkpoint cache. Logs distinguish DOWNLOADING_MODEL
from LOADING_MODEL, and report CHECKPOINT_DOWNLOAD_FAILED for checkpoint access or
loading failures. The health file is `.state/nksr_health.json`; UI checks also write
`.state/nksr_check.log`. Network failure does not affect GLIM or point preparation.

## Start, prepare, reconstruct

```bash
# Run from the workstation repository root.
./scripts/run_system.sh
```

Open the local app at `http://127.0.0.1:8080`, choose a session in Library, and
select its matching LiDAR trajectory under Surface Reconstruction. Prepare the
points, check NKSR, then click Reconstruct Mesh. The mesh is available through
Download Mesh after validation. The log and stage status remain available on
failure. Cancel Reconstruction stops the managed worker and preserves all inputs.

The Advanced settings keep these two concepts separate:

- **Reconstruction mode: Auto, Full, Chunked, or Low RAM.** How the surface is
  reconstructed.
- **Mesh output: Single mesh, Separate meshes, or Both.** How the finished geometry
  is saved. Internally these remain the `merged`, `chunks`, and `both` values.
- **Chunk size (meters): Auto or a positive number.** One physical setting shared by
  reconstruction partitioning and mesh export. There is no second tile-size control.

Preparation settings are separate:

- **Preparation voxel: 1 cm = 0.01 m.** This selects actual world-space observations
  while keeping point, sensor origin, intensity and timestamp paired. Changing it
  or the trajectory marks the selected input stale and requires preparation again.
- **NKSR target voxel: 2 cm = 0.02 m.** Fixed in the backend independently of
  preparation sampling. The existing detail-level setting is retained as requested
  metadata but does not apply with this explicit target. Changing inference
  settings does not invalidate the prepared points; existing NPZ files work unchanged.
- **Use saved edited geometry (optional, off by default).** Select a completed export
  from one explicitly saved `map_editor` cleanup and choose a positive retention
  tolerance in meters. Preparation transforms the raw bag with that saved map's
  verified trajectory, then retains a raw observation only when its nearest retained
  exported point is within the tolerance, before preparation voxel sampling. Points,
  sensor origins, timestamps and intensity stay paired. This is approximate
  retained-geometry filtering, not an exact deletion mask: NKSR can still bridge a
  deleted region. Merged/offline-viewer edits, missing exports, changed saved maps,
  changed trajectories, unsafe paths and mismatched coordinate frames are rejected;
  no ICP alignment is attempted. The reference index is capped at 2,000,000 exported
  points (about 256 MiB estimated index memory) to avoid host OOM. Re-export and
  prepare again after changing a saved cleanup or tolerance.

Manual preparation, from a fresh shell (replace paths with your scan and choose
a new output directory):

```bash
# Run from the workstation repository root.
source scripts/env.sh
python tools/glim_nksr_prepare.py \
  --bag "data/sessions/SESSION/raw_bag" \
  --trajectory "data/sessions/SESSION/edits/EDIT/map_01/traj_lidar.txt" \
  --output-dir "data/sessions/SESSION/reconstruction/run_manual" \
  --voxel-size 0.01
```

Manual inference does not need ROS activation:

```bash
# Run from the workstation repository root.
NKSR_PYTHON="$(cat .state/nksr_python.txt)"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH OMP_NUM_THREADS=4 \
  "$NKSR_PYTHON" tools/nksr_worker.py \
  --input "data/sessions/SESSION/reconstruction/run_manual/input/nksr_input.npz" \
  --output "data/sessions/SESSION/reconstruction/run_manual/output/mesh.ply" \
  --device cuda --mode auto
```

Manual CLI directories need not be registered as managed UI jobs. Use the UI for
managed preparation, inference, cancellation, and persisted job state. Never choose
an existing output mesh path; the worker refuses to overwrite it.

## Modes and normals

AUTO is the default. It selects full mode for at most 250,000 input points with
at least 3 GiB free GPU memory, otherwise chunks. For automatic chunk size it
examines occupied metric cells for candidate sizes 20, 10, and 5 meters and picks
the largest with no more than 250,000 points per cell, falling back to 5 meters.
This is a conservative heuristic, not a guarantee of memory sufficiency. A CUDA
OOM in AUTO releases tensors/cache and makes at most one chunked retry, capped at
5 meters; the retry's own chunk size is what gets reported. FULL and CHUNKED do not
silently retry or switch inference to CPU.

Full inference passes `detail_level=None`, `voxel_size=0.02`, `approx_kernel_grad=True`,
`solver_tol=1e-4`, `fused_mode=True`, real sensor origins, and the normal preprocessor.
The pinned API applies explicit voxel size in preference to detail level and
handles full-mode scaling internally; the worker passes full-mode coordinates in meters.

Chunked inference uses overlap 0.05 and `detail_level=None`. Its native `ks` voxel
is 0.1 input units: previously this meant 0.1 m (10 cm). The worker now scales
points, sensor origins, and the selected physical chunk size by
`scale = 0.1 / 0.02 = 5.0` before calling NKSR. A 5 m chunk is passed as 25 model
units. Extracted vertices are divided by 5 before validation, bounds, PLY output,
and metadata; faces are unchanged. All output geometry remains in meters.
Chunk selection still operates on the original metric points. No preparation,
normal-estimation, overlap, or extraction settings change.

Upstream does not forward solver tolerance or voxel size through its chunk
dispatcher: no `voxel_size` is passed for chunks, and chunk subproblems retain
the upstream `1e-5` solver tolerance. Reconstruction log events report
`target_voxel_m`, `coordinate_scale`, physical `chunk_size` in meters,
`nksr_chunk_size` in model units, and the full-mode `voxel_size` override.
The worker records requested and actual settings rather than claiming full-mode
settings applied to chunks. Completed chunks are temporarily stored on CPU;
chunked dual-mesh extraction explicitly uses CPU to reduce peak GPU memory. The
standard output path extracts the single fused mesh once and splits it spatially, so
no individual chunk field is meshed independently; see
[Mesh output and the shared chunk size](#mesh-output-and-the-shared-chunk-size).

CUDA normal estimation uses the official
`nksr.get_estimate_normal_preprocess_fn(64, 85.0)`: 64 nearest neighbours and an
**85-degree orientation/inlier threshold**, not a spatial search radius. GLIM
sensor origins orient the normals. Current upstream's default normal-estimation
extension is CUDA-only, despite documentation describing general CPU support.
For explicit CPU inference our wrapper uses batched SciPy KNN/PCA with the same
sensor orientation and angular filtering. The neural field and dual mesh still
come exclusively from official NKSR; this is not another surface algorithm.
CPU inference and extraction can be very slow for large clouds.

## Mesh output and the shared chunk size

Mesh output is independent of the reconstruction mode:

- **Single mesh** (`merged`) saves only `output/mesh.ply`.
- **Separate meshes** (`chunks`) saves only the per-cell PLYs plus their manifest; it
  writes no `output/mesh.ply`.
- **Both** saves the merged mesh and the separate cells.

**Full and Chunked reconstruct once and partition the one final fused surface.**
The separate output is a *spatial split of the final mesh*, not a set of
independently extracted NKSR chunk fields. Independent field meshes can disagree at
their boundaries; partitioning an already fused surface preserves its geometry. This
means Separate does **not** avoid global mesh extraction in Chunked mode, and the
native per-field exporter that remains in `nksr_worker.py` is retained only for
compatibility and future advanced use — the standard output selection never calls it.

Low RAM keeps its existing architecture: every section is reconstructed by its own
isolated subprocess, so sections can differ at their boundaries. It now accepts all
three output selections. With **Separate meshes** the final `merge_meshes()` step is
skipped entirely and no `output/mesh.ply` is created; with **Both** the independent
tiles and the merged mesh are both written.

### Chunk size resolution

`effective_chunk_size_m` is the one physical size in meters. It is used as the NKSR
chunk size in Chunked mode and as the export cell edge everywhere. Resolution:

| Path | Effective size | `chunk_size_source` |
| --- | --- | --- |
| Explicit user value | That value | `user` |
| Chunked, Auto | Existing 20/10/5 m density heuristic | `auto_density` |
| Full, Auto | 5.0 m | `default` |
| Low RAM, Auto | 5.0 m independent tile edge | `default` |
| Auto reconstruction | Resolved mode's rule | as above |
| CUDA OOM retry that shrank the size | The successful retry's size | `oom_retry` |
| Legacy `tile_size` with no `chunk_size` (Low RAM) | That legacy value | `legacy_tile_size` |

Auto resolves the reconstruction mode first and derives the chunk size afterwards, so
an OOM retry never reports the abandoned request. Metadata records
`requested_chunk_size_m`, `effective_chunk_size_m`, and `chunk_size_source`; the UI
shows the effective value after processing, not just the Auto setting. An explicit
`chunk_size` always wins over a legacy `tile_size`, and the legacy field is accepted
only as a Low RAM fallback so old clients and saved metadata keep working.

### Spatial partitioning

`ui/backend/factory_mapping/mesh_partition.py` performs the split in original GLIM
world meters:

- Cells are 3D cubes of `effective_chunk_size_m` on a grid anchored at `[0, 0, 0]`,
  matching the physical NKSR chunk-size setting. The overlap-adjusted NKSR stride is
  never used as the export size.
- Ownership is `cell = floor(centroid / size)` per triangle, computed in float64.
  `floor` handles negative coordinates correctly; the grid is stable when input
  bounds or ordering change, and empty cells produce no file.
- Each triangle belongs to exactly one cell and is kept whole, with unchanged vertex
  positions, winding, geometry, and world scale. A triangle may extend past its
  nominal cell, and neighbouring tiles duplicate shared vertex coordinates without
  being topologically welded. Nothing is welded, decimated, smoothed, resampled,
  voxelized, retriangulated, or clipped.
- Each tile contains only the vertices its triangles reference, renumbered
  contiguously from zero, so loading every tile at its original world coordinates
  reproduces the source triangles without a new local origin.
- Cells are ordered z-major, then y, then x (`(z, y, x)` ascending integer indices),
  so filenames are deterministic.
- Memory stays bounded: faces are classified in batches of 65,536, per-cell face
  references are spooled to disk through at most 64 open handles, and one cell at a
  time is materialized, compacted, and written atomically as `*.partial.ply`.
- `SUM(tile_faces)` equals the source face count. `SUM(tile_vertices)` is usually
  larger than the source vertex count because boundary vertices are duplicated; it is
  recorded as the exported total, never as the source count.

Full and fused-Chunked output still need enough memory to reconstruct and extract the
complete surface. Separate output reduces final file sizes and downstream loading
pressure in Houdini, CloudCompare, or MeshLab, not NKSR's reconstruction peak RAM.

Full/Chunked output directory:

```text
output/
  nksr_metadata.json
  mesh.ply                  # Single mesh / Both only
  mesh_chunks/
    chunks.json
    chunk_0000.ply
    chunk_0001.ply
```

`chunks.json` records `version`, `export_strategy`, the actual `reconstruction_mode`,
`coordinate_system`, `units`, `tile_shape`, `grid_origin_m`, the cell `cell_order`,
`requested_chunk_size_m`, `effective_chunk_size_m`, `chunk_size_source`,
`total_chunks`, `source_faces`, `total_vertices`, `total_faces`, and one entry per
cell with `index`, `grid_index`, `file`, `nominal_bbox_min_m`, `nominal_bbox_max_m`,
`world_bbox_min`, `world_bbox_max`, `vertices`, `faces`, and `file_size_bytes`.
Manifest filenames are bare relative names; validation rejects path traversal,
symlink escapes, duplicate names, missing files, invalid face indices, nonfinite
vertices, and nonpositive effective chunk sizes. Older native per-field manifests
(no `export_strategy`) are still read and validated with the previous behavior.

Success is reported only after this validation passes, so a cancelled or partially
written export is never advertised as complete. Cancellation stays responsive during
partitioning because the worker keeps handling `SIGTERM`/`SIGINT` between batches.

## Low-RAM independent tiles

The original **Auto**, **Full**, and **Chunked** modes remain unchanged. For large
scans, select **Low RAM · independent tiles** in Surface Reconstruction's Advanced
settings and set **Chunk size (meters)**, which is this mode's independent tile edge.
The default is 5 m. This is the edge length of a world-aligned 3D cube, not the
preparation voxel or mesh resolution. Existing prepared input can be reused without
repeating preparation or GLIM.

This mode partitions paired points and sensor origins onto disk in bounded blocks.
Each point belongs to exactly one half-open cube, with indices `floor(point / tile_size)`.
Tiles have no overlap or cross-tile field blending. One fresh subprocess reconstructs
and extracts one tile, saves its mesh, and exits before the next subprocess starts.
The coordinator does not load torch, the model, or the entire point cloud into RAM.
Per-tile inference uses Full mode at the existing 2 cm NKSR target; extraction runs
on CPU with at most 100,000 field-query points per batch. Device, normal settings,
and MISE iterations still apply. Detail level, the original chunk size, and overlap
do not apply to the per-tile reconstruction.

All three mesh output selections are available. With **Single mesh** or **Both** the
final binary PLY is assembled in blocks, preserving world coordinates and adjusting
triangle indices. With **Separate meshes** the merge step is skipped entirely and only
the independent tile PLYs are written, which is the lowest-disk option. Low RAM never
uses native NKSR `FusedField` reconstruction.

Merged Low RAM output is **not stitched or guaranteed watertight**: independent tile
surfaces may have gaps or overlap at their boundaries. Metadata records this
limitation as `validation_status=WARNING` and `boundary_stitching=false`. Use an
original mode when globally blended surfaces are more important than memory.

Tile size bounds the spatial problem, not a hard RAM budget. A dense 5 m tile may
still be too large: try 2 m or 1 m, or fewer MISE iterations. Smaller tiles increase
process/model startup overhead and boundary artifacts. Tiles with fewer points than
Normal KNN or no reconstructed surface are recorded as skipped; if all tiles are
empty, the job fails. Other tile failures stop the job without publishing a final mesh.
Cancellation stops the active child; successful tile outputs remain for inspection.

Outputs under the run's `output/` directory include:

```text
tiles.json                         # tile cells, counts, outcomes, shared chunk size
tiles/tile_000001/mesh.ply          # independent tile surface
tiles/tile_000001/nksr_metadata.json
tiles/tile_000001/progress.json
mesh.ply                           # merged mesh; omitted in Separate meshes mode
nksr_metadata.json                  # aggregate settings, counts, boundary warning
```

Temporary decompressed input and partition files are removed on normal completion,
handled failure, or cancellation. Allow disk space for those temporary files and
both tile and merged meshes. An uncatchable kill or power loss can leave temporary
files; UI retries archive the previous output directory rather than overwrite it.
Retries currently start again, rather than resume individual tiles. Validation reads
one tile at a time, so it does not load every tile mesh at once.

Manual use with the isolated NKSR interpreter (the legacy `--tile-size` alias is
still accepted for the Low RAM tile edge, but `--chunk-size` is preferred):

```bash
"$NKSR_PYTHON" tools/nksr_worker.py \
  --input "path/to/input/nksr_input.npz" \
  --output "path/to/new-output/mesh.ply" \
  --mode low_ram --chunk-size 5 --mesh-output-mode merged --device cuda
```

## Tests and licensing

```bash
# Run from the workstation repository root.
PYTHONPATH="$PWD/ui/backend" .venv/bin/python -m pytest -q
RUN_NKSR_INTEGRATION=1 PYTHONPATH="$PWD/ui/backend" \
  .venv/bin/python -m pytest tests/test_nksr.py::test_real_nksr_integration -q
```

Ordinary CI mocks only the subprocess/runtime boundary; it needs no GPU or model
download. The opt-in test uses the first 20,000 records from `NKSR_TEST_INPUT` (or the local
validation run, if present), otherwise a small spherical input. It calls the
installed official worker and verifies real
vertices and triangles. Runtime installation and large meshes/checkpoints stay out
of git.

NKSR code is covered by the [NVIDIA Source Code License](https://github.com/nv-tlabs/NKSR/blob/main/LICENSE.txt).
The [upstream README](https://github.com/nv-tlabs/NKSR#testing-nksr-on-your-own-data)
identifies the kitchen-sink weights as CC-BY-SA 4.0. Consult those upstream terms
for use and distribution; this repository does not relicense or bundle the source
or checkpoint.

## Verified workstation run — 2026-10-05

This historical run predates the 2 cm metric-scale fix above and used the native
10 cm chunked voxel. Its counts and timings are not validation of the new resolution.

Official NKSR 1.0.3, commit `e40336845e67761343a756788e5a98b827d4a143`,
compiled without source modifications. Python 3.10.21, torch 2.7.0+cu128,
CUDA runtime/toolchain 12.8, torch-scatter 2.1.2+pt27cu128, Open3D 0.20.0.
GPU: NVIDIA GeForce RTX 5060 (8 GB), driver 580.178.04.
The kitchen-sink checkpoint downloaded successfully. Real CUDA and explicit CPU
smoke tests each generated 1,394 vertices / 2,780 triangles. The opt-in test also
passed on 20,000 actual prepared factory observations.

Session `20261003_135316_Scan_2026-10-03_13_53_16`, run `run_22bfe735334d`:

| Measurement | Result |
| --- | --- |
| Raw valid observations | 3,340,321 |
| Prepared points, 1 cm sampling | 1,806,674 |
| Input extent (meters) | 17.117 × 28.962 × 10.889 |
| Requested / actual mode | AUTO / CHUNKED |
| Chunk size / overlap | 5 m / 0.05 (56 upstream chunks) |
| Normal KNN / drop threshold | 64 / 85 degrees, upstream CUDA estimator |
| Detail level used | None (0.5 requested, inapplicable to chunks) |
| NKSR internal voxel override | None |
| OOM retries | 0 |
| GPU total / free before (torch) | 7.517 / 6.885 GiB |
| Peak torch allocated memory | 658.03 MiB (not total process/driver memory) |
| NKSR inference | 15.90 s |
| Dual mesh extraction (CPU) | 30.18 s |
| Worker elapsed | 55.58 s |
| Vertices / triangles | 323,786 / 629,328 |
| Binary mesh file size | 12,066,875 bytes |
| Bounds sanity result | PASS |

The 20/10/5-meter candidates had maximum occupied-cell counts of
509,039 / 505,227 / 356,239. AUTO chose its conservative 5-meter fallback.
This run establishes a working chunk configuration on this machine; full mode
was not attempted on the 1.8M-point dataset. Full mode passed the smaller tests.

Input bounds: `[-2.813, -23.451, -5.408]` to `[14.305, 5.511, 5.481]` meters.
Mesh bounds: `[-2.822, -23.625, -3.909]` to `[14.521, 5.464, 5.470]` meters.
Maximum boundary differences per axis are 0.216 / 0.174 / 1.499 meters. This is
a coarse world-space sanity check, not a surface-accuracy assessment; the lowest
input points are not represented by the mesh. No ICP was applied.

The actual successful call was:

```python
reconstructor = nksr.Reconstructor(torch.device('cuda:0'), config='ks')
reconstructor.chunk_tmp_device = torch.device('cpu:0')
preprocess_fn = nksr.get_estimate_normal_preprocess_fn(64, 85.0)
field = reconstructor.reconstruct(
    input_xyz, sensor=input_sensor,
    detail_level=None, chunk_size=5.0, overlap_ratio=0.05,
    approx_kernel_grad=True, fused_mode=True, preprocess_fn=preprocess_fn,
)
field.to_('cpu:0')
reconstructor.network.to('cpu:0')
mesh = field.extract_dual_mesh(mise_iter=1)
```

The successful production job and outputs are in:

```text
data/sessions/20261003_135316_Scan_2026-10-03_13_53_16/reconstruction/run_22bfe735334d/
```

`exports/run_003_13b1a4a9.ply` remained unchanged, SHA-256:
`ded9f09b2e7429dc77660abaf128194548d92a9ca4da019652a1449e7c3e7ae4`.
Existing GLIM export regression tests continue to pass, including missing-NKSR
isolation. This task did not invoke GLIM's exporter to overwrite any existing file.

Final validation: `RUN_NKSR_INTEGRATION=1 PYTHONPATH="$PWD/ui/backend" .venv/bin/python -m pytest -q`
completed with **136 passed** (one existing Starlette/AnyIO deprecation warning).
The ordinary suite completed with **135 passed, 1 opt-in test skipped**.
Browser checks verified prepared/stale states, disabled detail level for chunks,
READY GPU status, and the real completed mesh counts/download control, with no
browser console errors. A second setup-helper run completed successfully without
rebuilding the matching NKSR installation.
