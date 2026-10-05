# Optional pretrained NKSR surface reconstruction

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
output/mesh.ply                 # binary little-endian vertices AND triangle faces
output/nksr_metadata.json
attempts/previous_<id>/         # previous mesh outputs / state, retained on retries
```

Mesh completion requires a successful worker exit plus an independently parsed,
nonempty triangle PLY with finite vertices and valid face indices. Vertex-only
clouds are rejected. Mesh-bound differences are recorded as PASS or WARNING;
point-count differences from the optimized GLIM export are expected. No ICP or
coordinate alignment is applied. A WARNING preserves the mesh for inspection.

## Isolated installation

The verified upstream source is [NVIDIA NKSR](https://github.com/nv-tlabs/NKSR),
commit `e40336845e67761343a756788e5a98b827d4a143` (package 1.0.3).
The helper uses its `environment.yml`, `requirements.txt`, and package build.
It requires x86_64 Linux, Conda, a compatible NVIDIA driver, and a CUDA 12.8
compiler. It creates Python 3.10 and torch 2.7.0+cu128 in a dedicated environment.
No packages are installed into ROS Humble, system Python, or the app's `.venv`.

```bash
cd "/home/ubuntu-ros/Documents/GLIM Factory Mapping/factory_mapping"
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
Conda CUDA target headers are added through `CPATH`; NVIDIA source is not patched.
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
cd "/home/ubuntu-ros/Documents/GLIM Factory Mapping/factory_mapping"
./scripts/run_system.sh
```

Open the local app at `http://127.0.0.1:8080`, choose a session in Library, and
select its matching LiDAR trajectory under Surface Reconstruction. Prepare the
points, check NKSR, then click Reconstruct Mesh. The mesh is available through
Download Mesh after validation. The log and stage status remain available on
failure. Cancel Reconstruction stops the managed worker and preserves all inputs.

The Advanced settings keep these two concepts separate:

- **Preparation voxel: 1 cm = 0.01 m.** This selects actual world-space observations
  while keeping point, sensor origin, intensity and timestamp paired. Changing it
  or the trajectory marks the selected input stale and requires preparation again.
- **NKSR detail level: 0.5.** Full inference uses this, with internal
  `voxel_size=None`. Sampling size is never passed as NKSR's internal voxel size.
  Changing inference settings does not invalidate the prepared points.

Manual preparation, from a fresh shell (replace paths with your scan and choose
a new output directory):

```bash
cd "/home/ubuntu-ros/Documents/GLIM Factory Mapping/factory_mapping"
source scripts/env.sh
python tools/glim_nksr_prepare.py \
  --bag "data/sessions/SESSION/raw_bag" \
  --trajectory "data/sessions/SESSION/edits/EDIT/map_01/traj_lidar.txt" \
  --output-dir "data/sessions/SESSION/reconstruction/run_manual" \
  --voxel-size 0.01
```

Manual inference does not need ROS activation:

```bash
cd "/home/ubuntu-ros/Documents/GLIM Factory Mapping/factory_mapping"
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
5 meters. FULL and CHUNKED do not silently retry or switch inference to CPU.

Full inference passes `detail_level`, `voxel_size=None`, `approx_kernel_grad=True`,
`solver_tol=1e-4`, `fused_mode=True`, real sensor origins, and the normal preprocessor.
Chunked inference uses metric `chunk_size`, overlap 0.05, and `detail_level=None`.
Upstream does not forward solver tolerance or internal voxel size through its
chunk dispatcher: chunk subproblems use the upstream `1e-5` solver tolerance.
The worker records requested and actual settings rather than claiming full-mode
settings applied to chunks. Completed chunks are temporarily stored on CPU;
chunked dual-mesh extraction explicitly uses CPU to reduce peak GPU memory.

CUDA normal estimation uses the official
`nksr.get_estimate_normal_preprocess_fn(64, 85.0)`: 64 nearest neighbours and an
**85-degree orientation/inlier threshold**, not a spatial search radius. GLIM
sensor origins orient the normals. Current upstream's default normal-estimation
extension is CUDA-only, despite documentation describing general CPU support.
For explicit CPU inference our wrapper uses batched SciPy KNN/PCA with the same
sensor orientation and angular filtering. The neural field and dual mesh still
come exclusively from official NKSR; this is not another surface algorithm.
CPU inference and extraction can be very slow for large clouds.

## Tests and licensing

```bash
cd "/home/ubuntu-ros/Documents/GLIM Factory Mapping/factory_mapping"
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
