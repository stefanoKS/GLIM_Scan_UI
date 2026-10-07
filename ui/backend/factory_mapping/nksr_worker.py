"""Official pretrained NKSR worker. Runs ONLY in the isolated NKSR interpreter."""
import argparse
import gc
import json
from pathlib import Path
import signal
import sys
import tempfile
import time
import traceback
import numpy as np
from .nksr_mesh import write_mesh

DEFAULT_NKSR_TARGET_VOXEL_M = 0.02
# Native ks voxel at pinned NKSR e403368; independent of preparation sampling.
NKSR_NATIVE_VOXEL_SIZE = 0.1
MESH_OUTPUT_MODES = ('merged', 'chunks', 'both')
CHUNKS_DIR_NAME = 'mesh_chunks'


class WorkerError(RuntimeError):
    def __init__(self, code, message, details=None):
        super().__init__(message); self.code = code; self.details = details or {}


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False)); temp.replace(path)


def classify(error, stage):
    if isinstance(error, WorkerError): return error.code
    if 'out of memory' in str(error).lower(): return 'CUDA_OOM'
    if stage == 'DOWNLOADING_MODEL': return 'CHECKPOINT_DOWNLOAD_FAILED'
    if stage == 'EXTRACTING_MESH': return 'MESH_EXTRACTION_FAILED'
    if stage == 'SAVING_MESH': return 'MESH_INVALID'
    if stage == 'LOADING_INPUT': return 'INPUT_INVALID'
    return 'NKSR_RECONSTRUCTION_FAILED'


def load_input(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            points = np.asarray(data['points'], dtype=np.float32)
            sensors = np.asarray(data['sensor_origins'], dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or sensors.shape != points.shape:
            raise ValueError('Expected nonempty paired points and sensor_origins with shape [N,3]')
        if not np.isfinite(points).all() or not np.isfinite(sensors).all():
            raise ValueError('Input contains nonfinite values')
        return points, sensors
    except Exception as error:
        raise WorkerError('INPUT_INVALID', str(error)) from error


def to_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def finite_or_none(values):
    """JSON-safe coordinates: unbounded (-inf/+inf) become null."""
    return [None if not np.isfinite(value) else float(value) for value in np.asarray(values).ravel()]


def chunk_grid_from_centers(centers):
    """Per-axis chunk stride and grid indices derived from the actual chunk centers.

    NKSR lays chunk centers on a regular axis-aligned grid. Empty chunks are
    never reconstructed by NKSR, so the returned grid index is the 0-based rank
    of each center along its axis rather than the absolute NKSR cell index.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or not len(centers):
        raise WorkerError('EMPTY_TILE', 'NKSR produced no chunk fields')
    strides = np.full(3, np.inf)
    grid_index = np.zeros((len(centers), 3), dtype=np.int64)
    for axis in range(3):
        values, inverse = np.unique(centers[:, axis], return_inverse=True)
        if len(values) > 1:
            strides[axis] = float(np.min(np.diff(values)))
        grid_index[:, axis] = inverse
    return strides, grid_index


def chunk_core_bounds(centers):
    """Half-open [min, max) core ownership bounds per chunk, in scaled coordinates.

    Adjacent chunks split each axis at the midpoint between their centers, so
    every region belongs to exactly one chunk and no triangle is exported by two
    neighboring chunks. The outermost chunk along each axis keeps its outer
    boundary by extending to +/-inf.
    """
    centers = np.asarray(centers, dtype=np.float64)
    bounds = []
    for center in centers:
        lo = np.full(3, -np.inf)
        hi = np.full(3, np.inf)
        for axis in range(3):
            values = np.unique(centers[:, axis])
            pos = int(np.searchsorted(values, center[axis]))
            if pos > 0:
                lo[axis] = 0.5 * (values[pos - 1] + center[axis])
            if pos + 1 < len(values):
                hi[axis] = 0.5 * (center[axis] + values[pos + 1])
        bounds.append((lo, hi))
    return bounds


def crop_mesh_to_core(vertices, faces, core_min, core_max):
    """Remove triangles whose centroid lies outside the half-open core region.

    Triangle ownership by centroid is deterministic: each centroid falls in
    exactly one chunk core, so overlap regions are never exported twice.
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if faces.size == 0:
        return vertices[:0].copy(), faces[:0].copy()
    centroids = vertices[faces].mean(axis=1)
    keep = np.all(centroids >= core_min, axis=1) & np.all(centroids < core_max, axis=1)
    faces = faces[keep]
    if faces.size == 0:
        return vertices[:0].copy(), faces[:0].copy()
    used = np.zeros(len(vertices), dtype=bool)
    used[faces.reshape(-1)] = True
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    return vertices[used].copy(), remap[faces]


def select_chunk(points):
    # Largest candidate with at most 250k observations per nonoverlapped cell.
    # Metric candidates are checked against actual density, not just the bounds.
    for size in (20., 10., 5.):
        _, counts = np.unique(np.floor(points/size).astype(np.int64), axis=0, return_counts=True)
        if counts.max() <= 250_000: return size
    return 5.


def reconstruction_kwargs(settings, mode, chunk_size, preprocess):
    result = dict(sensor=None, detail_level=None,
                  approx_kernel_grad=True, fused_mode=True, preprocess_fn=preprocess)
    if mode == 'chunked':
        # Upstream does NOT forward solver_tol/voxel_size through this path.
        result.update(chunk_size=chunk_size, overlap_ratio=settings.overlap_ratio)
    else:
        result.update(voxel_size=DEFAULT_NKSR_TARGET_VOXEL_M, solver_tol=1e-4)
    return result


def runtime(device_setting):
    try:
        import torch
        import nksr
    except (ImportError, OSError) as error:
        raise WorkerError('NKSR_NOT_INSTALLED', f'NKSR environment cannot import its dependencies: {error}') from error
    available = torch.cuda.is_available()
    # An explicit CPU index also avoids upstream get_device calling current_device on CPU-only hosts.
    device = torch.device('cuda:0' if available and device_setting != 'cpu' else 'cpu:0')
    metadata = dict(python_version=sys.version.split()[0], python=sys.executable,
                    nksr_version=nksr.__version__, torch_version=torch.__version__,
                    cuda_runtime=torch.version.cuda, cuda_available=available,
                    gpu_name=torch.cuda.get_device_name(0) if available else None,
                    device=str(device), config='ks', checkpoint='ks',
                    nksr_git_commit=None)
    provenance = Path(sys.prefix)/'nksr-provenance.json'
    if provenance.is_file(): metadata.update(json.loads(provenance.read_text()))
    if device_setting == 'cuda' and not available:
        raise WorkerError('CUDA_UNAVAILABLE', 'CUDA is unavailable in the NKSR environment', metadata)
    return torch, nksr, device, metadata


def load_model(torch, nksr, device, event):
    from nksr.configs import get_hparams, load_checkpoint_from_url
    url = get_hparams('ks').url
    cached = Path(torch.hub.get_dir())/'checkpoints'/url.rsplit('/',1)[-1]
    event('LOADING_MODEL' if cached.is_file() else 'DOWNLOADING_MODEL', checkpoint_url=url)
    # Download separately so connectivity failures are distinguished from inference failures.
    try:
        load_checkpoint_from_url(url)
    except Exception as error:
        raise WorkerError('CHECKPOINT_DOWNLOAD_FAILED', f'Kitchen-sink checkpoint could not be loaded: {error}') from error
    event('LOADING_MODEL', checkpoint_cached=True)
    return nksr.Reconstructor(device, config='ks')


def cpu_normal_preprocess(knn, drop_angle):
    """CPU PCA equivalent of the upstream CUDA-only nearest-neighbour extension.

    No alternate surface method: only normal estimation runs here; the field and
    mesh still come from the official pretrained NKSR model.
    """
    from scipy.spatial import cKDTree
    def preprocess(xyz, normal, sensor):
        if normal is not None or sensor is None:
            raise WorkerError('INPUT_INVALID', 'CPU normal estimation requires points and sensor origins')
        if len(xyz) < knn: return None
        points=xyz.detach().cpu().numpy(); origins=sensor.detach().cpu().numpy()
        tree=cKDTree(points); normals=np.empty_like(points)
        for start in range(0,len(points),8192):
            block=points[start:start+8192]
            _, indices=tree.query(block,k=knn,workers=4)
            neighbours=points[indices].reshape(len(block),knn,3)
            centered=neighbours-neighbours.mean(axis=1,keepdims=True)
            covariance=np.einsum('nki,nkj->nij',centered,centered)
            _,vectors=np.linalg.eigh(covariance)
            normals[start:start+len(block)]=vectors[:,:,0]
        rays=origins-points
        rays/=np.maximum(np.linalg.norm(rays,axis=1,keepdims=True),1e-6)
        cosine=np.einsum('ij,ij->i',normals,rays)
        normals[cosine<0]*=-1
        keep=np.isfinite(normals).all(axis=1)&(abs(cosine)>np.cos(np.deg2rad(drop_angle)))
        import torch
        return xyz[keep],torch.from_numpy(normals[keep]).to(xyz),None
    return preprocess


def extract_and_save_chunks(fields, centers, rotations, settings, scale, chunks_dir, event, torch, device):
    """Extract, crop, transform and save each NKSR chunk field independently.

    Chunk meshes are processed one at a time and never held simultaneously. Each
    field is moved to CPU before extraction (the existing small-memory recipe),
    transformed into scaled global coordinates, cropped to its half-open core
    ownership region, compacted, and written in world meters.
    """
    strides, grid_index = chunk_grid_from_centers(centers)
    bounds = chunk_core_bounds(centers)
    chunks_dir = Path(chunks_dir)
    if chunks_dir.exists():
        raise WorkerError('MESH_INVALID', 'Chunk output exists; choose a new output directory')
    chunks_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    total_vertices = total_faces = 0
    union_min = np.full(3, np.inf)
    union_max = np.full(3, -np.inf)
    total = len(fields)
    for index, (field, center, rotation) in enumerate(zip(fields, centers, rotations)):
        event('EXTRACTING_CHUNK_MESH', chunk=index + 1, total_chunks=total)
        field.to_('cpu:0')
        mesh = field.extract_dual_mesh(mise_iter=settings.mise_iter)
        local_vertices = to_numpy(mesh.v)
        faces = to_numpy(mesh.f)
        del mesh
        if rotation is None:
            global_vertices = local_vertices + center
        else:
            global_vertices = local_vertices @ np.asarray(rotation).T + center
        core_min, core_max = bounds[index]
        core_vertices, core_faces = crop_mesh_to_core(global_vertices, faces, core_min, core_max)
        del local_vertices, global_vertices, faces
        entry = dict(index=index, grid_index=grid_index[index].tolist(),
                     field_origin_scaled=center.tolist(),
                     core_bbox_min=finite_or_none(core_min), core_bbox_max=finite_or_none(core_max))
        if core_faces.size == 0:
            entry.update(file=None, vertices=0, faces=0, world_bbox_min=None, world_bbox_max=None,
                         note='No triangles inside the chunk core region')
        else:
            world_vertices = core_vertices / scale
            filename = f'chunk_{index:04d}.ply'
            stats = write_mesh(chunks_dir / filename, world_vertices, core_faces)
            entry.update(file=filename, vertices=int(stats['vertex_count']), faces=int(stats['face_count']),
                         world_bbox_min=stats['bounding_box_min'], world_bbox_max=stats['bounding_box_max'])
            total_vertices += stats['vertex_count']
            total_faces += stats['face_count']
            union_min = np.minimum(union_min, world_vertices.min(axis=0))
            union_max = np.maximum(union_max, world_vertices.max(axis=0))
            del world_vertices
        entries.append(entry)
        del core_vertices, core_faces, field
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        event('SAVING_CHUNK_MESH', chunk=index + 1, total_chunks=total,
              vertices=entry['vertices'], faces=entry['faces'])
    manifest = dict(version=1, coordinate_system='GLIM_world', units='meters',
                    output_mode=settings.mesh_output_mode,
                    target_voxel_m=DEFAULT_NKSR_TARGET_VOXEL_M, coordinate_scale=scale,
                    nksr_chunk_size_scaled=settings.chunk_size * scale if settings.chunk_size else None,
                    overlap_ratio=settings.overlap_ratio,
                    total_chunks=total, total_vertices=total_vertices, total_faces=total_faces,
                    chunk_stride_scaled=[None if not np.isfinite(s) else float(s) for s in strides],
                    ownership_rule='Half-open midpoint between adjacent chunk centers; '
                                   'outer chunks keep their outer boundary; triangle-centroid ownership.',
                    chunks=entries)
    atomic_json(chunks_dir / 'chunks.json', manifest)
    union = None if not np.isfinite(union_min).all() else (union_min.tolist(), union_max.tolist())
    return manifest, dict(total_vertices=total_vertices, total_faces=total_faces,
                          union_bounds=union)


def write_full_single_chunk(settings, vertices, faces, chunks_dir):
    """Export a full (non-chunked) reconstruction as one chunk plus its manifest."""
    chunks_dir = Path(chunks_dir)
    if chunks_dir.exists():
        raise WorkerError('MESH_INVALID', 'Chunk output exists; choose a new output directory')
    chunks_dir.mkdir(parents=True, exist_ok=True)
    stats = write_mesh(chunks_dir / 'chunk_0000.ply', vertices, faces)
    entry = dict(index=0, grid_index=[0, 0, 0], field_origin_scaled=[0.0, 0.0, 0.0],
                 core_bbox_min=None, core_bbox_max=None, file='chunk_0000.ply',
                 vertices=int(stats['vertex_count']), faces=int(stats['face_count']),
                 world_bbox_min=stats['bounding_box_min'], world_bbox_max=stats['bounding_box_max'],
                 note='Full (non-chunked) reconstruction exported as a single chunk')
    manifest = dict(version=1, coordinate_system='GLIM_world', units='meters',
                    output_mode=settings.mesh_output_mode,
                    target_voxel_m=DEFAULT_NKSR_TARGET_VOXEL_M, coordinate_scale=1.0,
                    nksr_chunk_size_scaled=None, overlap_ratio=None,
                    total_chunks=1, total_vertices=int(stats['vertex_count']),
                    total_faces=int(stats['face_count']), chunks=[entry])
    atomic_json(chunks_dir / 'chunks.json', manifest)
    return manifest, dict(total_vertices=int(stats['vertex_count']), total_faces=int(stats['face_count']),
                          union_bounds=(stats['bounding_box_min'], stats['bounding_box_max']))


def execute(points, sensors, settings, event, torch, nksr, device, reconstructor):
    free = torch.cuda.mem_get_info(device)[0] if device.type == 'cuda' else None
    output_mode = getattr(settings, 'mesh_output_mode', 'merged')
    mode = settings.mode
    if mode == 'auto': mode = 'full' if len(points) <= 250_000 and (free is None or free >= 3*1024**3) else 'chunked'
    chunk_size = settings.chunk_size or (select_chunk(points) if mode == 'chunked' or settings.mode == 'auto' else None)
    attempts = []
    reconstructor.chunk_tmp_device = torch.device('cpu:0')
    preprocess = nksr.get_estimate_normal_preprocess_fn(settings.normal_knn, settings.normal_drop_angle_deg)
    if device.type=='cpu': preprocess=cpu_normal_preprocess(settings.normal_knn,settings.normal_drop_angle_deg)
    manifest = None
    chunk_totals = None
    chunk_seconds = None

    def attempt(current_mode):
        nonlocal manifest, chunk_totals, chunk_seconds
        xyz = torch.from_numpy(points).float().to(device)
        sensor = torch.from_numpy(sensors).float().to(device)
        kwargs = reconstruction_kwargs(settings, current_mode, chunk_size, preprocess)
        scale = NKSR_NATIVE_VOXEL_SIZE / DEFAULT_NKSR_TARGET_VOXEL_M if current_mode == 'chunked' else 1.0
        if current_mode == 'chunked':
            xyz = xyz * scale
            sensor = sensor * scale
            kwargs['chunk_size'] = chunk_size * scale
        kwargs['sensor'] = sensor
        event('RECONSTRUCTING', requested_mode=settings.mode, actual_mode=current_mode,
              mesh_output_mode=output_mode,
              chunk_size=chunk_size if current_mode == 'chunked' else None,
              detail_level=kwargs['detail_level'], target_voxel_m=DEFAULT_NKSR_TARGET_VOXEL_M,
              coordinate_scale=scale, nksr_chunk_size=kwargs.get('chunk_size'),
              voxel_size=kwargs.get('voxel_size'))
        started = time.monotonic()
        with torch.inference_mode():
            field = reconstructor.reconstruct(xyz, **kwargs)
            if field is None:
                raise WorkerError('EMPTY_TILE' if settings.tile_worker else 'NKSR_RECONSTRUCTION_FAILED',
                                  'NKSR returned no field after normal filtering')
            reconstruct_seconds = time.monotonic()-started
            del xyz, sensor, kwargs
            # Per-chunk export: reuse the individual chunk fields NKSR already built.
            if current_mode == 'chunked' and output_mode in ('chunks', 'both'):
                if not (hasattr(field, 'fields') and hasattr(field, 'transforms')):
                    raise WorkerError('NKSR_CHUNK_FIELDS_UNAVAILABLE',
                                      'NKSR did not expose individual chunk fields; per-chunk export is unavailable')
                fields = field.fields
                transforms = field.transforms
                if not fields:
                    raise WorkerError('EMPTY_TILE', 'NKSR produced no chunk fields')
                centers = np.stack([np.asarray(t.t, dtype=np.float64) for t in transforms])
                rotations = [np.asarray(t.q.rotation_matrix, dtype=np.float64) for t in transforms]
                event('EXTRACTING_CHUNK_MESHES', extraction_device='cpu', total_chunks=len(fields),
                      message='Per-chunk extraction on CPU')
                reconstructor.network.to('cpu:0')
                if device.type == 'cuda': torch.cuda.empty_cache()
                chunks_dir = settings.output.parent / CHUNKS_DIR_NAME
                chunk_started = time.monotonic()
                manifest, chunk_totals = extract_and_save_chunks(
                    fields, centers, rotations, settings, scale, chunks_dir, event, torch, device)
                chunk_seconds = time.monotonic() - chunk_started
                if output_mode == 'chunks':
                    del field, fields, transforms, centers, rotations
                    gc.collect()
                    if device.type == 'cuda': torch.cuda.empty_cache()
                    return None, None, reconstruct_seconds, 0.0, manifest, chunk_totals, chunk_seconds
                # 'both': fall through to the fused extraction below; chunk fields are on CPU.
            # Official small-memory recipe: CPU extraction, no silent CPU inference fallback.
            if current_mode == 'chunked' or settings.tile_worker:
                message='Independent tile extraction on CPU' if settings.tile_worker else 'Chunked mesh extraction uses CPU and may be slow'
                event('EXTRACTING_MESH', extraction_device='cpu', message=message)
                field.to_('cpu:0'); reconstructor.network.to('cpu:0')
                if device.type == 'cuda': torch.cuda.empty_cache()
            else: event('EXTRACTING_MESH', extraction_device=str(device))
            started = time.monotonic()
            extraction=dict(mise_iter=settings.mise_iter)
            if settings.tile_worker: extraction['max_points']=100000
            mesh = field.extract_dual_mesh(**extraction)
            if settings.tile_worker and (not len(mesh.v) or not len(mesh.f)):
                raise WorkerError('EMPTY_TILE','Tile produced no triangles')
            vertices = to_numpy(mesh.v)
            if current_mode == 'chunked': vertices = vertices / scale
            return vertices, to_numpy(mesh.f), reconstruct_seconds, time.monotonic()-started, manifest, chunk_totals, chunk_seconds

    for retry in range(2):
        try:
            vertices, faces, recons_seconds, extraction_seconds, manifest, chunk_totals, chunk_seconds = attempt(mode)
            break
        except Exception as error:
            if classify(error, 'RECONSTRUCTING') != 'CUDA_OOM' or settings.mode != 'auto' or retry:
                raise
            attempts.append(dict(mode=mode, error='CUDA_OOM'))
            event('RECONSTRUCTING', message='CUDA OOM: releasing tensors; one chunked retry', actual_mode='chunked')
        # Outside except: traceback/tensors can now be collected.
        gc.collect(); torch.cuda.empty_cache()
        mode = 'chunked'; chunk_size = min(chunk_size or 5., 5.)
        reconstructor.network.to(device)
    result = dict(requested_mode=settings.mode, actual_mode=mode, mesh_output_mode=output_mode,
        chunk_count=None,
        chunk_size=chunk_size if mode == 'chunked' else None, overlap_ratio=settings.overlap_ratio,
        requested_detail_level=settings.detail_level, detail_level=None,
        nksr_internal_voxel_size=DEFAULT_NKSR_TARGET_VOXEL_M if mode == 'full' else None,
        target_voxel_m=DEFAULT_NKSR_TARGET_VOXEL_M,
        coordinate_scale=NKSR_NATIVE_VOXEL_SIZE / DEFAULT_NKSR_TARGET_VOXEL_M if mode == 'chunked' else 1.0,
        normal_knn=settings.normal_knn,
        normal_drop_angle_deg=settings.normal_drop_angle_deg, mise_iter=settings.mise_iter,
        normal_backend='nksr_cuda' if device.type=='cuda' else 'scipy_cpu_pca',
        approx_kernel_grad=True, fused_mode=True, solver_tol=1e-4 if mode == 'full' else 1e-5,
        solver_note='Upstream chunk dispatch uses default solver_tol=1e-5' if mode == 'chunked' else None,
        attempts=attempts, reconstruction_seconds=recons_seconds, extraction_seconds=extraction_seconds,
        gpu_free_before=free, gpu_free_after=torch.cuda.mem_get_info(device)[0] if device.type=='cuda' else None,
        gpu_total_memory=torch.cuda.mem_get_info(device)[1] if device.type=='cuda' else None,
        gpu_peak_allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
    if manifest is not None:
        result.update(chunk_manifest=manifest, chunk_count=manifest['total_chunks'],
                      chunk_vertices_total=chunk_totals['total_vertices'],
                      chunk_faces_total=chunk_totals['total_faces'],
                      chunk_union_bounds=chunk_totals['union_bounds'],
                      chunk_extraction_seconds=chunk_seconds)
    return vertices, faces, result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path); p.add_argument('--output', type=Path)
    p.add_argument('--progress', type=Path); p.add_argument('--metadata', type=Path)
    p.add_argument('--check', action='store_true', help='Real pretrained model + inference + mesh smoke test')
    p.add_argument('--health-output', type=Path)
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--mode', choices=['auto','full','chunked','low_ram'], default='auto')
    p.add_argument('--mesh-output-mode', choices=list(MESH_OUTPUT_MODES), default='merged',
                   help='merged: fused mesh only; chunks: per-chunk meshes only; both: chunks plus fused mesh')
    p.add_argument('--tile-size', type=float, default=5., help='Independent low-RAM tile edge length in meters')
    p.add_argument('--tile-worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--detail-level', type=float, default=.5)
    p.add_argument('--chunk-size', type=float)
    p.add_argument('--overlap-ratio', type=float, default=.05)
    p.add_argument('--normal-knn', type=int, default=64)
    p.add_argument('--normal-drop-angle-deg', type=float, default=85.)
    p.add_argument('--mise-iter', type=int, default=1)
    return p


def main():
    settings = parser().parse_args()
    if not (0 <= settings.detail_level <= 1 and 0 <= settings.overlap_ratio < 1 and
            0 < settings.normal_drop_angle_deg <= 90 and settings.normal_knn > 0 and
            0 <= settings.mise_iter <= 4 and np.isfinite(settings.tile_size) and settings.tile_size > 0 and
            (not settings.tile_worker or settings.mode=='full') and
            (settings.chunk_size is None or np.isfinite(settings.chunk_size) and settings.chunk_size > 0)):
        raise SystemExit('Invalid reconstruction settings')
    stage = 'LOADING_INPUT'; metadata = {'python':sys.executable}; started=time.monotonic()
    def event(next_stage, **fields):
        nonlocal stage
        stage=next_stage
        obj=dict(stage=stage, **fields)
        print(json.dumps(obj), flush=True)
        if settings.progress: atomic_json(settings.progress, obj)
    def cancel(signum, frame): raise WorkerError('CANCELLED','Reconstruction cancelled')
    signal.signal(signal.SIGTERM,cancel); signal.signal(signal.SIGINT,cancel)
    try:
        event('LOADING_INPUT')
        if settings.check:
            # A real sensor-oriented sphere test; the resulting surface is always produced by NKSR.
            rng=np.random.default_rng(42); points=rng.normal(size=(2048,3)).astype(np.float32)
            points /= np.linalg.norm(points,axis=1,keepdims=True); sensors=points*3
            scratch=tempfile.TemporaryDirectory(prefix='nksr-smoke-')
            settings.output=Path(scratch.name)/'mesh.ply'
        else:
            if not settings.input or not settings.output: raise WorkerError('INPUT_INVALID','--input and --output are required')
            if settings.mode=='low_ram':
                from .nksr_tiled import run_tiled
                return run_tiled(settings,event)
            points, sensors=load_input(settings.input)
        if settings.output.exists(): raise WorkerError('MESH_INVALID','Output exists; choose a new output path')
        torch,nksr,device,metadata=runtime(settings.device)
        event('LOADING_MODEL', **metadata)
        if device.type=='cpu': event('LOADING_MODEL',message='CPU requested/selected; large maps can be extremely slow')
        else: torch.cuda.reset_peak_memory_stats(device)
        reconstructor=load_model(torch,nksr,device,event)
        vertices,faces,result=execute(points,sensors,settings,event,torch,nksr,device,reconstructor)
        metadata.update(result)
        output_mode = settings.mesh_output_mode
        chunks_dir = settings.output.parent / CHUNKS_DIR_NAME
        write_merged = output_mode != 'chunks'
        chunk_totals = None
        if result.get('chunk_manifest') is not None:
            chunk_totals = dict(total_vertices=result['chunk_vertices_total'],
                                total_faces=result['chunk_faces_total'],
                                union_bounds=result['chunk_union_bounds'])
        # A full (non-chunked) reconstruction has no individual fields; export it as
        # one chunk so "chunks" and "both" remain meaningful without extra work.
        if result['actual_mode'] != 'chunked' and output_mode in ('chunks', 'both'):
            manifest, chunk_totals = write_full_single_chunk(settings, vertices, faces, chunks_dir)
            metadata.update(chunk_manifest_path=str(chunks_dir/'chunks.json'),
                            chunk_count=manifest['total_chunks'],
                            chunk_vertices_total=chunk_totals['total_vertices'],
                            chunk_faces_total=chunk_totals['total_faces'])
            metadata['chunk_manifest'] = manifest
        elif result.get('chunk_manifest') is not None:
            metadata['chunk_manifest_path'] = str(chunks_dir/'chunks.json')
        if write_merged:
            event('SAVING_MESH')
            metadata.update(write_mesh(settings.output,vertices,faces))
            mesh_lo,mesh_hi=vertices.min(axis=0),vertices.max(axis=0)
            mesh_vertices,mesh_faces=len(vertices),len(faces)
        else:
            # chunks only: the fused mesh was intentionally not extracted or saved.
            union=chunk_totals.get('union_bounds') if chunk_totals else None
            if not union:
                raise WorkerError('MESH_INVALID','All chunk meshes were empty after core cropping')
            mesh_lo,mesh_hi=np.asarray(union[0]),np.asarray(union[1])
            mesh_vertices,mesh_faces=chunk_totals['total_vertices'],chunk_totals['total_faces']
            metadata.update(vertex_count=mesh_vertices,face_count=mesh_faces,
                            mesh_bbox=[mesh_lo.tolist(),mesh_hi.tolist()])
        lo,hi=points.min(axis=0),points.max(axis=0)
        difference=np.maximum(abs(mesh_lo-lo),abs(mesh_hi-hi))
        tolerance=max(1.,float(np.linalg.norm(hi-lo))*.25)
        metadata.update(input_path=str(settings.input) if settings.input else None,input_points=len(points), input_bbox=[lo.tolist(),hi.tolist()],
                        mesh_bbox=[mesh_lo.tolist(),mesh_hi.tolist()], bbox_difference=difference.tolist(),
                        validation_status='WARNING' if np.max(difference)>tolerance else 'PASS',
                        elapsed_seconds=time.monotonic()-started, checkpoint_loaded=True, sensor_origins_used=True)
        path=settings.metadata or settings.output.parent/'nksr_metadata.json'
        atomic_json(path,metadata)
        event('COMPLETED',vertices=mesh_vertices,faces=mesh_faces,
              mesh_output_mode=output_mode,chunk_count=metadata.get('chunk_count'))
        if settings.health_output:
            atomic_json(settings.health_output,dict(metadata, status='READY' if device.type=='cuda' else ('CPU_READY' if metadata['cuda_available'] else 'CUDA_UNAVAILABLE'),
                        cpu_ready=device.type=='cpu', smoke_passed=True, checked_at=time.time()))
    except Exception as error:
        code=classify(error,stage)
        if code=='EMPTY_TILE' and settings.tile_worker:
            path=settings.metadata or settings.output.parent/'nksr_metadata.json'
            atomic_json(path,dict(metadata,status='EMPTY_TILE',message=str(error)))
            event('SKIPPED',message=str(error))
            return 0
        if isinstance(error,WorkerError): metadata.update(error.details)
        traceback.print_exc()
        event('CANCELLED' if code=='CANCELLED' else 'FAILED',error_type=code,message=str(error))
        if settings.metadata: atomic_json(settings.metadata,dict(metadata,error_type=code,error=str(error)))
        if settings.health_output: atomic_json(settings.health_output,dict(metadata,status=code,smoke_passed=False,message=str(error),checked_at=time.time()))
        return 1
    return 0


if __name__ == '__main__': sys.exit(main())
