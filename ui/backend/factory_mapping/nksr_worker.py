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
from .mesh_partition import partition_mesh
from .nksr_mesh import write_mesh

DEFAULT_NKSR_TARGET_VOXEL_M = 0.02
# Native ks voxel at pinned NKSR e403368; independent of preparation sampling.
NKSR_NATIVE_VOXEL_SIZE = 0.1
MESH_OUTPUT_MODES = ('merged', 'chunks', 'both')
# Shared physical default when no explicit chunk size is requested and the density heuristic does not apply.
DEFAULT_CHUNK_SIZE_M = 5.0
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
    if stage in ('SAVING_MESH', 'PARTITIONING_MESH', 'SAVING_MESH_CHUNKS'): return 'MESH_INVALID'
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


def derived_stride(centers):
    """Fallback stride from the surviving centers, only if the NKSR grid disagrees."""
    steps = [np.diff(np.unique(np.asarray(centers, dtype=np.float64)[:, axis])) for axis in range(3)]
    steps = [step[step > 0].min() for step in steps if step.size]
    return float(min(steps)) if steps else 1.0


def chunk_grid_layout(centers, chunk_size_scaled, overlap_ratio):
    """Relative NKSR grid indices that preserve skipped cells, plus the stride used.

    NKSR lays chunk centers on the regular grid
    ``stride = chunk_size * (1 - overlap_ratio)`` and skips candidate cells with
    no points, so ``field.transforms`` is not a complete Cartesian grid. Relative
    indices come from the nominal stride and the smallest surviving center per
    axis, so holes are preserved (``[0, 38] -> [0, 2]`` for a 19 m scaled stride)
    instead of being rank-compressed to ``[0, 1]``.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or not len(centers):
        raise WorkerError('EMPTY_TILE', 'NKSR produced no chunk fields')
    stride = float(chunk_size_scaled) * (1.0 - overlap_ratio) if chunk_size_scaled else None
    source = 'nksr_chunk_size'
    if not (stride and np.isfinite(stride) and stride > 0):
        stride, source = derived_stride(centers), 'derived_from_centers'
    origin = centers.min(axis=0)
    relative = np.rint((centers - origin) / stride).astype(np.int64)
    # A mismatched nominal stride would silently mis-index the grid; only then fall back.
    if source == 'nksr_chunk_size' and np.abs(centers - (origin + relative * stride)).max() > max(1e-3 * stride, 1e-6):
        stride, source = derived_stride(centers), 'derived_from_centers'
        relative = np.rint((centers - origin) / stride).astype(np.int64)
    return stride, relative, source


def chunk_core_bounds(centers, relative_index, stride_scaled):
    """Nominal core ownership box per chunk, in scaled coordinates.

    A side is bounded by the midpoint to a genuinely adjacent active grid cell:
    relative index difference of one along that axis and at most one on the
    other axes. Cells elsewhere in the scene never bound a chunk, so sparse
    active grids are not split against unrelated chunks. Unbounded sides stay
    +/-inf. Triangle ownership itself is decided by :func:`owned_triangle_mask`;
    this box is the nominal core cell for reporting.
    """
    centers = np.asarray(centers, dtype=np.float64)
    relative_index = np.asarray(relative_index, dtype=np.int64)
    bounds = []
    for index in range(len(centers)):
        lo = np.full(3, -np.inf)
        hi = np.full(3, np.inf)
        for axis in range(3):
            others = [other for other in range(3) if other != axis]
            for other in range(len(centers)):
                if other == index:
                    continue
                if any(abs(relative_index[other][a] - relative_index[index][a]) > 1 for a in others):
                    continue
                step = int(relative_index[other][axis] - relative_index[index][axis])
                if step == -1:
                    lo[axis] = max(lo[axis], 0.5 * (centers[index][axis] + centers[other][axis]))
                elif step == 1:
                    hi[axis] = min(hi[axis], 0.5 * (centers[index][axis] + centers[other][axis]))
        bounds.append((lo, hi))
    return bounds


OWNERSHIP_BLOCK_ELEMENTS = 2_000_000


def owned_triangle_mask(centroids, centers, chunk_size_scaled, index):
    """Mask of the triangles the emitting chunk ``index`` keeps, in scaled coordinates.

    A triangle belongs to the active chunk whose nominal cube contains its
    centroid, choosing the nearest chunk center and, on a tie (for example a
    diagonal overlap), the lowest field index. When no active cube contains the
    centroid, the emitting chunk keeps the triangle: NKSR skips empty cells, so
    ``field.transforms`` is not a complete grid and a chunk elsewhere in the
    scene must never crop geometry that no cube claims. Working on centroids
    keeps ownership deterministic, so an overlap region is never exported twice.
    """
    centers = np.asarray(centers, dtype=np.float64)
    points = np.asarray(centroids, dtype=np.float64)
    fields = len(centers)
    if not fields:
        raise WorkerError('EMPTY_TILE', 'NKSR produced no chunk fields')
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    half = 0.5 * float(chunk_size_scaled) if chunk_size_scaled else np.inf
    keep = np.zeros(len(points), dtype=bool)
    block = max(1, min(len(points), OWNERSHIP_BLOCK_ELEMENTS // fields))
    for start in range(0, len(points), block):
        stop = min(start + block, len(points))
        local = points[start:stop]
        distance = np.zeros((stop - start, fields), dtype=np.float64)
        inside = np.ones((stop - start, fields), dtype=bool)
        for axis in range(3):
            delta = local[:, axis][:, None] - centers[:, axis][None, :]
            distance += delta * delta
            if np.isfinite(half):
                inside &= np.abs(delta) <= half
        covered = inside.any(axis=1)
        distance = np.where(inside | ~covered[:, None], distance, np.inf)
        block_keep = np.argmin(distance, axis=1) == index
        keep[start:stop] = block_keep | ~covered
    return keep


def compact_mesh(vertices, faces):
    """Drop vertices unused by the surviving triangles and remap the faces."""
    faces = np.asarray(faces)
    if faces.size == 0:
        return np.asarray(vertices)[:0].copy(), faces.reshape(0, 3)
    used = np.zeros(len(vertices), dtype=bool)
    used[faces.reshape(-1)] = True
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    return np.asarray(vertices)[used].copy(), remap[faces]


def crop_mesh_to_owned(vertices, faces, index, centers, chunk_size_scaled):
    """Keep only the triangles that ``index`` owns, then compact the mesh."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if faces.size == 0:
        return vertices[:0].copy(), faces.reshape(0, 3)
    keep = owned_triangle_mask(vertices[faces].mean(axis=1), centers, chunk_size_scaled, index)
    return compact_mesh(vertices, faces[keep])


def select_chunk(points):
    # Largest candidate with at most 250k observations per nonoverlapped cell.
    # Metric candidates are checked against actual density, not just the bounds.
    for size in (20., 10., 5.):
        _, counts = np.unique(np.floor(points/size).astype(np.int64), axis=0, return_counts=True)
        if counts.max() <= 250_000: return size
    return 5.


def resolve_chunk_size(requested_chunk_size_m, chunk_size_source, mode, points):
    """One physical chunk size in metres, shared by reconstruction and mesh export.

    An explicit request always wins. Otherwise Chunked keeps the existing NKSR density
    heuristic, and Full and Low RAM export the historical 5 m default. ``mode`` must
    already be resolved from ``auto`` so an unsuccessful retry reports the size it used.
    """
    if requested_chunk_size_m is not None:
        return float(requested_chunk_size_m), (chunk_size_source or 'user')
    if mode == 'chunked':
        return select_chunk(points), 'auto_density'
    return DEFAULT_CHUNK_SIZE_M, 'default'


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


def extract_and_save_chunks(fields, centers, rotations, settings, scale, chunk_size_m,
                            chunks_dir, event, torch, device):
    """Legacy native per-field exporter, retained for compatibility and advanced use only.

    It extracts each NKSR chunk field independently and crops it to its owned region,
    so neighbouring fields can differ at their boundaries. The standard ``chunks`` and
    ``both`` output selections do NOT use this: they extract the one final fused mesh
    and split it spatially with :mod:`.mesh_partition`, which preserves fused geometry.

    Chunk meshes are processed one at a time and never held simultaneously. Each
    field is moved to CPU before extraction (the existing small-memory recipe),
    transformed into scaled global coordinates, cropped to its owned triangles,
    compacted, and written in world meters.
    """
    chunk_size_scaled = chunk_size_m * scale
    stride_scaled, relative_index, stride_source = chunk_grid_layout(
        centers, chunk_size_scaled, settings.overlap_ratio)
    bounds = chunk_core_bounds(centers, relative_index, stride_scaled)
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
        core_vertices, core_faces = crop_mesh_to_owned(
            global_vertices, faces, index, centers, chunk_size_scaled)
        del local_vertices, global_vertices, faces
        core_min, core_max = bounds[index]
        entry = dict(index=index, grid_index=relative_index[index].tolist(),
                     field_origin_scaled=center.tolist(),
                     core_bbox_min=finite_or_none(core_min / scale),
                     core_bbox_max=finite_or_none(core_max / scale),
                     core_bbox_min_scaled=finite_or_none(core_min),
                     core_bbox_max_scaled=finite_or_none(core_max))
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
                    chunk_size_m=chunk_size_m, nksr_chunk_size_scaled=chunk_size_scaled,
                    chunk_stride_scaled=stride_scaled, chunk_stride_source=stride_source,
                    overlap_ratio=settings.overlap_ratio,
                    total_chunks=total, total_vertices=total_vertices, total_faces=total_faces,
                    ownership_rule='Triangle centroid belongs to the active chunk whose nominal cube '
                                   'contains it, nearest center first, lowest field index on a tie; a centroid '
                                   'inside no active cube stays with its emitting chunk. core_bbox_* is the '
                                   'nominal core cell from midpoints to genuinely adjacent active cells and is '
                                   'unbounded (null) where no adjacent cell exists.',
                    chunks=entries)
    atomic_json(chunks_dir / 'chunks.json', manifest)
    union = None if not np.isfinite(union_min).all() else (union_min.tolist(), union_max.tolist())
    return manifest, dict(total_vertices=total_vertices, total_faces=total_faces,
                          union_bounds=union)


def execute(points, sensors, settings, event, torch, nksr, device, reconstructor):
    free = torch.cuda.mem_get_info(device)[0] if device.type == 'cuda' else None
    output_mode = getattr(settings, 'mesh_output_mode', 'merged')
    mode = settings.mode
    if mode == 'auto': mode = 'full' if len(points) <= 250_000 and (free is None or free >= 3*1024**3) else 'chunked'
    requested_chunk_size_m = settings.chunk_size
    chunk_size, chunk_size_source = resolve_chunk_size(
        requested_chunk_size_m, getattr(settings, 'chunk_size_source', None), mode, points)
    attempts = []
    reconstructor.chunk_tmp_device = torch.device('cpu:0')
    preprocess = nksr.get_estimate_normal_preprocess_fn(settings.normal_knn, settings.normal_drop_angle_deg)
    if device.type=='cpu': preprocess=cpu_normal_preprocess(settings.normal_knn,settings.normal_drop_angle_deg)

    def attempt(current_mode):
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
              effective_chunk_size_m=chunk_size, chunk_size_source=chunk_size_source,
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
            # One final fused surface per attempt; every output mode partitions this mesh,
            # so a triangle is never extracted from an individual NKSR chunk field.
            # Chunked extraction produces scaled global coordinates: restore GLIM metres
            # here, so partitioning and export always work in real metres.
            if current_mode == 'chunked': vertices = vertices / scale
            return vertices, to_numpy(mesh.f), reconstruct_seconds, time.monotonic()-started

    for retry in range(2):
        try:
            vertices, faces, recons_seconds, extraction_seconds = attempt(mode)
            break
        except Exception as error:
            if classify(error, 'RECONSTRUCTING') != 'CUDA_OOM' or settings.mode != 'auto' or retry:
                raise
            attempts.append(dict(mode=mode, error='CUDA_OOM'))
            event('RECONSTRUCTING', message='CUDA OOM: releasing tensors; one chunked retry', actual_mode='chunked')
        # Outside except: traceback/tensors can now be collected.
        gc.collect(); torch.cuda.empty_cache()
        mode = 'chunked'
        if chunk_size > DEFAULT_CHUNK_SIZE_M:
            # Report the size the successful retry actually used.
            chunk_size, chunk_size_source = DEFAULT_CHUNK_SIZE_M, 'oom_retry'
        reconstructor.network.to(device)
    scale = NKSR_NATIVE_VOXEL_SIZE / DEFAULT_NKSR_TARGET_VOXEL_M if mode == 'chunked' else 1.0
    result = dict(requested_mode=settings.mode, actual_mode=mode, mesh_output_mode=output_mode,
        chunk_count=None,
        requested_chunk_size_m=requested_chunk_size_m, effective_chunk_size_m=chunk_size,
        chunk_size_source=chunk_size_source,
        chunk_size=chunk_size if mode == 'chunked' else None, overlap_ratio=settings.overlap_ratio,
        nksr_chunk_size_scaled=chunk_size * scale if mode == 'chunked' else None,
        requested_detail_level=settings.detail_level, detail_level=None,
        nksr_internal_voxel_size=DEFAULT_NKSR_TARGET_VOXEL_M if mode == 'full' else None,
        target_voxel_m=DEFAULT_NKSR_TARGET_VOXEL_M, coordinate_scale=scale,
        normal_knn=settings.normal_knn,
        normal_drop_angle_deg=settings.normal_drop_angle_deg, mise_iter=settings.mise_iter,
        normal_backend='nksr_cuda' if device.type=='cuda' else 'scipy_cpu_pca',
        approx_kernel_grad=True, fused_mode=True, solver_tol=1e-4 if mode == 'full' else 1e-5,
        solver_note='Upstream chunk dispatch uses default solver_tol=1e-5' if mode == 'chunked' else None,
        attempts=attempts, reconstruction_seconds=recons_seconds, extraction_seconds=extraction_seconds,
        gpu_free_before=free, gpu_free_after=torch.cuda.mem_get_info(device)[0] if device.type=='cuda' else None,
        gpu_total_memory=torch.cuda.mem_get_info(device)[1] if device.type=='cuda' else None,
        gpu_peak_allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        source_vertex_count=int(len(vertices)), source_face_count=int(len(faces)))
    return vertices, faces, result


def write_outputs(settings, vertices, faces, result, event):
    """Save the selected mesh output: the merged mesh, spatial export cells, or both.

    Full and fused-Chunked reconstruction both reach this with the one final surface, so
    ``chunks`` never extracts individual NKSR fields and ``both`` never reconstructs
    twice. ``chunks`` alone writes no merged mesh. Returns
    ``(fields, mesh_lo, mesh_hi, mesh_vertices, mesh_faces)``.
    """
    output_mode = settings.mesh_output_mode
    chunks_dir = settings.output.parent / CHUNKS_DIR_NAME
    fields = dict(mesh_output_mode=output_mode, output_bytes=0)
    chunk_totals = None
    if output_mode in ('chunks', 'both'):
        partition_started = time.monotonic()
        manifest, chunk_totals = partition_mesh(
            vertices, faces, result['effective_chunk_size_m'], chunks_dir,
            reconstruction_mode=result['actual_mode'],
            requested_chunk_size_m=result['requested_chunk_size_m'],
            chunk_size_source=result['chunk_size_source'], event=event)
        fields.update(chunk_manifest_path=f'{CHUNKS_DIR_NAME}/chunks.json',
                      export_strategy=manifest['export_strategy'],
                      chunk_count=chunk_totals['chunk_count'],
                      chunk_vertices_total=chunk_totals['total_vertices'],
                      chunk_faces_total=chunk_totals['total_faces'],
                      chunk_union_bounds=chunk_totals['union_bounds'],
                      chunk_size_m=manifest['effective_chunk_size_m'],
                      partition_seconds=time.monotonic()-partition_started)
        fields['output_bytes'] += sum(chunk['file_size_bytes'] for chunk in manifest['chunks'])
    if output_mode != 'chunks':
        event('SAVING_MESH')
        stats = write_mesh(settings.output, vertices, faces)
        fields.update(stats)
        fields['output_bytes'] += stats['mesh_file_size']
        return fields, vertices.min(axis=0), vertices.max(axis=0), len(vertices), len(faces)
    # chunks only: no merged mesh is written at all.
    union = chunk_totals['union_bounds']
    mesh_lo, mesh_hi = np.asarray(union[0]), np.asarray(union[1])
    fields.update(vertex_count=chunk_totals['total_vertices'], face_count=chunk_totals['total_faces'],
                  mesh_bbox=[mesh_lo.tolist(), mesh_hi.tolist()])
    return fields, mesh_lo, mesh_hi, chunk_totals['total_vertices'], chunk_totals['total_faces']


class _Parser(argparse.ArgumentParser):
    """Parser that folds the legacy Low RAM ``--tile-size`` edge into the shared chunk size."""

    def parse_args(self, args=None, namespace=None):
        settings = super().parse_args(args, namespace)
        if settings.chunk_size is None and settings.tile_size is not None:
            # Legacy clients sent the Low RAM tile edge instead of the shared chunk size.
            settings.chunk_size, settings.chunk_size_source = settings.tile_size, 'legacy_tile_size'
        settings.tile_size = None
        settings.requested_chunk_size_m = settings.chunk_size
        return settings


def parser():
    p = _Parser(description=__doc__)
    p.add_argument('--input', type=Path); p.add_argument('--output', type=Path)
    p.add_argument('--progress', type=Path); p.add_argument('--metadata', type=Path)
    p.add_argument('--check', action='store_true', help='Real pretrained model + inference + mesh smoke test')
    p.add_argument('--health-output', type=Path)
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--mode', choices=['auto','full','chunked','low_ram'], default='auto')
    p.add_argument('--mesh-output-mode', choices=list(MESH_OUTPUT_MODES), default='merged',
                   help='merged: fused mesh only; chunks: spatial mesh cells only; both: cells plus fused mesh')
    p.add_argument('--tile-size', type=float,
                   help='Legacy alias for the shared Low RAM tile edge; prefer --chunk-size')
    p.add_argument('--tile-worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--detail-level', type=float, default=.5)
    p.add_argument('--chunk-size', type=float,
                   help='One physical chunk size in metres for reconstruction partitioning and mesh export')
    p.add_argument('--chunk-size-source', choices=['user','legacy_tile_size'],
                   help='Where an explicit --chunk-size came from, for reporting only')
    p.add_argument('--overlap-ratio', type=float, default=.05)
    p.add_argument('--normal-knn', type=int, default=64)
    p.add_argument('--normal-drop-angle-deg', type=float, default=85.)
    p.add_argument('--mise-iter', type=int, default=1)
    return p


def main():
    settings = parser().parse_args()
    if not (0 <= settings.detail_level <= 1 and 0 <= settings.overlap_ratio < 1 and
            0 < settings.normal_drop_angle_deg <= 90 and settings.normal_knn > 0 and
            0 <= settings.mise_iter <= 4 and (not settings.tile_worker or settings.mode=='full') and
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
        fields,mesh_lo,mesh_hi,mesh_vertices,mesh_faces=write_outputs(settings,vertices,faces,result,event)
        metadata.update(fields)
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
              mesh_output_mode=settings.mesh_output_mode,chunk_count=metadata.get('chunk_count'))
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
