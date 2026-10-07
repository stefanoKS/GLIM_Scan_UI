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


def execute(points, sensors, settings, event, torch, nksr, device, reconstructor):
    free = torch.cuda.mem_get_info(device)[0] if device.type == 'cuda' else None
    mode = settings.mode
    if mode == 'auto': mode = 'full' if len(points) <= 250_000 and (free is None or free >= 3*1024**3) else 'chunked'
    chunk_size = settings.chunk_size or (select_chunk(points) if mode == 'chunked' or settings.mode == 'auto' else None)
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
            def array(value): return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)
            vertices = array(mesh.v)
            if current_mode == 'chunked': vertices = vertices / scale
            return vertices, array(mesh.f), reconstruct_seconds, time.monotonic()-started

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
        mode = 'chunked'; chunk_size = min(chunk_size or 5., 5.)
        reconstructor.network.to(device)
    return vertices, faces, dict(requested_mode=settings.mode, actual_mode=mode,
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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path); p.add_argument('--output', type=Path)
    p.add_argument('--progress', type=Path); p.add_argument('--metadata', type=Path)
    p.add_argument('--check', action='store_true', help='Real pretrained model + inference + mesh smoke test')
    p.add_argument('--health-output', type=Path)
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--mode', choices=['auto','full','chunked','low_ram'], default='auto')
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
        event('SAVING_MESH')
        metadata.update(write_mesh(settings.output,vertices,faces))
        lo,hi=points.min(axis=0),points.max(axis=0)
        mesh_lo,mesh_hi=vertices.min(axis=0),vertices.max(axis=0)
        difference=np.maximum(abs(mesh_lo-lo),abs(mesh_hi-hi))
        tolerance=max(1.,float(np.linalg.norm(hi-lo))*.25)
        metadata.update(input_path=str(settings.input) if settings.input else None,input_points=len(points), input_bbox=[lo.tolist(),hi.tolist()],
                        mesh_bbox=[mesh_lo.tolist(),mesh_hi.tolist()], bbox_difference=difference.tolist(),
                        validation_status='WARNING' if np.max(difference)>tolerance else 'PASS',
                        elapsed_seconds=time.monotonic()-started, checkpoint_loaded=True, sensor_origins_used=True)
        path=settings.metadata or settings.output.parent/'nksr_metadata.json'
        atomic_json(path,metadata)
        event('COMPLETED',vertices=len(vertices),faces=len(faces))
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
