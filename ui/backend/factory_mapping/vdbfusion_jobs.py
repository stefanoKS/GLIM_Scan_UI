"""Subprocess-only VDBFusion orchestration.

The web/ROS backend never imports the native VDBFusion library, so a native memory
fault can only take down this managed subprocess. Unlike the NKSR worker, this
worker reads the raw ROS bag itself, so the ROS Python bindings and libraries must
stay reachable in its environment.
"""
import asyncio
import os
import re
import time
import uuid
from pathlib import Path
from . import engines
from .storage import atomic_json, now, read_json

MESH_OUTPUT_MODES = ('merged', 'chunks', 'both')
PROGRESS_FILE = 'vdbfusion_progress.json'
HEALTH_FILE = '.state/vdbfusion_health.json'
PREPARED_FILE = engines.PREPARED_MARKERS[engines.VDBFUSION]


def interpreter(root):
    configured = os.environ.get('VDBFUSION_PYTHON')
    saved = root/'.state/vdbfusion_python.txt'
    if not configured and saved.is_file():
        configured = saved.read_text().strip()
    return Path(configured or Path.home()/'.cache/factory-mapping/vdbfusion-env/bin/python').expanduser().absolute()


def worker_environment():
    env = os.environ.copy()
    # ROS paths are intentionally kept: the worker streams the raw bag itself.
    env.pop('PYTHONHOME', None)
    env['OMP_NUM_THREADS'] = env.get('OMP_NUM_THREADS', '4')
    return env


def worker_path():
    return Path(__file__).resolve().parents[3]/'tools/vdbfusion_worker.py'


def health(service):
    if service.root.joinpath('.state/deployment.json').is_file():
        deployment = read_json(service.root/'.state/deployment.json', {})
        if deployment.get('mode') == 'record_only':
            return dict(status='RECORD_ONLY',
                        message='VDBFusion is a workstation engine; export recordings and surface them there')
    python = interpreter(service.root)
    if not python.is_file():
        return dict(status='VDBFUSION_NOT_INSTALLED', message='Run scripts/setup_vdbfusion.sh', python=str(python))
    result = read_json(service.root/HEALTH_FILE, {})
    if (result.get('python') != str(python) or not result or
            time.time()-result.get('checked_at', 0) > 86400):
        result = dict(status='UNVERIFIED',
                      message='Run Check VDBFusion to integrate and extract a real native TSDF mesh',
                      python=str(python))
    if service.pm.active('vdbfusion_check'):
        result = {**result, 'status': 'CHECKING'}
    return result


async def check(service):
    """Run the real native smoke test in the isolated interpreter."""
    service.require_processing()
    python = interpreter(service.root)
    if not python.is_file():
        raise ValueError('VDBFUSION_NOT_INSTALLED: run scripts/setup_vdbfusion.sh')
    if service.capture.busy or service.active or any(service.pm.active(k) for k in
            ('vdbfusion', 'vdbfusion_check', 'nksr', 'nksr_check', 'reconstruction', 'offline', 'glim', 'tool')):
        raise ValueError('Wait for active capture or processing before checking VDBFusion')
    target = service.root/HEALTH_FILE
    atomic_json(target, dict(status='CHECKING', python=str(python), checked_at=time.time()))

    async def done(item):
        if item['state'] != 'completed' and read_json(target, {}).get('status') == 'CHECKING':
            atomic_json(target, dict(status='FAILED', python=str(python),
                                     message='VDBFusion check failed; inspect vdbfusion_check.log'))
    return await service.pm.start(
        'vdbfusion_check', [str(python), str(worker_path()), '--check', '--health-output', str(target)],
        service.root/'.state/vdbfusion_check.log', worker_environment(), done)


def get_run(service, sid, rid):
    if not re.fullmatch(r'run_[a-f0-9]{12}', rid):
        raise ValueError('Invalid reconstruction run')
    session = service.sessions.get(sid)
    run = session/'reconstruction'/rid
    if not run.is_dir() or any(p.is_symlink() for p in (run, run.parent)):
        raise ValueError('Reconstruction run not found')
    return run


def run_algorithm(service, sid, rid):
    """Engine of one run, resolved without requiring the run to exist.

    Requests for a run that has no readable metadata fall back to NKSR, which is the
    historical behavior for jobs created before algorithm selection existed.
    """
    if not re.fullmatch(r'run_[a-f0-9]{12}', rid):
        raise ValueError('Invalid reconstruction run')
    try:
        run = service.sessions.get(sid)/'reconstruction'/rid
    except Exception:  # noqa: BLE001 - an unknown session is reported by the engine path
        return engines.DEFAULT_ALGORITHM
    if not run.is_dir() or run.is_symlink():
        return engines.DEFAULT_ALGORITHM
    return engines.normalize_algorithm(read_json(run/'job.json', {}).get('algorithm'))


def prepared_record(run):
    return read_json(run/PREPARED_FILE, None)


def master_settings(request):
    """The subset of a mesh request that the worker needs, validated in-process."""
    if not isinstance(request, dict):
        raise ValueError('VDBFusion settings must be an object')
    return request


def prepare_arguments(root, run, bag, trajectory, topic, settings, edit_workspace=None):
    """Command for the lightweight VDBFusion preparation step (validation and preflight)."""
    args = [str(interpreter(root)), str(worker_path()), '--prepare',
            '--bag', str(bag), '--trajectory', str(trajectory), '--topic', topic,
            '--prepare-output', str(run/PREPARED_FILE)]
    args += settings_arguments(settings)
    if edit_workspace is not None:
        args += ['--edited-workspace', str(edit_workspace)]
    return args


def settings_arguments(settings):
    """Translate validated VDBFusion settings into worker arguments."""
    args = []
    if settings.get('preset') is not None:
        args += ['--preset', str(settings['preset'])]
    if settings.get('voxel_size_m') is not None:
        args += ['--voxel-size', repr(float(settings['voxel_size_m']))]
    if settings.get('sdf_trunc_m') is not None:
        args += ['--sdf-trunc', repr(float(settings['sdf_trunc_m']))]
    args += ['--space-carving', '1' if settings.get('space_carving') else '0']
    if settings.get('origin_error_budget_m') is not None:
        args += ['--origin-error-budget', repr(float(settings['origin_error_budget_m']))]
    if settings.get('batch_points') is not None:
        args += ['--batch-points', str(int(settings['batch_points']))]
    for key, flag in (('association_spacing_multiplier', '--association-spacing-multiplier'),
                      ('boundary_margin_m', '--boundary-margin'),
                      ('memory_budget_gib', '--memory-budget-gib')):
        if settings.get(key) is not None:
            args += [flag, repr(float(settings[key]))]
    if settings.get('unsupported_observations') is not None:
        args += ['--unsupported-policy', str(settings['unsupported_observations'])]
    if settings.get('mask_deleted_triangles') is not None:
        args += ['--mask-deleted-triangles', '1' if settings['mask_deleted_triangles'] else '0']
    for key, flag in (('roi_min_m', '--roi-min'), ('roi_max_m', '--roi-max')):
        if settings.get(key):
            args += [flag, *[repr(float(value)) for value in settings[key]]]
    return args


def validate_completed(output, returncode):
    """Validate the mesh a finished worker claims to have written."""
    from .nksr_mesh import inspect_mesh
    from .nksr_jobs import validate_chunks_completed
    if returncode != 0:
        raise ValueError('Worker exited unsuccessfully')
    metadata = read_json(output/'vdbfusion_metadata.json', {})
    if not metadata:
        raise ValueError('VDBFusion metadata is missing')
    mode = metadata.get('mesh_output_mode', 'merged')
    if mode not in MESH_OUTPUT_MODES:
        raise ValueError('Unknown mesh output mode in worker metadata')
    if metadata.get('engine') != engines.VDBFUSION:
        raise ValueError('Worker metadata does not identify the VDBFusion engine')
    if mode != 'chunks':
        stats = inspect_mesh(output/'mesh.ply')
        if stats['vertex_count'] != metadata.get('vertex_count') or stats['face_count'] != metadata.get('face_count'):
            raise ValueError('Mesh counts do not match worker metadata')
    if mode != 'merged':
        validate_chunks_completed(output, metadata)
    if metadata.get('validation_status') != 'PASS':
        raise ValueError('Worker did not report a passed mesh validation')
    return metadata


async def reconstruct(service, sid, rid, request):
    """Launch the managed native worker for one prepared VDBFusion run."""
    service.require_processing()
    from .reconstruction_jobs import saved_edit_source
    run = get_run(service, sid, rid)
    job = read_json(run/'job.json', {})
    if engines.normalize_algorithm(job.get('algorithm')) != engines.VDBFUSION:
        raise ValueError('This run was prepared for a different reconstruction algorithm')
    record = prepared_record(run)
    if not record or record.get('state') != 'PREPARED':
        raise ValueError('Prepare VDBFusion point input first')
    python = interpreter(service.root)
    if not python.is_file():
        raise ValueError('VDBFUSION_NOT_INSTALLED: run scripts/setup_vdbfusion.sh')
    if service.mock:
        raise ValueError('Real VDBFusion requires a real raw bag and trajectory')
    if service.capture.busy or service.active or any(service.pm.active(k) for k in
            ('vdbfusion', 'vdbfusion_check', 'nksr', 'nksr_check', 'reconstruction', 'offline', 'export',
             'tool', 'glim', 'recording')):
        raise ValueError('Finish capture and active processing first')
    # Re-verify the saved cleanup against the prepared record before a long native run.
    source = job.get('edited_geometry_source')
    if job.get('filter_edited_geometry'):
        source = source or {}
        current = saved_edit_source(service, service.sessions.get(sid), source.get('edit_id'), source.get('tolerance_m'))
        if any(current.get(key) != source.get(key)
               for key in ('saved_map_fingerprint', 'trajectory_fingerprint', 'export_path')):
            raise ValueError('Prepared input is stale because the saved cleanup source changed; prepare again')
        # GATE 7E: an edited run is blocked unless the cleanup geometry was validated.
        geometry = record.get('edited_geometry') or {}
        if record.get('edit_filter_accuracy') != 'validated_approximate' or not geometry.get('removed_points'):
            raise ValueError(
                'Edited-geometry filtering cannot be validated for this saved cleanup, so the VDBFusion job is '
                'blocked. Re-export the cleanup, or run without saved edited geometry.')
    output = run/'output'
    input_marker = run/PREPARED_FILE
    if input_marker.is_symlink() or output.is_symlink():
        raise ValueError('Invalid reconstruction path')
    # Preserve successful and failed prior attempts. Never overwrite a validated mesh.
    archive = run/'attempts'/('previous_'+uuid.uuid4().hex[:12])
    for name in ('output', 'mesh_job.json', PROGRESS_FILE):
        path = run/name
        if path.exists():
            archive.mkdir(parents=True, exist_ok=True)
            path.rename(archive/name)
    data = dict(state='RUNNING', started_at=now(), algorithm=engines.VDBFUSION, engine=engines.VDBFUSION,
                settings=request, python=str(python))
    atomic_json(run/'mesh_job.json', data)
    settings = settings_arguments(request)
    settings += ['--mesh-output-mode', str(request.get('mesh_output_mode', 'merged'))]
    if request.get('chunk_size') is not None:
        settings += ['--chunk-size', repr(float(request['chunk_size']))]
    args = [str(python), str(worker_path()), '--bag', record['bag'], '--trajectory', record['trajectory'],
            '--topic', record.get('topic', '/livox/lidar'), '--output', str(output/'mesh.ply'),
            '--metadata', str(output/'vdbfusion_metadata.json'), '--progress', str(run/PROGRESS_FILE)]
    args += settings
    if job.get('filter_edited_geometry'):
        args += ['--edited-workspace', str(service.edit_workspace(sid, source['edit_id']))]

    async def done(item):
        progress = read_json(run/PROGRESS_FILE, {})
        data.update(ended_at=now(), returncode=item['returncode'])
        if item['state'] == 'cancelled' or progress.get('error_type') == 'CANCELLED':
            # The worker reports CANCELLED whenever it received an interrupt, so an
            # operator-initiated stop is never recorded as an engine failure.
            data.update(state='CANCELLED', error_type='CANCELLED', message='VDBFusion reconstruction cancelled')
        elif item['state'] == 'completed':
            try:
                data.update(metadata=await asyncio.to_thread(validate_completed, output, item['returncode']),
                            state='COMPLETED')
            except Exception as error:
                data.update(state='FAILED', error_type='MESH_INVALID', message=str(error))
        else:
            data.update(state='FAILED', error_type=progress.get('error_type', 'VDBFUSION_RECONSTRUCTION_FAILED'),
                        message=progress.get('message', 'VDBFusion worker failed; inspect job.log'))
        atomic_json(run/'mesh_job.json', data)
    try:
        await service.pm.start('vdbfusion', args, run/'job.log', worker_environment(), done)
    except Exception as error:
        data.update(state='FAILED', error_type='VDBFUSION_NOT_INSTALLED', message=str(error))
        atomic_json(run/'mesh_job.json', data)
        raise
    return data


async def cancel(service, sid, rid):
    """Stop the managed worker and return serializable mesh state, never the raw process item."""
    from .nksr_jobs import mesh_state
    run = get_run(service, sid, rid)
    item = service.pm.items.get('vdbfusion', {})
    if item and Path(item['log']).parent != run:
        raise ValueError('Another reconstruction owns the worker')
    await service.pm.stop('vdbfusion', 60, cancel=True)
    return mesh_state(run, engines.VDBFUSION)
