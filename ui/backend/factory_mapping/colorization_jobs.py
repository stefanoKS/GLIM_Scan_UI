"""Session-scoped colorization jobs launched through the managed subprocess."""
import sys
import uuid
from pathlib import Path

from .storage import atomic_json, read_json, now

SCRIPT = Path(__file__).resolve().parents[3] / 'tools' / 'glim_colorize.py'

# CLI flags passed through from job settings (snake_case -> dash-case).
BOOLEAN_FLAGS = ('transfer_glim', 'transfer_nksr', 'depth_edge_rejection')
VALUE_FLAGS = ('voxel_size', 'max_time_delta', 'min_depth', 'max_depth',
               'occlusion_base_tolerance', 'occlusion_range_scale',
               'validation_frames', 'chunk_points',
               'max_color_observations_per_voxel', 'depth_edge_radius', 'depth_edge_threshold',
               'transfer_radius', 'transfer_k',
               'surface_transfer_radius', 'surface_transfer_k')


def job_state(run, job):
    if job.get('state') in ('COMPLETED', 'completed'):
        return 'COMPLETED'
    if job.get('state') in ('CANCELLED', 'cancelled'):
        return 'CANCELLED'
    if job.get('state') in ('failed', 'interrupted'):
        return {'failed': 'FAILED', 'interrupted': 'INTERRUPTED'}[job['state']]
    return 'RUNNING' if job.get('state') == 'running' else 'NOT_RUNNING'


def view(service, sid):
    session = service.sessions.get(sid)
    jobs = []
    for run in sorted((session / 'colorization').glob('run_*/job.json')):
        job = read_json(run, {})
        job['progress'] = read_json(run.parent / 'progress.json', {}).get('message', '')
        job['metadata'] = read_json(run.parent / 'metadata.json', {})
        state = job_state(run.parent, job)
        if state == 'RUNNING' and not service.pm.active('colorization'):
            state = 'INTERRUPTED'
        job['state'] = state
        jobs.append(job)
    jobs.sort(key=lambda job: job.get('created_at', ''))
    return dict(jobs=jobs, raw_bag=(session / 'raw_bag/metadata.yaml').is_file(),
                camera_enabled=bool(read_json(session / 'active_config.json', {}).get('system', {}).get('camera', {}).get('enabled')))


async def start(service, sid, settings):
    service.require_processing()
    session = service.sessions.get(sid)
    if service.mock:
        raise ValueError('Colorization requires a real recorded session')
    if service.capture.busy or service.active or any(service.pm.active(key) for key in
            ('recording', 'glim', 'offline', 'export', 'tool', 'calibration_record',
             'calibration_tool', 'reconstruction', 'nksr', 'nksr_check', 'vdbfusion', 'vdbfusion_check', 'colorization')):
        raise ValueError('Finish capture and processing before colorizing')
    if not (session / 'raw_bag/metadata.yaml').is_file():
        raise ValueError('Raw bag metadata is missing')
    config = read_json(session / 'active_config.json', {})
    if not config.get('system', {}).get('camera', {}).get('enabled'):
        raise ValueError('Session did not record RGB; colorization requires recorded camera images')
    run = session / 'colorization' / ('run_' + uuid.uuid4().hex[:12])
    run.mkdir(parents=True)
    job = dict(id=run.name, state='running', created_at=now(),
               settings={k: v for k, v in settings.items() if k != 'allow_unvalidated_calibration'},
               allow_unvalidated_calibration=bool(settings.get('allow_unvalidated_calibration')))
    atomic_json(run / 'job.json', job)
    args = [sys.executable, str(SCRIPT), '--session', str(session),
            '--output-dir', str(run), '--progress-json', str(run / 'progress.json')]
    if settings.get('allow_unvalidated_calibration'):
        args.append('--allow-unvalidated-calibration')
    for key in BOOLEAN_FLAGS:
        if settings.get(key):
            args.append('--' + key.replace('_', '-'))
    for key in VALUE_FLAGS:
        value = settings.get(key)
        if value is not None:
            args += ['--' + key.replace('_', '-'), str(value)]
    async def done(item):
        job.update(ended_at=now(), returncode=item['returncode'])
        metadata = read_json(run / 'metadata.json', {})
        stats = metadata.get('statistics', {})
        outputs = [str((run / 'output' / name).relative_to(session))
                   for name in ('colored_points.ply', 'colored_points.npz')]
        if item['state'] == 'cancelled':
            job.update(state='CANCELLED')
        elif item['state'] == 'completed':
            if (run / 'output' / 'colored_points.ply').is_file() and (run / 'output' / 'colored_points.npz').is_file():
                job.update(state='COMPLETED', percentage_colored=stats.get('percentage_colored'),
                           colored_point_count=stats.get('colored_point_count'),
                           final_point_count=stats.get('final_point_count'), outputs=outputs)
            else:
                job.update(state='failed', error='Colorization finished without the expected outputs')
        else:
            job.update(state='failed', error=item.get('error') or 'Colorization worker failed; inspect job.log')
        atomic_json(run / 'job.json', job)
    try:
        from .commands import ros_env
        env = ros_env(service.config, offline=True)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + ':' + env.get('PYTHONPATH', '')
        await service.pm.start('colorization', args, run / 'job.log', env, done)
    except Exception as error:
        job.update(state='failed', error=str(error))
        atomic_json(run / 'job.json', job)
        raise
    return job


async def cancel(service, sid, cid):
    session = service.sessions.get(sid)
    run = session / 'colorization' / cid
    if not run.is_dir() or run.is_symlink():
        raise ValueError('Colorization run not found')
    item = service.pm.items.get('colorization', {})
    if item and Path(item.get('log', '')).parent != run:
        raise ValueError('Another colorization run owns the active worker')
    await service.pm.stop('colorization', 20, cancel=True)
    job = read_json(run / 'job.json', {})
    job.update(state='CANCELLED', ended_at=now())
    atomic_json(run / 'job.json', job)
    return job
