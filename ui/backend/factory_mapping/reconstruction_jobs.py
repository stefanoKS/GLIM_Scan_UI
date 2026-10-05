"""Session-scoped preparation jobs, independent of GLIM export and NKSR installation."""
import sys
import uuid
from pathlib import Path
from .storage import read_json, atomic_json, now
from .reconstruction import validate_voxel_size


def safe_trajectory(session, relative):
    path = session/relative
    if (Path(relative).is_absolute() or '..' in Path(relative).parts or
            not Path(relative).parts or Path(relative).parts[0] not in ('processing', 'edits') or
            path.name != 'traj_lidar.txt' or not path.is_file() or
            session.resolve() not in path.resolve().parents or
            any(p.is_symlink() for p in (path, *path.parents) if p != session.parent)):
        raise ValueError('Select a valid trajectory from this session')
    return path


def preparation_state(run, job):
    if job.get('state') in ('completed','PREPARED') and (run/'input/nksr_input.npz').is_file():
        return 'PREPARED'
    if job.get('state') in ('completed','PREPARED'): return 'NOT_PREPARED'
    return {'running':'PREPARING','failed':'FAILED','cancelled':'CANCELLED','interrupted':'INTERRUPTED'}.get(job.get('state'),job.get('state','NOT_PREPARED'))


def view(service, sid):
    session = service.sessions.get(sid)
    trajectories = []
    for folder in ('processing', 'edits'):
        for path in sorted((session/folder).glob('**/traj_lidar.txt')):
            relative = str(path.relative_to(session))
            try: safe_trajectory(session, relative)
            except ValueError: continue
            trajectories.append(relative)
    jobs = []
    for path in sorted((session/'reconstruction').glob('run_*/job.json')):
        job = read_json(path, {})
        job['progress'] = read_json(path.parent/'progress.json', {}).get('message', '')
        job['metadata'] = read_json(path.parent/'validation/comparison.json', {})
        if job.get('state') == 'running' and not service.pm.active('reconstruction'):
            job['state'] = 'interrupted'
        job['state']=preparation_state(path.parent,job)
        from .nksr_jobs import mesh_state
        job['mesh']=mesh_state(path.parent)
        if job['mesh']['state']=='RUNNING':
            worker=service.pm.items.get('nksr',{})
            if not service.pm.active('nksr') or str(path.parent/'job.log')!=worker.get('log'):
                job['mesh'].update(state='FAILED',stage='FAILED',message='Worker interrupted; inspect logs')
            elif worker.get('state')=='orphaned':
                job['mesh'].update(stage='INTERRUPTED',message='Previous worker is still alive; inspect process recovery before restarting')
        jobs.append(job)
    jobs.sort(key=lambda job: job.get('created_at', ''))
    from .nksr_jobs import health
    return dict(trajectories=trajectories, jobs=jobs, nksr=health(service), raw_bag=(session/'raw_bag/metadata.yaml').is_file())


async def start(service, sid, trajectory, voxel_size_m, save_full_density=False):
    service.require_processing()
    size = validate_voxel_size(voxel_size_m)
    session = service.sessions.get(sid)
    source = safe_trajectory(session, trajectory)
    if service.mock:
        raise ValueError('Reconstruction preparation requires a real raw bag and trajectory')
    if service.capture.busy or service.active or any(service.pm.active(k) for k in
            ('recording', 'glim', 'offline', 'tool', 'calibration_record', 'calibration_tool', 'reconstruction', 'nksr', 'nksr_check')):
        raise ValueError('Finish capture and processing before preparing reconstruction')
    if not (session/'raw_bag/metadata.yaml').is_file():
        raise ValueError('Raw bag metadata is missing')
    run = session/'reconstruction'/('run_'+uuid.uuid4().hex[:12])
    run.mkdir(parents=True)
    job = dict(id=run.name, state='running', created_at=now(), trajectory=trajectory,
               voxel_size_m=size, save_full_density=save_full_density)
    atomic_json(run/'job.json', job)
    config = read_json(session/'active_config.json', service.config)
    args = [sys.executable, '-u', '-m', 'factory_mapping.reconstruction',
            '--bag', str(session/'raw_bag'), '--trajectory', str(source),
            '--output-dir', str(run), '--voxel-size', str(size),
            '--topic', config['sensor']['points_topic']]
    if save_full_density:
        args.append('--save-full-density')
    async def done(item):
        job.update(state='PREPARED' if item['state']=='completed' and (run/'input/nksr_input.npz').is_file() else ('failed' if item['state']=='completed' else item['state']), ended_at=now(), returncode=item['returncode'])
        atomic_json(run/'job.json', job)
    try:
        # The managed subprocess supports recovery, shutdown and log capture.
        from .commands import ros_env
        env = ros_env(service.config, offline=True)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + ':' + env.get('PYTHONPATH', '')
        await service.pm.start('reconstruction', args, run/'job.log', env, done)
    except Exception as error:
        job.update(state='failed', error=str(error))
        atomic_json(run/'job.json', job)
        raise
    return job
