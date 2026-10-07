"""Session-scoped preparation jobs, independent of GLIM export and NKSR installation."""
import sys
import uuid
from pathlib import Path
from .storage import read_json, atomic_json, now
from .reconstruction import validate_filter_tolerance, validate_voxel_size
from . import glim_tools


def safe_trajectory(session, relative):
    path = session/relative
    if (Path(relative).is_absolute() or '..' in Path(relative).parts or
            not Path(relative).parts or Path(relative).parts[0] not in ('processing', 'edits') or
            path.name != 'traj_lidar.txt' or not path.is_file() or
            session.resolve() not in path.resolve().parents or
            any(p.is_symlink() for p in (path, *path.parents) if p != session.parent)):
        raise ValueError('Select a valid trajectory from this session')
    return path


def saved_edit_source(service, session, edit_id, tolerance_m):
    if not edit_id or tolerance_m is None:
        raise ValueError('Select a saved cleanup export and a positive tolerance')
    tolerance_m = validate_filter_tolerance(tolerance_m)
    workspace = service.edit_workspace(session.name, edit_id)
    metadata = read_json(workspace/'workspace.json', {})
    sources = metadata.get('sources', [])
    if metadata.get('tool') != 'map_editor' or len(sources) != 1 or sources[0].get('session') != session.name:
        raise ValueError('Filtering supports only a verified single-session map-editor cleanup from this session')
    if metadata.get('pose_policy') != 'map_editor_fixed_poses':
        raise ValueError('Saved edit does not prove fixed trajectory poses; create a new map-editor cleanup')
    dump = workspace/'saved_map'
    if dump.is_symlink() or (workspace/'map_01').is_symlink():
        raise ValueError('Saved cleanup sources cannot contain symlinks')
    glim_tools.validate_dump(dump)
    saved_trajectory = dump/'traj_lidar.txt'
    original_trajectory = workspace/'map_01/traj_lidar.txt'
    if not saved_trajectory.is_file() or saved_trajectory.is_symlink() or not original_trajectory.is_file() or original_trajectory.is_symlink():
        raise ValueError('Saved cleanup must include its original and saved trajectories')
    if glim_tools.file_fingerprint(saved_trajectory) != glim_tools.file_fingerprint(original_trajectory):
        raise ValueError('Saved-map trajectory differs from the cleanup source; do not align frames with ICP. Re-save a fixed-pose map-editor cleanup.')
    exported = read_json(workspace/'export.json', {})
    relative_export = exported.get('export_path')
    if (exported.get('state') != 'completed' or exported.get('edit_id') != edit_id or
            exported.get('source_session') != session.name or not isinstance(relative_export, str)):
        raise ValueError('Export the explicitly saved cleanup before preparing filtered NKSR input')
    export = session/relative_export
    if (Path(relative_export).is_absolute() or '..' in Path(relative_export).parts or
            not export.is_file() or export.is_symlink() or session.resolve() not in export.resolve().parents):
        raise ValueError('Saved cleanup export provenance is invalid')
    current_map = glim_tools.fingerprint(dump)
    current_trajectory = glim_tools.file_fingerprint(saved_trajectory)
    if exported.get('saved_map_fingerprint') != current_map or exported.get('trajectory_fingerprint') != current_trajectory:
        raise ValueError('Saved cleanup changed after export; export it again before preparing filtered NKSR input')
    return dict(edit_id=edit_id, export=export, trajectory=saved_trajectory, tolerance_m=tolerance_m,
                saved_map_fingerprint=current_map, trajectory_fingerprint=current_trajectory,
                export_path=relative_export)


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
    edits=[]
    for workspace in sorted((session/'edits').glob('edit_*')):
        if workspace.is_dir() and not workspace.is_symlink():
            metadata=read_json(workspace/'workspace.json',{})
            exported=read_json(workspace/'export.json',{})
            if metadata.get('tool')=='map_editor' and len(metadata.get('sources',[]))==1:
                edits.append(dict(id=workspace.name, state=metadata.get('state'), export_ready=exported.get('state')=='completed'))
    return dict(trajectories=trajectories, edited_sources=edits, jobs=jobs, nksr=health(service), raw_bag=(session/'raw_bag/metadata.yaml').is_file())


async def start(service, sid, trajectory, voxel_size_m, save_full_density=False,
                filter_edited_geometry=False, edit_id=None, filter_tolerance_m=None):
    service.require_processing()
    size = validate_voxel_size(voxel_size_m)
    session = service.sessions.get(sid)
    source = safe_trajectory(session, trajectory)
    edit_source = None
    if filter_edited_geometry:
        edit_source = saved_edit_source(service, session, edit_id, filter_tolerance_m)
        source = edit_source['trajectory']
        trajectory = str(source.relative_to(session))
    elif edit_id is not None or filter_tolerance_m is not None:
        raise ValueError('Saved edited geometry settings require filtering to be enabled')
    if service.mock:
        raise ValueError('Reconstruction preparation requires a real raw bag and trajectory')
    if service.capture.busy or service.active or any(service.pm.active(k) for k in
            ('recording', 'glim', 'offline', 'export', 'tool', 'calibration_record', 'calibration_tool', 'reconstruction', 'nksr', 'nksr_check')):
        raise ValueError('Finish capture and processing before preparing reconstruction')
    if not (session/'raw_bag/metadata.yaml').is_file():
        raise ValueError('Raw bag metadata is missing')
    run = session/'reconstruction'/('run_'+uuid.uuid4().hex[:12])
    run.mkdir(parents=True)
    job = dict(id=run.name, state='running', created_at=now(), trajectory=trajectory,
               voxel_size_m=size, save_full_density=save_full_density,
               filter_edited_geometry=filter_edited_geometry,
               edited_geometry_source={key:value for key,value in edit_source.items() if key not in ('export','trajectory')} if edit_source else None)
    atomic_json(run/'job.json', job)
    config = read_json(session/'active_config.json', service.config)
    args = [sys.executable, '-u', '-m', 'factory_mapping.reconstruction',
            '--bag', str(session/'raw_bag'), '--trajectory', str(source),
            '--output-dir', str(run), '--voxel-size', str(size),
            '--topic', config['sensor']['points_topic']]
    if save_full_density:
        args.append('--save-full-density')
    if edit_source:
        provenance = dict(filter_enabled=True, edited_geometry_source=job['edited_geometry_source'],
                          trajectory=trajectory, filter_description='Approximate retained-geometry filtering; NKSR can still bridge deleted regions.')
        provenance_path=run/'provenance.json'; atomic_json(provenance_path, provenance)
        args += ['--edited-export', str(edit_source['export']), '--filter-tolerance', str(edit_source['tolerance_m']),
                 '--provenance-json', str(provenance_path)]
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
