"""Session-scoped preparation jobs, independent of GLIM export and NKSR installation."""
import os
import sys
import uuid
from pathlib import Path
from .storage import read_json, atomic_json, now
from .reconstruction import validate_filter_tolerance, validate_voxel_size
from . import engines, glim_tools


def safe_trajectory(session, relative):
    path = session/relative
    if (Path(relative).is_absolute() or '..' in Path(relative).parts or
            not Path(relative).parts or Path(relative).parts[0] not in ('processing', 'edits') or
            path.name != 'traj_lidar.txt' or not path.is_file() or
            session.resolve() not in path.resolve().parents or
            any(p.is_symlink() for p in (path, *path.parents) if p != session.parent)):
        raise ValueError('Select a valid trajectory from this session')
    return path


def saved_edit_source(service, session, edit_id, tolerance_m=None, require_tolerance=False):
    """Verify one saved map-editor cleanup and return its immutable lineage.

    ``tolerance_m`` is the NKSR proximity tolerance. VDBFusion derives its own
    association radius from the measured cleanup geometry, so it does not need one.
    """
    if not edit_id:
        raise ValueError('Select a saved cleanup export')
    if require_tolerance and tolerance_m is None:
        raise ValueError('Select a saved cleanup export and a positive tolerance')
    tolerance_m = validate_filter_tolerance(tolerance_m) if tolerance_m is not None else None
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
    algorithm = engines.normalize_algorithm(job.get('algorithm'))
    marker = run/engines.PREPARED_MARKERS[algorithm]
    if job.get('state') in ('completed', 'PREPARED') and marker.is_file():
        return 'PREPARED'
    if job.get('state') in ('completed', 'PREPARED'):
        return 'NOT_PREPARED'
    return {'running': 'PREPARING', 'failed': 'FAILED', 'cancelled': 'CANCELLED',
            'interrupted': 'INTERRUPTED'}.get(job.get('state'), job.get('state', 'NOT_PREPARED'))


def prepared_metadata(run, algorithm):
    """Prepared-input summary for one engine, in the shape the UI renders."""
    if algorithm == engines.VDBFUSION:
        record = read_json(run/engines.PREPARED_MARKERS[engines.VDBFUSION], {})
        if not record:
            return {}
        scan = record.get('bag_scan') or {}
        return dict(algorithm=engines.VDBFUSION, preparation='source_validation_only',
                    points_before_voxel=scan.get('estimated_observations'),
                    observations_are_estimate=True, lidar_frames=scan.get('lidar_frames'),
                    observed_bbox=[scan.get('observed_bbox_min_m'), scan.get('observed_bbox_max_m')],
                    settings=record.get('settings'), preflight=record.get('preflight'),
                    edit_filter_accuracy=record.get('edit_filter_accuracy'),
                    edited_geometry=record.get('edited_geometry'))
    metadata = read_json(run/'validation/comparison.json', {})
    if metadata:
        metadata['algorithm'] = engines.NKSR
    return metadata


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
        algorithm = engines.normalize_algorithm(job.get('algorithm'))
        job['algorithm'] = algorithm
        job['progress'] = read_json(path.parent/'progress.json', {}).get('message', '')
        if algorithm == engines.VDBFUSION:
            job['progress'] = read_json(path.parent/'vdbfusion_progress.json', {}).get('message', job['progress'])
        job['metadata'] = prepared_metadata(path.parent, algorithm)
        if job.get('state') == 'running' and not service.pm.active('reconstruction'):
            job['state'] = 'interrupted'
        job['state']=preparation_state(path.parent,job)
        from .nksr_jobs import mesh_state
        job['mesh']=mesh_state(path.parent, algorithm)
        key = engines.process_key(algorithm)
        if job['mesh']['state']=='RUNNING':
            worker=service.pm.items.get(key,{})
            if not service.pm.active(key) or str(path.parent/'job.log')!=worker.get('log'):
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
    return dict(trajectories=trajectories, edited_sources=edits, jobs=jobs, nksr=health(service),
                vdbfusion=vdbfusion_health(service), raw_bag=(session/'raw_bag/metadata.yaml').is_file())


def vdbfusion_health(service):
    from .vdbfusion_jobs import health
    return health(service)


async def start(service, sid, trajectory, voxel_size_m, save_full_density=False,
                filter_edited_geometry=False, edit_id=None, filter_tolerance_m=None,
                algorithm=None, vdbfusion=None):
    service.require_processing()
    engine = engines.normalize_algorithm(algorithm)
    size = validate_voxel_size(voxel_size_m)
    session = service.sessions.get(sid)
    source = safe_trajectory(session, trajectory)
    edit_source = None
    if filter_edited_geometry:
        # NKSR measures proximity at the requested tolerance; VDBFusion derives its
        # association radius from the measured cleanup geometry instead.
        edit_source = saved_edit_source(service, session, edit_id, filter_tolerance_m,
                                       require_tolerance=engine == engines.NKSR)
        source = edit_source['trajectory']
        trajectory = str(source.relative_to(session))
    elif edit_id is not None or filter_tolerance_m is not None:
        raise ValueError('Saved edited geometry settings require filtering to be enabled')
    if service.mock:
        raise ValueError('Reconstruction preparation requires a real raw bag and trajectory')
    if engine == engines.VDBFUSION:
        from .vdbfusion_jobs import interpreter
        if not interpreter(service.root).is_file():
            raise ValueError('VDBFUSION_NOT_INSTALLED: run scripts/setup_vdbfusion.sh')
    if service.capture.busy or service.active or any(service.pm.active(k) for k in
            ('recording', 'glim', 'offline', 'export', 'tool', 'calibration_record', 'calibration_tool',
             'reconstruction', *engines.ENGINE_PROCESS_KEYS)):
        raise ValueError('Finish capture and processing before preparing reconstruction')
    if not (session/'raw_bag/metadata.yaml').is_file():
        raise ValueError('Raw bag metadata is missing')
    run = session/'reconstruction'/('run_'+uuid.uuid4().hex[:12])
    run.mkdir(parents=True)
    stored_source = ({key: value for key, value in edit_source.items() if key not in ('export', 'trajectory')}
                     if edit_source else None)
    job = dict(id=run.name, state='running', created_at=now(), trajectory=trajectory,
               algorithm=engine, voxel_size_m=size, save_full_density=save_full_density,
               filter_edited_geometry=filter_edited_geometry, edited_geometry_source=stored_source)
    if engine == engines.VDBFUSION:
        job['vdbfusion_settings'] = validate_vdbfusion_settings(vdbfusion)
    atomic_json(run/'job.json', job)
    config = read_json(session/'active_config.json', service.config)
    topic = config['sensor']['points_topic']
    if engine == engines.VDBFUSION:
        from .vdbfusion_jobs import prepare_arguments
        # The VDBFusion preparation step validates sources and resources only; raw
        # observations are streamed during mesh reconstruction.
        args = prepare_arguments(service.root, run, session/'raw_bag', source, topic,
                                 dict(job['vdbfusion_settings'], mesh_output_mode='merged'),
                                 service.edit_workspace(sid, edit_id) if edit_source else None)
        env = os.environ.copy()
        env.pop('PYTHONHOME', None)
    else:
        args = [sys.executable, '-u', '-m', 'factory_mapping.reconstruction',
                '--bag', str(session/'raw_bag'), '--trajectory', str(source),
                '--output-dir', str(run), '--voxel-size', str(size), '--topic', topic]
        if save_full_density:
            args.append('--save-full-density')
        if edit_source:
            provenance = dict(filter_enabled=True, edited_geometry_source=stored_source,
                              trajectory=trajectory,
                              filter_description='Approximate retained-geometry filtering; NKSR can still bridge deleted regions.')
            provenance_path = run/'provenance.json'
            atomic_json(provenance_path, provenance)
            args += ['--edited-export', str(edit_source['export']),
                     '--filter-tolerance', str(edit_source['tolerance_m']),
                     '--provenance-json', str(provenance_path)]
        from .commands import ros_env
        env = ros_env(service.config, offline=True)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + ':' + env.get('PYTHONPATH', '')
    marker = run/engines.PREPARED_MARKERS[engine]

    async def done(item):
        prepared = item['state'] == 'completed' and marker.is_file()
        job.update(state='PREPARED' if prepared else ('failed' if item['state'] == 'completed' else item['state']),
                   ended_at=now(), returncode=item['returncode'])
        atomic_json(run/'job.json', job)
    try:
        # The managed subprocess supports recovery, shutdown and log capture.
        await service.pm.start('reconstruction', args, run/'job.log', env, done)
    except Exception as error:
        job.update(state='failed', error=str(error))
        atomic_json(run/'job.json', job)
        raise
    return job


def validate_vdbfusion_settings(settings):
    """Validate the VDBFusion request that preparation will preflight."""
    from .vdbfusion import resolve_settings
    resolved = resolve_settings(settings or {})
    # Keep the caller's advanced controls that resolve_settings does not own.
    for key in ('association_spacing_multiplier', 'boundary_margin_m', 'unsupported_observations',
                'mask_deleted_triangles', 'roi_min_m', 'roi_max_m', 'memory_budget_gib'):
        if settings and settings.get(key) is not None:
            resolved[key] = settings[key]
    return resolved
