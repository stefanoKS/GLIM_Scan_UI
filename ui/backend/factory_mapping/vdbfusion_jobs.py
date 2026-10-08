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
# Memory supervisor (GATE 8): the worker can be inside a native call that holds the GIL,
# so the samples are taken by the backend from the worker's process tree (an "outside in"
# sample) rather than by a thread inside the worker.
MEMORY_SAMPLE_INTERVAL_SECONDS = 2.0
MEMORY_PRESSURE_RATIO = 0.95
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


class MemorySupervisor:
    """Sample the VDBFusion worker process tree while it runs, and record where it peaked.

    The backend samples rather than the worker, so sampling keeps working while the worker is
    inside a native call that holds the interpreter lock. Every figure is labelled as a
    *sampled* peak at a documented interval: it is a lower bound on the true peak, never a
    claim to have instrumented the allocations.

    On dangerous pressure the supervisor requests cooperative cancellation once - the worker
    checks cancellation at every bounded batch boundary - and records the event. It never
    kills the worker itself, and it never touches previous outputs.
    """

    def __init__(self, run_dir, pid, budget_bytes, interval=MEMORY_SAMPLE_INTERVAL_SECONDS,
                 pressure_ratio=MEMORY_PRESSURE_RATIO, on_pressure=None, read_rss=None,
                 read_stage=None, max_samples=None):
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.pid = pid
        self.limit_bytes = int(budget_bytes or 0)
        self.interval = max(float(interval), 0.0)
        self.pressure_ratio = float(pressure_ratio)
        self.on_pressure = on_pressure
        self.read_rss = read_rss or self._tree_rss
        self.read_stage = read_stage or self._stage
        self.max_samples = max_samples
        self.samples = 0
        self.peak_rss_bytes = 0
        self.peak_stage = None
        self.last_rss_bytes = 0
        self.samples_by_stage = {}
        self.elapsed_by_stage = {}
        self.pressure_events = []
        self._pressure_requested = False
        self._started = None

    def _tree_rss(self):
        import psutil
        try:
            process = psutil.Process(self.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            return 0
        total = 0
        processes = [process]
        try:
            processes += process.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        for item in processes:
            try:
                total += item.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return int(total)

    def _stage(self):
        if self.run_dir is None:
            return None
        return read_json(self.run_dir/PROGRESS_FILE, {}).get('stage')

    def sample(self, at=None):
        """Take one measurement and apply the pressure policy."""
        import time as _time
        at = _time.monotonic() if at is None else at
        if self._started is None:
            self._started = at
        rss = int(self.read_rss() or 0)
        stage = self.read_stage() or self.peak_stage
        self.samples += 1
        self.last_rss_bytes = rss
        if stage:
            self.samples_by_stage[stage] = self.samples_by_stage.get(stage, 0) + 1
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss
            self.peak_stage = stage
        pressure = bool(self.limit_bytes) and rss > self.limit_bytes * self.pressure_ratio
        if pressure and not self._pressure_requested:
            self._pressure_requested = True
            event = dict(stage=stage, rss_bytes=rss, limit_bytes=self.limit_bytes,
                         at=time.time(), action='requested cooperative cancellation')
            self.pressure_events.append(event)
            if self.on_pressure is not None:
                asyncio.ensure_future(self.on_pressure(event))
        return rss

    def close(self, at=None):
        """Close the stage timings from the per-stage sample counts."""
        for stage, count in self.samples_by_stage.items():
            self.elapsed_by_stage[stage] = round(count * self.interval, 3)
        return self.summary()

    def summary(self):
        return dict(sampled=True, sample_interval_seconds=self.interval, samples=self.samples,
                    peak_rss_bytes=self.peak_rss_bytes, peak_stage=self.peak_stage,
                    last_rss_bytes=self.last_rss_bytes, limit_bytes=self.limit_bytes,
                    pressure_ratio=self.pressure_ratio,
                    samples_by_stage=dict(self.samples_by_stage),
                    elapsed_seconds_by_stage=dict(self.elapsed_by_stage),
                    pressure_events=self.pressure_events,
                    note='Sampled from the backend over the worker process tree at the documented interval, so the '
                         'peak is a lower bound on the true peak, not an instrumented measurement.')

    async def run(self):
        try:
            while self.max_samples is None or self.samples < self.max_samples:
                self.sample()
                if self.interval:
                    await asyncio.sleep(self.interval)
                else:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:  # a finished or stopped worker ends the sampling
            raise
        finally:
            self.close()


def supervisor_limit_bytes(settings):
    """The byte-valued limit the memory supervisor samples against.

    Derived from the same normaliser the engine and the worker use, so a budget set in the
    UI (``memory_budget_gib``) is honoured instead of leaving the supervisor with a zero
    limit that can never escalate.
    """
    from .vdbfusion import DEFAULT_MEMORY_BUDGET_BYTES, resolve_memory_budget_bytes
    return resolve_memory_budget_bytes(settings or {}) or DEFAULT_MEMORY_BUDGET_BYTES


def settings_status(record, current, key='settings'):
    """Compare a prepared VDBFusion record against the settings a mesh request asks for.

    Only semantic settings are compared, and only where the request supplies a value: an
    omitted value means "use the prepared one", which is what the run then uses.
    """
    from .vdbfusion import compare_semantic_settings
    prepared = (record or {}).get(key) or {}
    return compare_semantic_settings(prepared, current)


def enforce_prepared_identity(run, record, job, request, sid=None, source_identity=None):
    """Refuse a reconstruction whose prepared settings or source no longer describe it.

    Returns the settings the worker must run with: the prepared semantic settings plus the
    request's execution-only choices. A mismatch is reported with the differing keys and an
    explicit re-prepare instruction, never silently resolved in favour of either side.
    """
    from .vdbfusion import compare_source_identity, effective_settings
    prepared = record.get('settings') or {}
    status = settings_status(record, request)
    if not status['matches']:
        raise ValueError(
            'PREPARED_SETTINGS_STALE: this run was prepared with different VDBFusion settings, so its validated '
            'resource preflight and edited-geometry reference no longer describe it. Differing '
            f'{", ".join(status["differing"])}: prepared '
            f'{ {key: value.get("prepared") for key, value in status["detail"].items()} }, requested '
            f'{ {key: value.get("requested") for key, value in status["detail"].items()} }. '
            'Prepare VDBFusion again with these settings; the requested values are not substituted silently.')
    identity = record.get('identity') or {}
    prepared_source = identity.get('source') or {}
    if prepared_source and source_identity is not None:
        diff = compare_source_identity(prepared_source, source_identity)
        if not diff['matches']:
            raise ValueError(
                'PREPARED_SOURCE_CHANGED: the prepared source is no longer the same recording or trajectory '
                f'(changed: {", ".join(diff["differing"])}). Prepare VDBFusion again before reconstructing.')
    return effective_settings(prepared, request)


def enforce_resources(record, settings, run):
    """Re-run the resource preflight before a long native run and refuse when it fails."""
    from .vdbfusion import preflight
    scan = record.get('bag_scan') or {}
    geometry = record.get('edited_geometry') or {}
    report = preflight(run, settings, observed_points=scan.get('estimated_observations'),
                       observed_bbox=(scan['observed_bbox_min_m'], scan['observed_bbox_max_m'])
                       if scan.get('observed_bbox_min_m') else None,
                       bag_bytes=record.get('bag_bytes'), occupancy=scan.get('occupancy'),
                       batch_points=settings.get('batch_points'),
                       memory_budget_bytes=supervisor_limit_bytes(settings),
                       reference_points=geometry.get('reference_points'))
    if not report['ok']:
        raise ValueError('; '.join(report['failures'] + report['suggestions']))
    return report


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
    # When the worker recorded a publish manifest, the artifacts it lists must be the ones on
    # disk: that is what makes a partially published output set detectable.
    if metadata.get('publish_manifest'):
        from .vdbfusion import PUBLISH_MANIFEST, artifact_size_bytes, output_artifact_names
        manifest = read_json(output/PUBLISH_MANIFEST, None)
        if not manifest:
            raise ValueError('Worker reported a publish manifest but it is missing')
        expected_artifacts = sorted(output_artifact_names(mode))
        recorded = sorted(manifest.get('artifacts') or {})
        if recorded != expected_artifacts or sorted(manifest.get('published') or []) != expected_artifacts:
            raise ValueError(f'Publish manifest records {recorded}, expected {expected_artifacts}')
        for name, entry in (manifest.get('artifacts') or {}).items():
            path = output/name
            if not path.exists():
                raise ValueError(f'Publish manifest lists {name} but it is not on disk')
            if entry.get('bytes') is not None and entry['bytes'] != artifact_size_bytes(path):
                raise ValueError(f'Published artifact {name} does not match the manifest size')
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
    # The prepared settings and the prepared source are authoritative. A request that asks
    # for a different resolution is refused, and the run always executes with the settings
    # its validated preflight and edited-geometry reference were built from.
    from .vdbfusion import source_identity
    current_source = source_identity(record['bag'], record['trajectory'], record.get('topic', '/livox/lidar'),
                                    service.edit_workspace(sid, source['edit_id'])
                                    if job.get('filter_edited_geometry') and source else None,
                                    edit_fingerprints=dict(edit_id=source['edit_id'])
                                    if job.get('filter_edited_geometry') and source else None)
    effective = enforce_prepared_identity(run, record, job, request, sid=sid, source_identity=current_source)
    enforce_resources(record, effective, run)
    # Preserve successful and failed prior attempts. Never overwrite a validated mesh.
    archive = run/'attempts'/('previous_'+uuid.uuid4().hex[:12])
    for name in ('output', 'mesh_job.json', PROGRESS_FILE, 'staging'):
        path = run/name
        if path.exists():
            archive.mkdir(parents=True, exist_ok=True)
            path.rename(archive/name)
    data = dict(state='RUNNING', started_at=now(), algorithm=engines.VDBFUSION, engine=engines.VDBFUSION,
                settings=effective, requested_settings=request, python=str(python),
                prepared_semantic_fingerprint=(record.get('identity') or {}).get('semantic_fingerprint'))
    atomic_json(run/'mesh_job.json', data)
    settings = settings_arguments(effective)
    settings += ['--mesh-output-mode', str(effective.get('mesh_output_mode', 'merged'))]
    if effective.get('chunk_size') is not None:
        settings += ['--chunk-size', repr(float(effective['chunk_size']))]
    args = [str(python), str(worker_path()), '--bag', record['bag'], '--trajectory', record['trajectory'],
            '--topic', record.get('topic', '/livox/lidar'), '--output', str(output/'mesh.ply'),
            '--metadata', str(output/'vdbfusion_metadata.json'), '--progress', str(run/PROGRESS_FILE)]
    args += settings
    if job.get('filter_edited_geometry'):
        args += ['--edited-workspace', str(service.edit_workspace(sid, source['edit_id']))]

    async def done(item):
        progress = read_json(run/PROGRESS_FILE, {})
        if supervisor_task is not None and not supervisor_task.done():
            supervisor_task.cancel()
        if supervisor is not None:
            data['memory'] = supervisor.close()
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
    supervisor = None
    supervisor_task = None
    try:
        item = await service.pm.start('vdbfusion', args, run/'job.log', worker_environment(), done)
        # The prepared record has no execution-only fields, so the limit comes from the
        # request (memory_budget_gib) unless an explicit byte value was supplied.
        limit = supervisor_limit_bytes(effective)
        supervisor = MemorySupervisor(run, item.get('pid'), limit,
                                      on_pressure=lambda event: service.pm.stop('vdbfusion', 60, cancel=True))
        supervisor_task = asyncio.ensure_future(supervisor.run())
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
