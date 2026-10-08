#!/usr/bin/env python3
"""VDBFusion worker: real native TSDF integration, extraction and mesh output.

Run with the isolated VDBFusion interpreter (``VDBFUSION_PYTHON``), never with the
ROS or NKSR interpreters. The web backend never imports the native library, so a
native fault can only take down this managed subprocess.

The recording is streamed: bounded observation batches are read from the raw bag,
transformed with the GLIM trajectory, optionally filtered against a verified saved
cleanup, grouped by real interpolated sensor origins and handed to the native
integrator. No whole-cloud array or full-bag load exists anywhere in this process.
"""
import argparse
import gc
import json
import signal
import shutil
import sys
import time
import traceback
from pathlib import Path
import numpy as np
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factory_mapping import vdbfusion as V  # noqa: E402
from factory_mapping.mesh_partition import partition_mesh  # noqa: E402
from factory_mapping.nksr_mesh import inspect_mesh  # noqa: E402

DEFAULT_CHUNK_SIZE_M = 5.0
DEFAULT_CHECK_BATCH = 4_000
STAGE_ORDER = ('INIT', 'VALIDATING_SOURCE', 'PREFLIGHT', 'READING_BAG', 'INTEGRATING', 'EXTRACTING_MESH',
               'VALIDATING_MESH', 'SAVING_MESH', 'PARTITIONING_MESH', 'SAVING_MESH_CHUNKS', 'COMPLETED')


class WorkerError(RuntimeError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def classify(error, stage):
    if isinstance(error, WorkerError):
        return error.code
    if isinstance(error, V.MemoryBudgetExceeded):
        return 'MEMORY_BUDGET_EXCEEDED'
    if isinstance(error, (ModuleNotFoundError, ImportError, OSError)) and stage in ('INIT', 'VALIDATING_SOURCE'):
        return 'VDBFUSION_NOT_INSTALLED'
    if stage in ('READING_BAG',):
        return 'RAW_BAG_UNREADABLE'
    if stage == 'VALIDATING_SOURCE' and 'bag' in str(error).lower():
        return 'RAW_BAG_UNREADABLE'
    if stage == 'PREFLIGHT':
        return 'PREFLIGHT_FAILED'
    if stage == 'INTEGRATING':
        return 'INTEGRATION_FAILED'
    if stage == 'EXTRACTING_MESH':
        return 'MESH_EXTRACTION_FAILED'
    if stage in ('VALIDATING_MESH', 'SAVING_MESH', 'PARTITIONING_MESH', 'SAVING_MESH_CHUNKS'):
        return 'MESH_INVALID'
    return 'VDBFUSION_RECONSTRUCTION_FAILED'


def runtime_metadata():
    import importlib.metadata
    import vdbfusion
    import numpy
    import scipy
    try:
        plyfile_version = importlib.metadata.version('plyfile')
    except importlib.metadata.PackageNotFoundError:
        plyfile_version = None
    metadata = dict(python=sys.executable, python_version=sys.version.split()[0],
                    vdbfusion_version=getattr(vdbfusion, '__version__', None),
                    vdbfusion_module=str(Path(vdbfusion.__file__).resolve()),
                    numpy_version=numpy.__version__, scipy_version=scipy.__version__,
                    plyfile_version=plyfile_version)
    provenance = Path(sys.prefix) / 'vdbfusion-provenance.json'
    if provenance.is_file():
        metadata.update(json.loads(provenance.read_text()))
        metadata.update(python=sys.executable, python_version=sys.version.split()[0])
    return metadata


def roi_clipped_bbox(settings, observed_bbox):
    """Intersect the sampled scan bounds with the requested region of interest."""
    lower = np.asarray(observed_bbox[0], dtype=np.float64)
    upper = np.asarray(observed_bbox[1], dtype=np.float64)
    requested = bool(settings.get('roi_min_m')) or bool(settings.get('roi_max_m'))
    if settings.get('roi_min_m'):
        lower = np.maximum(lower, np.asarray(settings['roi_min_m'], dtype=np.float64))
    if settings.get('roi_max_m'):
        upper = np.minimum(upper, np.asarray(settings['roi_max_m'], dtype=np.float64))
    # A flat sampled extent is legitimate, so only a genuinely empty intersection fails.
    if requested and np.any(upper < lower):
        raise WorkerError('SETTINGS_INVALID',
                          'The region of interest does not overlap the observed scan bounds')
    return lower.tolist(), upper.tolist()


def validate_mesh_arrays(vertices, triangles):
    vertices = np.asarray(vertices)
    triangles = np.asarray(triangles)
    if vertices.dtype != np.float64 or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise WorkerError('MESH_INVALID', f'Native extraction returned {vertices.dtype}/{vertices.shape} vertices')
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in 'iu':
        raise WorkerError('MESH_INVALID', f'Native extraction returned {triangles.dtype}/{triangles.shape} triangles')
    if not len(vertices) or not len(triangles):
        raise WorkerError('MESH_EMPTY', 'Native extraction returned an empty triangle mesh')
    if not np.isfinite(vertices).all():
        raise WorkerError('MESH_INVALID', 'Native extraction returned non-finite vertices')
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise WorkerError('MESH_INVALID', 'Native extraction returned triangle indices outside the vertex range')


def arg_settings(args):
    """Translate CLI arguments into the validated settings input of :mod:`vdbfusion`."""
    return dict(voxel_size_m=args.voxel_size, sdf_trunc_m=args.sdf_trunc, preset=args.preset,
                space_carving=args.space_carving == '1', mesh_output_mode=args.mesh_output_mode,
                memory_budget_gib=args.memory_budget_gib, batch_points=args.batch_points,
                origin_error_budget_m=args.origin_error_budget,
                roi_min_m=list(args.roi_min) if args.roi_min else None,
                roi_max_m=list(args.roi_max) if args.roi_max else None)


def run_check():
    """Real primitive check: import, construct a TSDF, integrate, extract, validate.

    Import success alone is never sufficient (GATE 3).
    """
    import vdbfusion
    metadata = runtime_metadata()
    rng = np.random.default_rng(20261008)
    # A synthetic plane seen from three real-looking viewpoints inside the z range.
    points = []
    origins = []
    for index, offset in enumerate((0.0, 0.35, -0.3)):
        count = DEFAULT_CHECK_BATCH
        x = rng.uniform(-1.5, 1.5, count)
        y = rng.uniform(-1.5, 1.5, count)
        points.append(np.column_stack((x, y, np.full(count, 3.0))))
        origins.append(np.array([offset, offset / 2.0, 0.0]))
    start = time.monotonic()
    volume = vdbfusion.VDBVolume(0.05, 0.15, False)
    for batch, origin in zip(points, origins):
        volume.integrate(np.ascontiguousarray(batch, dtype=np.float64),
                         np.ascontiguousarray(origin, dtype=np.float64))
    vertices, triangles = volume.extract_triangle_mesh()
    validate_mesh_arrays(vertices, triangles)
    vertices = np.asarray(vertices)
    metadata.update(status='READY', smoke_passed=True, check_seconds=time.monotonic() - start,
                    check_voxel_size_m=0.05, check_sdf_trunc_m=0.15,
                    check_vertices=int(len(vertices)), check_triangles=int(len(triangles)),
                    check_bounding_box_min=vertices.min(axis=0).tolist(),
                    check_bounding_box_max=vertices.max(axis=0).tolist(),
                    native_backend='upstream PRBonn VDBFusion Python bindings (OpenVDB TSDF)',
                    message='Native integration and triangle extraction produced a finite index-valid mesh')
    return metadata


def bag_topic_frames(bag, topic):
    """Message count for one topic, read from the bag's own metadata.yaml."""
    import yaml
    metadata = Path(bag) / 'metadata.yaml'
    if not metadata.is_file():
        return None
    try:
        info = yaml.safe_load(metadata.read_text())['rosbag2_bagfile_information']
        for entry in info.get('topics_with_message_count', []):
            item = entry.get('topic_metadata', entry)
            if item.get('name') == topic:
                return int(entry.get('message_count', 0)) or None
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return None


def sample_bag_extent(bag, trajectory, topic, samples=8, voxel_size_m=None):
    """Evenly spread bounded sample of real world-space observations.

    Seeks across the interval where the trajectory and the bag actually overlap, so a
    trajectory trimmed to a subset still yields a meaningful scan summary. Only a
    handful of frames are decoded, keeping preparation lightweight even for a 10 GiB
    recording.
    """
    from factory_mapping.reconstruction import get_cloud_arrays, transform_points
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2

    total_frames = bag_topic_frames(bag, topic)
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='sqlite3'),
                    rosbag2_py.ConverterOptions('', ''))
    except Exception as error:  # noqa: BLE001 - a storage failure is a source problem, not a crash
        raise WorkerError('RAW_BAG_UNREADABLE', f'Raw bag cannot be opened: {error}') from error
    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    sampled = 0
    observations = 0
    # Bounded sample of real observations, used to *measure* how many TSDF voxels the
    # recording actually touches instead of assuming one voxel per observation.
    collected = []

    def collect(payload):
        nonlocal sampled, observations
        xyz, intensity, timestamps = get_cloud_arrays(deserialize_message(payload, PointCloud2))
        points, origins, valid = transform_points(xyz, timestamps, trajectory)
        observations += len(points)
        if len(points):
            lower[:] = np.minimum(lower, points.min(axis=0))
            upper[:] = np.maximum(upper, points.max(axis=0))
            if voxel_size_m:
                collected.append(np.ascontiguousarray(points, dtype=np.float64))
        sampled += 1

    stamps = seek_targets(bag, trajectory, samples)
    for stamp in stamps:
        try:
            reader.seek(stamp)
        except Exception:  # noqa: BLE001 - a storage backend without seek falls back below
            break
        for _ in range(400):
            if not reader.has_next():
                break
            name, data, _ = reader.read_next()
            if name == topic:
                collect(data)
                break
    if not sampled:
        # Seek unavailable or empty: walk a bounded prefix instead.
        while reader.has_next() and sampled < samples:
            name, data, _ = reader.read_next()
            if name == topic:
                collect(data)
    if not sampled or not np.isfinite(lower).all():
        raise WorkerError('RAW_BAG_UNREADABLE',
                          'Sampled LiDAR frames produced no observations inside the trajectory range')
    per_frame = observations / sampled
    estimated = int(round(per_frame * total_frames)) if total_frames else int(observations)
    occupancy = None
    if collected:
        sample_points = np.concatenate(collected)
        del collected
        cells = np.floor(sample_points / float(voxel_size_m)).astype(np.int64)
        occupied = int(len(np.unique(cells.view([('x', '<i8'), ('y', '<i8'), ('z', '<i8')]))))
        occupancy = dict(voxels=occupied, observations=int(len(sample_points)),
                         voxel_size_m=float(voxel_size_m),
                         note='Measured on evenly spread sampled frames at the requested voxel size, so it is a '
                              'sample of the touched surface, not a promise for the whole recording.')
    return dict(lidar_frames=total_frames, sampled_frames=sampled, sampled_observations=int(observations),
                estimated_observations=estimated, observations_per_frame=float(per_frame),
                extent_is_sample=True, occupancy=occupancy,
                extent_note='Observed bounds come from evenly spread sampled frames, not from the whole recording.',
                observed_bbox_min_m=lower.tolist(), observed_bbox_max_m=upper.tolist()), estimated


def seek_targets(bag, trajectory, samples):
    """Bag timestamps to sample, restricted to the trajectory's own time range."""
    start_ns = bag_start_time(bag)
    end_ns = start_ns + bag_duration_ns(bag)
    span_start = max(start_ns, int(round(float(trajectory[0, 0]) * 1e9)))
    span_end = min(end_ns, int(round(float(trajectory[-1, 0]) * 1e9)))
    if span_end <= span_start:
        return []
    fractions = [index / max(samples - 1, 1) for index in range(samples)]
    return [span_start + int((span_end - span_start) * fraction) for fraction in fractions]


def bag_start_time(bag):
    import yaml
    info = yaml.safe_load((Path(bag) / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
    return int(info['starting_time']['nanoseconds_since_epoch'])


def bag_duration_ns(bag):
    import yaml
    info = yaml.safe_load((Path(bag) / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
    return int(info['duration']['nanoseconds'])


def load_edit_reference(workspace, association_radius_m, association_multiplier):
    if workspace is None:
        return None
    reference = V.load_edited_reference(workspace, association_radius_m=association_radius_m,
                                        association_spacing_multiplier=association_multiplier)
    if not (reference.metadata['retained_points'] and reference.metadata['removed_points']):
        raise WorkerError('EDIT_REFERENCE_UNSUPPORTED',
                          'Saved cleanup geometry cannot be validated: it has no removed geometry to exclude')
    return reference


def resolve_chunk_size(requested_chunk_size_m, observed_bbox):
    if requested_chunk_size_m is not None:
        value = float(requested_chunk_size_m)
        if not np.isfinite(value) or value <= 0:
            raise WorkerError('SETTINGS_INVALID', 'Chunk size must be a positive, finite size in metres')
        return value, 'user'
    if observed_bbox is None:
        return DEFAULT_CHUNK_SIZE_M, 'default'
    lower, upper = np.asarray(observed_bbox[0]), np.asarray(observed_bbox[1])
    span = float(np.max(upper - lower))
    if not np.isfinite(span) or span <= 0:
        return DEFAULT_CHUNK_SIZE_M, 'default'
    # Target roughly 100 export cells while keeping a human-friendly size.
    return float(max(1.0, round(span / 10.0, 1))), 'auto_extent'


def write_outputs(settings, vertices, faces, observed_bbox, event):
    """Save the selected mesh output from the one final fused mesh.

    Chunked output reuses the existing validated spatial partitioner on the single
    fused surface, so no independent TSDF tile is ever reconstructed and no
    tile-boundary seam is introduced.
    """
    mode = settings['mesh_output_mode']
    output = Path(settings['output'])
    # Everything is written into a staging directory first; the caller publishes it only
    # after the staged mesh has been validated, so a partial or invalid mesh is never at
    # the published path (GATE 11).
    chunks_dir = (Path(settings['staging_root']) if settings.get('staging_root') else output.parent) / 'mesh_chunks'
    fields = dict(mesh_output_mode=mode, output_bytes=0)
    chunk_totals = manifest = None
    if mode in ('chunks', 'both'):
        size, source = resolve_chunk_size(settings.get('chunk_size'), observed_bbox)
        event('PARTITIONING_MESH', message=f'Partitioning the fused mesh into {size:g} m export cells')
        manifest, chunk_totals = partition_mesh(
            vertices, faces, size, chunks_dir, reconstruction_mode='vdbfusion_fused_tsdf',
            requested_chunk_size_m=size, chunk_size_source=source, event=event)
        fields.update(chunk_manifest_path='mesh_chunks/chunks.json',
                      export_strategy=manifest['export_strategy'], chunk_count=chunk_totals['chunk_count'],
                      chunk_vertices_total=chunk_totals['total_vertices'],
                      chunk_faces_total=chunk_totals['total_faces'],
                      chunk_union_bounds=chunk_totals['union_bounds'],
                      chunk_size_m=manifest['effective_chunk_size_m'],
                      chunk_size_source=source,
                      requested_chunk_size_m=manifest['requested_chunk_size_m'])
        fields['output_bytes'] += sum(chunk['file_size_bytes'] for chunk in manifest['chunks'])
    if mode != 'chunks':
        event('SAVING_MESH')
        stats = V.write_mesh_ply(output, vertices, faces)
        V.validate_written_mesh(output, stats)
        fields.update(stats)
        fields['output_bytes'] += stats['mesh_file_size']
    return fields


def reconstruct(settings, event, cancel=None):
    bag = Path(settings['bag'])
    trajectory_path = Path(settings['trajectory'])
    if not bag.is_dir():
        raise WorkerError('RAW_BAG_UNREADABLE', 'Raw bag directory is missing')
    if not trajectory_path.is_file():
        raise WorkerError('INPUT_INVALID', 'Trajectory file is missing')
    output = Path(settings['output']) if settings['output'] else None
    if output is not None and output.exists():
        raise WorkerError('MESH_INVALID', 'Output exists; choose a new output path')

    event('VALIDATING_SOURCE')
    trajectory, trajectory_report = V.load_trajectory(trajectory_path)
    resolved = V.resolve_settings(settings, origin_error_budget_m=settings.get('origin_error_budget_m'))
    edit_reference = load_edit_reference(settings.get('edited_workspace'),
                                         settings.get('association_radius_m'),
                                         settings.get('association_spacing_multiplier'))
    extent, estimated_points = sample_bag_extent(bag, trajectory, settings['topic'],
                                                voxel_size_m=resolved['voxel_size_m'])
    bag_bytes = sum(p.stat().st_size for p in bag.rglob('*') if p.is_file())
    clipped = roi_clipped_bbox(resolved, (extent['observed_bbox_min_m'], extent['observed_bbox_max_m']))
    # The preflight is re-run here with the settings this run actually executes with, so a
    # resource change since preparation - or a finer resolution requested at mesh time - is
    # refused instead of silently attempted.
    budget_bytes = V.resolve_memory_budget_bytes(settings)
    preflight = V.preflight(output.parent if output else Path.cwd(), resolved,
                            observed_points=estimated_points, observed_bbox=clipped, bag_bytes=bag_bytes,
                            occupancy=extent.get('occupancy'),
                            batch_points=resolved.get('batch_points'),
                            memory_budget_bytes=budget_bytes,
                            reference_points=(edit_reference.metadata.get('reference_points')
                                              if edit_reference else None))
    if not preflight['ok']:
        raise WorkerError('RESOURCE_PREFLIGHT_FAILED', '; '.join(
            preflight['failures'] + preflight['suggestions']))
    event('PREFLIGHT', **{key: preflight[key] for key in
                          ('free_ram_bytes', 'free_disk_bytes', 'estimated_tsdf_bytes',
                           'estimated_peak_bytes', 'usable_bytes')},
          warnings=preflight['warnings'], message='; '.join(preflight['warnings']) if preflight['warnings'] else
          'Resource preflight passed')

    volume, stats = V.integrate_bag(bag, trajectory, settings['topic'], resolved, output=output,
                                    edit_reference=edit_reference,
                                    boundary_margin_m=settings.get('boundary_margin_m') or 0.0,
                                    unsupported_policy=settings.get('unsupported_policy') or 'exclude',
                                    memory_budget_bytes=budget_bytes,
                                    event=event, cancel=cancel, peak_limit_bytes=budget_bytes)
    event('EXTRACTING_MESH')
    V.check_cancellation(cancel)
    vertices, faces, mesh_stats = V.extract_and_mask(
        volume, edit_reference, association_radius_m=(edit_reference.resolve_association_radius()
                                                     if edit_reference else None),
        boundary_margin_m=settings.get('boundary_margin_m') or 0.0, event=event,
        mask_deleted_triangles=bool(settings.get('mask_deleted_triangles', True)), cancel=cancel)
    del volume
    gc.collect()
    # A cancellation that arrived while the native extractor held the interpreter is honoured
    # here, before the geometry audit and before any mesh is written to disk. Measured on a
    # 7.1 million triangle mesh: without this checkpoint the worker still spent about eight
    # seconds writing a 135 MiB staged PLY before noticing.
    V.check_cancellation(cancel)
    event('VALIDATING_MESH')
    metadata = V.validate_and_report(vertices, faces, resolved, extra=dict(
        settings=dict(resolved), trajectory=trajectory_report, bag_scan=extent, preflight=preflight,
        integration=stats, extraction=mesh_stats, engine='vdbfusion',
        algorithm='vdbfusion'),
        observed_bbox=(clipped if clipped else (extent['observed_bbox_min_m'], extent['observed_bbox_max_m'])))
    if edit_reference is not None:
        # Mirror the filtering decision at the top level so a reader never has to dig
        # into the integration details to see how the saved edit was applied.
        metadata.update(edit_filter_accuracy=stats.get('edit_filter_accuracy'),
                        association_radius_m=stats.get('association_radius_m'),
                        unsupported_observations=stats.get('unsupported_observations'),
                        mask_deleted_triangles=bool(settings.get('mask_deleted_triangles', True)),
                        edited_geometry=stats.get('edit_reference'))
    staging = output.parent/'staging' if output is not None else None
    if staging is not None:
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
    try:
        fields = write_outputs(dict(resolved, output=(staging/output.name if staging else None),
                                   staging_root=staging, chunk_size=settings.get('chunk_size')), vertices, faces,
                               (extent['observed_bbox_min_m'], extent['observed_bbox_max_m']), event)
        metadata.update(fields)
        if staging is not None:
            # Validate the whole staged set, then publish it. An invalid or incomplete set must
            # never replace a valid published one (GATE 11).
            V.check_cancellation(cancel)
            manifest = V.publish_output_set(staging, output.parent, settings['mesh_output_mode'],
                                           expected=dict(vertex_count=len(vertices), face_count=len(faces)) if
                                           settings['mesh_output_mode'] != 'chunks' else None,
                                           metadata=fields)
            published = manifest['published']
            metadata['published_outputs'] = [str(output.parent/name) for name in published]
            metadata['publish_manifest'] = f'{V.PUBLISH_MANIFEST}'
            # The manifest is the completion marker: recording its attempt id in the metadata
            # (written only on success) lets the backend refuse any set that is not this run's.
            metadata['publish_attempt_id'] = manifest['attempt_id']
    finally:
        # A cancelled or failed run must not leave an unpublished partial mesh on disk: the
        # published path is never touched, and the staging area is removed. A successful
        # publish has already removed it.
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    if output is not None and settings['mesh_output_mode'] != 'chunks':
        persisted = inspect_mesh(output)
        metadata['vertex_count'] = persisted['vertex_count']
        metadata['face_count'] = persisted['face_count']
        metadata['validation_status'] = 'PASS'
        metadata['validation_note'] = ('The saved PLY was re-read with the independent validator; vertex and face '
                                       'counts match the mesh produced by the native TSDF extraction.')
    else:
        metadata['validation_status'] = 'PASS'
    metadata['elapsed_seconds'] = time.monotonic() - settings['_started']
    metadata['peak_rss_bytes'] = int(max(stats.get('peak_rss_bytes', 0), metadata.get('peak_rss_bytes', 0)))
    return metadata


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true', help='Real native integration + extraction smoke test')
    p.add_argument('--health-output', type=Path)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--bag', type=Path)
    p.add_argument('--trajectory', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--metadata', type=Path)
    p.add_argument('--progress', type=Path)
    p.add_argument('--prepare-output', type=Path)
    p.add_argument('--edited-workspace', type=Path)
    p.add_argument('--association-radius', type=float)
    p.add_argument('--association-spacing-multiplier', type=float)
    p.add_argument('--boundary-margin', type=float)
    p.add_argument('--unsupported-policy', choices=['exclude', 'include'], default='exclude')
    p.add_argument('--mask-deleted-triangles', choices=['0', '1'], default='1')
    p.add_argument('--memory-budget-gib', type=float)
    p.add_argument('--topic', default='/livox/lidar')
    p.add_argument('--preset', choices=sorted(V.PRESETS))
    p.add_argument('--voxel-size', type=float)
    p.add_argument('--sdf-trunc', type=float)
    p.add_argument('--space-carving', choices=['0', '1'], default='0')
    p.add_argument('--origin-error-budget', type=float)
    p.add_argument('--batch-points', type=int)
    p.add_argument('--mesh-output-mode', choices=list(V.MESH_OUTPUT_MODES), default='merged')
    p.add_argument('--chunk-size', type=float)
    p.add_argument('--roi-min', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    p.add_argument('--roi-max', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    return p


def main():
    args = parser().parse_args()
    started = time.monotonic()
    stage = 'INIT'
    metadata = {'python': sys.executable}

    def event(next_stage, **fields):
        nonlocal stage
        stage = next_stage
        obj = dict(stage=stage, at=time.time(), **fields)
        print(json.dumps(obj), file=sys.stderr, flush=True)
        if args.progress:
            atomic_json(args.progress, obj)

    # Cancellation is cooperative: a flag is raised and checked at every bounded
    # batch boundary, so the native integrator is never abandoned mid-call. A second
    # signal unwinds immediately instead. SIGTERM behaves exactly like SIGINT because it is
    # the escalation step of the managed stop ladder: a handler that only sets the flag lets
    # the worker stop at the next checkpoint and report CANCELLED instead of being killed
    # with no state, and the parent still escalates to SIGKILL if a checkpoint is too far
    # away. A Python signal handler only runs when the interpreter regains control, so this
    # never claims to interrupt a native call.
    cancelled = {'count': 0}

    def on_signal(signum, frame):
        cancelled['count'] += 1
        if cancelled['count'] > 1:
            raise WorkerError('CANCELLED', 'VDBFusion reconstruction cancelled')

    def check_cancel():
        if cancelled['count']:
            raise WorkerError('CANCELLED', 'VDBFusion reconstruction cancelled')

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        if args.check:
            health = run_check()
            health['checked_at'] = time.time()
            if args.health_output:
                atomic_json(args.health_output, health)
            print(json.dumps(health), flush=True)
            return 0
        event('INIT')
        try:
            runtime = runtime_metadata()
        except (ImportError, OSError) as error:
            raise WorkerError('VDBFUSION_NOT_INSTALLED',
                              f'VDBFusion environment cannot import the native library: {error}') from error
        metadata.update(runtime)
        if args.prepare:
            if not (args.bag and args.trajectory and args.prepare_output):
                raise WorkerError('SETTINGS_INVALID', '--prepare requires --bag, --trajectory and --prepare-output')
            if args.prepare_output.exists():
                raise WorkerError('MESH_INVALID', 'Preparation output exists; choose a new run directory')
            event('VALIDATING_SOURCE')
            trajectory, trajectory_report = V.load_trajectory(args.trajectory)
            resolved = V.resolve_settings(arg_settings(args))
            # Advanced controls arrive as their own arguments, so record the values this
            # preparation actually used: the preparation identity is what a later
            # reconstruction is compared against, and an omitted key must never read as a
            # silent mismatch.
            for key, value in (('association_spacing_multiplier', args.association_spacing_multiplier),
                               ('boundary_margin_m', args.boundary_margin),
                               ('unsupported_observations', args.unsupported_policy),
                               ('mask_deleted_triangles', args.mask_deleted_triangles == '1')):
                if value is not None:
                    resolved[key] = value
            edit_reference = load_edit_reference(args.edited_workspace, args.association_radius,
                                                args.association_spacing_multiplier)
            extent, estimated_points = sample_bag_extent(args.bag, trajectory, args.topic,
                                                voxel_size_m=resolved['voxel_size_m'])
            bag_bytes = sum(p.stat().st_size for p in args.bag.rglob('*') if p.is_file())
            event('PREFLIGHT')
            clipped = roi_clipped_bbox(resolved, (extent['observed_bbox_min_m'], extent['observed_bbox_max_m']))
            preflight = V.preflight(args.prepare_output, resolved, observed_points=estimated_points,
                                    observed_bbox=clipped, bag_bytes=bag_bytes,
                                    occupancy=extent.get('occupancy'),
                                    reference_points=(edit_reference.metadata.get('reference_points')
                                                      if edit_reference else None))
            if not preflight['ok']:
                raise WorkerError('RESOURCE_PREFLIGHT_FAILED', '; '.join(
                    preflight['failures'] + preflight['suggestions']))
            source = V.source_identity(args.bag, args.trajectory, args.topic, args.edited_workspace,
                                       edit_fingerprints=dict(edit_id=Path(args.edited_workspace).name)
                                       if args.edited_workspace else None)
            prepared = dict(
                version=1, algorithm='vdbfusion', engine='vdbfusion', state='PREPARED',
                created_at=time.time(), python=sys.executable, runtime=runtime,
                trajectory=str(args.trajectory), trajectory_report=trajectory_report,
                bag=str(args.bag), topic=args.topic, bag_scan=extent, bag_bytes=bag_bytes,
                settings=resolved, preflight=preflight,
                identity=V.preparation_identity(source, resolved),
                edited_geometry=dict(edit_reference.metadata) if edit_reference else None,
                edit_filter_accuracy='validated_approximate' if edit_reference else None,
                preparation_note='VDBFusion preparation validates sources and resources only; raw observations are '
                                 'streamed during mesh reconstruction and no prepared point cloud is written.',
                elapsed_seconds=time.monotonic() - started)
            atomic_json(args.prepare_output, prepared)
            event('COMPLETED', **{'prepared': str(args.prepare_output)})
            print(json.dumps(prepared), flush=True)
            return 0
        if not (args.bag and args.trajectory and args.output):
            raise WorkerError('SETTINGS_INVALID', '--bag, --trajectory and --output are required')
        settings = dict(arg_settings(args))
        settings.update(bag=args.bag, trajectory=args.trajectory, output=args.output, topic=args.topic,
                        edited_workspace=args.edited_workspace, association_radius_m=args.association_radius,
                        association_spacing_multiplier=args.association_spacing_multiplier,
                        boundary_margin_m=args.boundary_margin, unsupported_policy=args.unsupported_policy,
                        mask_deleted_triangles=args.mask_deleted_triangles == '1',
                        memory_budget_bytes=V.resolve_memory_budget_bytes(dict(arg_settings(args))),
                        chunk_size=args.chunk_size, _started=started)
        metadata.update(reconstruct(settings, event, check_cancel))
        path = args.metadata or Path(args.output).parent / 'vdbfusion_metadata.json'
        atomic_json(path, metadata)
        event('COMPLETED', vertices=metadata.get('vertex_count'), faces=metadata.get('face_count'),
              mesh_output_mode=settings['mesh_output_mode'])
        print(json.dumps({'state': 'COMPLETED', 'vertex_count': metadata.get('vertex_count'),
                          'face_count': metadata.get('face_count')}), flush=True)
        return 0
    except Exception as error:  # noqa: BLE001 - every failure is classified and reported
        code = classify(error, stage)
        details = getattr(error, 'details', {}) or {}
        message = str(error) or error.__class__.__name__
        if code not in ('CANCELLED', 'MEMORY_BUDGET_EXCEEDED', 'VDBFUSION_NOT_INSTALLED'):
            print(traceback.format_exc(), file=sys.stderr, flush=True)
        failure = dict(stage=stage, error_type=code, message=message, **details)
        print(json.dumps(failure), file=sys.stderr, flush=True)
        if args.progress:
            atomic_json(args.progress, failure)
        if args.health_output:
            atomic_json(args.health_output, dict(status='FAILED', error_type=code, message=message,
                                                checked_at=time.time(), python=sys.executable))
        return 2 if code == 'CANCELLED' else 1


if __name__ == '__main__':
    sys.exit(main())
