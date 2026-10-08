"""VDBFusion TSDF surface reconstruction: bounded streaming, real sensor origins.

This module is the single source of truth for the VDBFusion engine. It imports
NumPy and the standard library only; ROS bag reading, the native VDBFusion
library, SciPy and PLY writing are imported lazily so the module stays importable
from the web backend, the preparation subprocess and the isolated worker.

Nothing here touches NKSR, Torch, CUDA or any existing GLIM export. World
coordinates are always metres and every sensor origin comes from the GLIM
trajectory interpolation in :mod:`factory_mapping.reconstruction`.
"""
import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from pathlib import Path
import numpy as np

# TSDF settings. Truncation is an independent control (GATE 6): the defaults and
# the presets keep the upstream VDBFusion 3x voxel ratio, but a user may move
# truncation on its own within the physical bounds validated below.
DEFAULT_VOXEL_SIZE_M = 0.02
DEFAULT_SDF_TRUNC_M = 0.06
DEFAULT_SPACE_CARVING = False
PRESETS = {
    'fast': dict(voxel_size_m=0.02, sdf_trunc_m=0.06),
    'detailed': dict(voxel_size_m=0.01, sdf_trunc_m=0.03),
    'experimental': dict(voxel_size_m=0.005, sdf_trunc_m=0.015),
}
PRESET_NOTES = {
    'fast': 'Fast TSDF for large factory scans.',
    'detailed': 'Detailed TSDF for a 10-50 m scene.',
    'experimental': 'Experimental 5 mm TSDF. Whole-factory scans can need very large memory and long runtimes.',
}
MIN_VOXEL_SIZE_M = 0.001
MAX_VOXEL_SIZE_M = 1.0
MAX_SDF_TRUNC_M = 5.0

MESH_OUTPUT_MODES = ('merged', 'chunks', 'both')
# Raw bag observations are read in bounded batches; no full-cloud allocation exists.
DEFAULT_BATCH_POINTS = 2_000_000
MIN_BATCH_POINTS = 10_000
MAX_BATCH_POINTS = 20_000_000
# Motion-aware origin grouping consumes at most this multiple of a group length.
GROUP_WINDOW_FACTOR = 8
# Soft memory budget for the whole worker process, in bytes (GATE 5).
DEFAULT_MEMORY_BUDGET_BYTES = 24 * 1024 ** 3
# Conservative order-of-magnitude cost of one touched TSDF voxel: a distance value,
# a weight and the amortized OpenVDB tree overhead around each leaf.
TSDF_BYTES_PER_TOUCHED_VOXEL = 32
# Reference geometry for edited-map filtering (GATE 7).
DEFAULT_ASSOCIATION_SPACING_MULTIPLIER = 4.0
DEFAULT_ASSOCIATION_SPACING_SAMPLE = 200_000
DEFAULT_REFERENCE_SAMPLING_M = 0.1
MIN_ASSOCIATION_RADIUS_M = 0.02
MAX_REFERENCE_INDEX_BYTES = 5 * 1024 ** 3
# Resource preflight model (GATE 7). Every figure below is a documented heuristic,
# not a promise; the estimate is reported component by component with the safety
# margin that was applied, so a reader can judge it.
PREFLIGHT_SAFETY_MARGIN = 1.3
PREFLIGHT_RAM_RESERVE_FRACTION = 0.15
PREFLIGHT_MIN_RAM_RESERVE_BYTES = 2 * 1024 ** 3
# Permanent bytes per reference point: world f64 (24 B) + kept mask (4 B) plus the
# kept/removed KD-tree index amortized over the reference set.
REFERENCE_BYTES_PER_POINT = 24 + 4 + 96
# One bounded input batch: world f64 (24 B) + origins f64 (24 B) + intensity f32 (4 B)
# + timestamps f64 (8 B) per observation.
BATCH_BYTES_PER_OBSERVATION = 60
# Extraction keeps faces as int64 (24 B/triangle) and about 0.5 vertices per triangle
# as float64 (12 B), before masking compacts both.
MESH_BYTES_PER_TRIANGLE = 40
# Binary PLY: int32 indices plus float32 vertices, and the writer builds one PlyData
# copy that is then independently re-read for validation.
PLY_BYTES_PER_TRIANGLE = 32
PLY_TRANSIENT_FACTOR = 3
# A triangle probe allocates 6 float64 samples per candidate triangle, bounded per batch.
MASK_PROBE_BYTES_PER_TRIANGLE = 6 * 24


def validate_voxel_size(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('TSDF voxel size must be a positive, finite size in meters')
    if value < MIN_VOXEL_SIZE_M or value > MAX_VOXEL_SIZE_M:
        raise ValueError(f'TSDF voxel size must be between {MIN_VOXEL_SIZE_M:g} m and {MAX_VOXEL_SIZE_M:g} m')
    return value


def validate_sdf_trunc(value, voxel_size_m=None):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('TSDF truncation distance must be a positive, finite distance in meters')
    if value > MAX_SDF_TRUNC_M:
        raise ValueError(f'TSDF truncation distance must not exceed {MAX_SDF_TRUNC_M:g} m')
    if voxel_size_m is not None and value < validate_voxel_size(voxel_size_m):
        raise ValueError('TSDF truncation distance must be at least the voxel size, otherwise no surface band exists')
    return value


def validate_space_carving(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('Space carving must be a boolean')
    return bool(value)


def validate_origin_error_budget(value, voxel_size_m):
    """The origin grouping tolerance is tied to the TSDF resolution (GATE 4)."""
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Motion-aware origin budget must be a positive, finite distance in meters')
    if value > validate_voxel_size(voxel_size_m):
        raise ValueError('Motion-aware origin budget must not exceed one TSDF voxel')
    return value


def validate_batch_points(value):
    value = int(value)
    if value < MIN_BATCH_POINTS or value > MAX_BATCH_POINTS:
        raise ValueError(f'Batch points must be between {MIN_BATCH_POINTS:,} and {MAX_BATCH_POINTS:,}')
    return value


def validate_mesh_output_mode(value):
    if value not in MESH_OUTPUT_MODES:
        raise ValueError(f'Mesh output mode must be one of {", ".join(MESH_OUTPUT_MODES)}')
    return value


class ReconstructionCancelled(RuntimeError):
    """Cooperative cancellation was requested while a bounded stage was running.

    Raised from :func:`check_cancellation`, which every long stage polls, so a
    cancellation can land inside mesh extraction or output writing instead of only
    between integration batches. The worker maps it to the ``CANCELLED`` state.
    """

    code = 'CANCELLED'


def check_cancellation(cancel):
    """Poll a cooperative cancellation callback and raise when it reports a stop.

    ``cancel`` may either return a boolean or raise its own cancellation error;
    both are accepted so the library and the worker can share one contract.
    """
    if cancel is None:
        return
    if cancel():
        raise ReconstructionCancelled('VDBFusion reconstruction cancelled')


def validate_roi(minimum, maximum):
    """Optional world-space ROI box. Both bounds are required together, in metres."""
    if minimum is None and maximum is None:
        return None, None
    if minimum is None or maximum is None:
        raise ValueError('A region of interest needs both a minimum and a maximum corner')
    lower = np.asarray(minimum, dtype=np.float64).reshape(-1)
    upper = np.asarray(maximum, dtype=np.float64).reshape(-1)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError('Region-of-interest corners must have three world coordinates in meters')
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError('Region-of-interest corners must be finite world coordinates in meters')
    if np.any(upper <= lower):
        raise ValueError('Region-of-interest maximum must be strictly greater than its minimum on every axis')
    return lower.tolist(), upper.tolist()


# Settings that change *what* a run produces. They are part of the preparation identity:
# a reconstruction with different values is a different job and must be prepared again.
SEMANTIC_SETTING_KEYS = (
    'preset', 'voxel_size_m', 'sdf_trunc_m', 'space_carving', 'origin_error_budget_m',
    'roi_min_m', 'roi_max_m', 'unsupported_observations', 'mask_deleted_triangles',
    'boundary_margin_m', 'association_spacing_multiplier',
)
# Settings that only change *how* a run executes or is written out. They may be changed
# between preparation and reconstruction without invalidating the prepared identity.
EXECUTION_SETTING_KEYS = ('batch_points', 'memory_budget_gib', 'mesh_output_mode', 'chunk_size')
PREPARATION_IDENTITY_VERSION = 1


def resolve_memory_budget_bytes(settings, override=None):
    """The one byte-valued memory limit used by every layer.

    The UI/API control is ``memory_budget_gib`` (gigabytes, as the field name says), the
    worker CLI passes gigabytes and the engine works in bytes. Reading only
    ``memory_budget_bytes`` silently ignored the configured budget - the memory supervisor
    received 0 and never escalated, and the preflight fell back to the default. Precedence
    is: explicit ``override`` (the worker's own resolved limit) > ``memory_budget_bytes`` >
    ``memory_budget_gib`` > :data:`DEFAULT_MEMORY_BUDGET_BYTES`.

    A non-positive, non-finite or non-numeric configured value is an error rather than a
    silent fall back to the default, so a malformed budget can never widen the limit.
    """
    for value, unit in ((override, 1), (settings.get('memory_budget_bytes'), 1),
                        (settings.get('memory_budget_gib'), 1024 ** 3)):
        if value is None:
            continue
        try:
            number = float(value) * unit
        except (TypeError, ValueError) as error:
            raise ValueError(f'Memory budget must be a number of bytes or GiB, not {value!r}') from error
        if not math.isfinite(number) or number <= 0:
            raise ValueError('Memory budget must be a positive, finite amount of memory')
        return int(number)
    return DEFAULT_MEMORY_BUDGET_BYTES


def canonical_settings(settings, keys=SEMANTIC_SETTING_KEYS):
    """Canonically typed semantic subset, so a fingerprint is deterministic."""
    canonical = {}
    for key in keys:
        value = (settings or {}).get(key)
        if value is None or isinstance(value, bool):
            canonical[key] = value
        elif isinstance(value, (list, tuple)):
            canonical[key] = [float(item) for item in value]
        elif isinstance(value, (int, float)):
            canonical[key] = float(value)
        else:
            canonical[key] = str(value)
    return canonical


def settings_fingerprint(settings, keys=SEMANTIC_SETTING_KEYS):
    """Stable SHA-256 fingerprint of the semantic settings."""
    payload = json.dumps(canonical_settings(settings, keys), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def compare_semantic_settings(prepared, current):
    """Compare the settings a run was prepared with against the current request.

    Execution-only settings (``batch_points``, memory budget, mesh output mode, chunk
    size) are excluded: they change how the run executes, not what it produces. A value the
    request omits is not a mismatch, it means "use the prepared value" - which is also what
    the reconstruction then runs with, so the prepared resolution can never be silently
    replaced by a different one.
    """
    prepared_canonical = canonical_settings(prepared)
    requested_canonical = canonical_settings(current)
    differing = []
    for key in SEMANTIC_SETTING_KEYS:
        if (current or {}).get(key) is None:
            continue
        if requested_canonical[key] != prepared_canonical[key]:
            differing.append(key)
    return dict(matches=not differing, differing=differing,
                detail={key: dict(prepared=prepared_canonical[key], requested=requested_canonical[key])
                        for key in differing},
                prepared=prepared_canonical, requested=requested_canonical)


def effective_settings(prepared, current):
    """Prepared semantic settings with the request's execution-only choices applied."""
    effective = dict(prepared)
    for key in EXECUTION_SETTING_KEYS:
        if (current or {}).get(key) is not None:
            effective[key] = current[key]
    return effective


def source_identity(bag, trajectory, topic, edited_workspace=None, edit_fingerprints=None):
    """Pin the prepared source so a later reconstruction can prove it is unchanged.

    The raw bag is read-only after capture and may be many gigabytes, so it is pinned by
    its byte total, file count and metadata mtime rather than a full hash; the small
    trajectory text file is hashed in full. This detects a replaced, re-recorded or
    trimmed source, which is what "the prepared input is stale" means in practice.
    """
    bag = Path(bag)
    trajectory = Path(trajectory)
    bag_files = [path for path in bag.rglob('*') if path.is_file()]
    metadata = bag / 'metadata.yaml'
    identity = dict(
        bag=str(bag), bag_files=len(bag_files), topic=str(topic),
        bag_bytes=int(sum(path.stat().st_size for path in bag_files)),
        bag_metadata_bytes=int(metadata.stat().st_size) if metadata.is_file() else None,
        bag_metadata_mtime=float(metadata.stat().st_mtime) if metadata.is_file() else None,
        trajectory=str(trajectory),
        trajectory_bytes=int(trajectory.stat().st_size) if trajectory.is_file() else None,
        trajectory_sha256=(hashlib.sha256(trajectory.read_bytes()).hexdigest()
                           if trajectory.is_file() else None),
        edited_workspace=str(edited_workspace) if edited_workspace is not None else None,
    )
    for key, value in (edit_fingerprints or {}).items():
        identity[f'edit_{key}'] = value
    payload = json.dumps(identity, sort_keys=True, separators=(',', ':'))
    identity['fingerprint'] = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    return identity


def compare_source_identity(prepared_source, current_source):
    """Diff two source identities, including the combined fingerprint."""
    keys = sorted(set(prepared_source) | set(current_source))
    differing = [key for key in keys if prepared_source.get(key) != current_source.get(key)]
    return dict(matches=not differing, differing=differing,
                detail={key: dict(prepared=prepared_source.get(key), current=current_source.get(key))
                        for key in differing})


def preparation_identity(source, settings):
    """The canonical identity a prepared VDBFusion run is pinned to."""
    return dict(version=PREPARATION_IDENTITY_VERSION, source=source,
                semantic=canonical_settings(settings),
                semantic_fingerprint=settings_fingerprint(settings),
                execution=canonical_settings(settings, EXECUTION_SETTING_KEYS),
                note='Semantic settings and source files are pinned at preparation. Execution-only settings '
                     '(batch size, memory budget, mesh output mode, chunk size) may change without re-preparing.')


def resolve_settings(settings, origin_error_budget_m=None):
    """Validate one VDBFusion request. Rejects NaN, negative and reversed values."""
    if not isinstance(settings, dict):
        raise ValueError('VDBFusion settings must be an object')
    preset = settings.get('preset')
    if preset is not None and preset not in PRESETS:
        raise ValueError(f'Unknown VDBFusion preset {preset!r}')
    base = dict(PRESETS[preset]) if preset else {}

    def field(name, fallback):
        value = settings.get(name)
        return fallback if value is None else value

    voxel_size_m = validate_voxel_size(field('voxel_size_m', base.get('voxel_size_m', DEFAULT_VOXEL_SIZE_M)))
    sdf_trunc_m = validate_sdf_trunc(field('sdf_trunc_m', base.get('sdf_trunc_m', DEFAULT_SDF_TRUNC_M)), voxel_size_m)
    result = dict(
        preset=preset,
        voxel_size_m=voxel_size_m,
        sdf_trunc_m=sdf_trunc_m,
        space_carving=validate_space_carving(field('space_carving', DEFAULT_SPACE_CARVING)),
        mesh_output_mode=validate_mesh_output_mode(field('mesh_output_mode', 'merged')),
        batch_points=validate_batch_points(field('batch_points', DEFAULT_BATCH_POINTS)),
    )
    budget = origin_error_budget_m if origin_error_budget_m is not None else settings.get('origin_error_budget_m')
    result['origin_error_budget_m'] = validate_origin_error_budget(
        voxel_size_m if budget is None else budget, voxel_size_m)
    result['roi_min_m'], result['roi_max_m'] = validate_roi(settings.get('roi_min_m'), settings.get('roi_max_m'))
    return result


def load_trajectory(path):
    """Read and validate a GLIM ``traj_lidar.txt`` (timestamp + XYZW quaternion)."""
    trajectory = np.loadtxt(path, ndmin=2)
    if trajectory.ndim != 2 or trajectory.shape[1] != 8:
        raise ValueError('Trajectory requires eight columns: timestamp and XYZW pose')
    if len(trajectory) < 2:
        raise ValueError('Trajectory requires at least two poses')
    if not np.isfinite(trajectory).all():
        raise ValueError('Trajectory contains non-finite values')
    stamps = trajectory[:, 0]
    if np.any(np.diff(stamps) <= 0):
        raise ValueError('Trajectory timestamps must increase strictly')
    norms = np.linalg.norm(trajectory[:, 4:8], axis=1)
    if np.any(norms == 0):
        raise ValueError('Trajectory contains a zero quaternion')
    # Normalize in place with an explicit report instead of silently trusting input.
    report = dict(poses=len(trajectory), quaternion_norm_min=float(norms.min()),
                  quaternion_norm_max=float(norms.max()),
                  t_start=float(stamps[0]), t_end=float(stamps[-1]),
                  duration_s=float(stamps[-1] - stamps[0]))
    trajectory[:, 4:8] /= norms[:, None]
    report['quaternion_normalized'] = bool(np.any(np.abs(norms - 1.0) > 1e-9))
    translation = trajectory[:, 1:4]
    report['path_length_m'] = float(np.linalg.norm(np.diff(translation, axis=0), axis=1).sum())
    report['bbox_min_m'] = translation.min(axis=0).tolist()
    report['bbox_max_m'] = translation.max(axis=0).tolist()
    return trajectory, report


def motion_aware_groups(origins, budget_m):
    """Split origins into maximal contiguous runs around one real trajectory origin.

    ``origins`` holds one world-space sensor origin per observation, in time
    order. Every observation is assigned to exactly one group; nothing is
    discarded. A group's representative is the actual interpolated trajectory
    origin of its first observation, and the returned error is the largest
    distance from that representative inside the group, which never exceeds
    ``budget_m``.
    """
    budget_m = float(budget_m)
    count = len(origins)
    start = 0
    window = 1
    while start < count:
        stop_limit = min(count, start + GROUP_WINDOW_FACTOR * window)
        block = origins[start:stop_limit]
        offsets = np.linalg.norm(block - block[0], axis=1)
        cumulative = np.maximum.accumulate(offsets)
        # searchsorted needs a nondecreasing sequence; `cumulative` is one by construction.
        length = int(np.searchsorted(cumulative, budget_m, side='right'))
        length = max(length, 1)
        yield start, start + length, block[0], float(cumulative[length - 1])
        window = length
        start += length


def max_origin_error(origins, budget_m):
    """Largest realized approximation error of :func:`motion_aware_groups`."""
    worst = 0.0
    groups = 0
    for _, _, _, error in motion_aware_groups(origins, budget_m):
        worst = max(worst, error)
        groups += 1
    return worst, groups


def batch_world_observations(frames, trajectory, batch_points, counters=None, progress=None, cancel=None,
                             timings=None):
    """Group decoded LiDAR frames into bounded world-space observation batches.

    ``frames`` yields ``(xyz, intensity, timestamps)`` in recorded order. Reuses the
    proven GLIM trajectory interpolation from :mod:`factory_mapping.reconstruction`
    for every point, so each observation gets its own interpolated sensor origin.

    ``batch_points`` is a hard bound on every yielded batch, not an average: a single
    LiDAR frame larger than the bound is split across several batches in recorded
    order, observations are never dropped, reordered or duplicated at a slice
    boundary, and the trailing partial batch is always emitted. ``counters`` records
    ``batches``, ``split_frames`` and ``max_batch_points`` so the bound can be checked
    from run metadata. Observations outside the trajectory time range are counted in
    ``counters`` and never silently clamped. ``timings`` accumulates per-phase
    seconds, which is what the benchmark CLI reports.
    """
    from .reconstruction import transform_points
    batch_points = validate_batch_points(batch_points)
    counters = counters if counters is not None else {}
    timings = timings if timings is not None else {}
    for key in ('frames', 'observations', 'outside_trajectory', 'batches', 'split_frames'):
        counters.setdefault(key, 0)
    counters.setdefault('max_batch_points', 0)
    for key in ('transform_seconds', 'batch_seconds'):
        timings.setdefault(key, 0.0)
    buffers = {key: [] for key in ('points', 'origins', 'intensity', 'timestamps')}

    def count_batch(length):
        counters['batches'] += 1
        counters['max_batch_points'] = max(counters['max_batch_points'], int(length))

    def drain():
        """Concatenate the buffered frames into one batch, or ``None`` when empty."""
        if not buffers['points']:
            return None
        started_batch = time.perf_counter()
        batch = {key: (np.concatenate(values) if len(values) > 1 else values[0])
                 for key, values in buffers.items()}
        timings['batch_seconds'] += time.perf_counter() - started_batch
        count_batch(len(batch['points']))
        for values in buffers.values():
            values.clear()
        return batch

    def window(values, start, stop):
        count_batch(stop - start)
        return {key: value[start:stop] for key, value in zip(buffers, values)}

    pending = 0
    for xyz, intensity, timestamps in frames:
        if cancel is not None:
            cancel()
        counters['frames'] += 1
        started = time.perf_counter()
        points, origins, valid = transform_points(xyz, timestamps, trajectory)
        timings['transform_seconds'] += time.perf_counter() - started
        # Observations outside the trajectory time range are counted and reported,
        # never silently clamped to the first or last pose.
        counters['outside_trajectory'] += int(len(valid) - np.count_nonzero(valid))
        if len(points):
            counters['observations'] += len(points)
            values = (points, origins, intensity[valid], timestamps[valid])
            if len(points) > batch_points:
                # A single frame larger than the bound is emitted, in recorded order, as
                # several batches. batch_points is the memory-safety limit for *every*
                # batch, not an average: no batch ever exceeds it, no observation is
                # dropped or duplicated at a slice boundary, and the final partial slice
                # is still emitted.
                counters['split_frames'] += 1
                buffered = drain()
                if buffered is not None:
                    yield buffered
                for start in range(0, len(points), batch_points):
                    yield window(values, start, min(start + batch_points, len(points)))
                pending = 0
            else:
                if pending and pending + len(points) > batch_points:
                    yield drain()
                    pending = 0
                for key, value in zip(buffers, values):
                    buffers[key].append(value)
                pending += len(points)
                if pending >= batch_points:
                    yield drain()
                    pending = 0
        if counters['frames'] % 200 == 0 and progress is not None:
            progress(f'Reading raw bag: {counters["frames"]:,} LiDAR frames, '
                     f'{counters["observations"]:,} world observations')
    if pending:
        yield drain()
    if not counters['frames']:
        raise ValueError('The raw bag has no LiDAR messages on the selected topic')


def iter_world_batches(bag, trajectory, topic, batch_points, counters=None, progress=None, cancel=None,
                       timings=None):
    """Stream a ROS 2 sqlite3 bag into bounded world-space observation batches."""
    from .reconstruction import get_cloud_arrays
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2

    timings = timings if timings is not None else {}
    for key in ('bag_read_seconds', 'decode_seconds'):
        timings.setdefault(key, 0.0)

    def frames():
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='sqlite3'),
                    rosbag2_py.ConverterOptions('', ''))
        while reader.has_next():
            started = time.perf_counter()
            name, data, _ = reader.read_next()
            if name != topic:
                continue
            timings['bag_read_seconds'] += time.perf_counter() - started
            started = time.perf_counter()
            payload = get_cloud_arrays(deserialize_message(data, PointCloud2))
            timings['decode_seconds'] += time.perf_counter() - started
            yield payload

    return batch_world_observations(frames(), trajectory, batch_points, counters=counters,
                                   progress=progress, cancel=cancel, timings=timings)


class EditedGeometryReference:
    """Retained/deleted classification from one verified map-editor cleanup.

    Saved GLIM submaps are byte-exact subsets of the pre-edit submaps they were
    copied from, which is verified here rather than assumed. Because GLIM merges
    keyframes into 0.1 m voxels, the retained set is a sparse sampling of the raw
    observations, so the association radius is measured from the reference
    geometry instead of being a fixed guess. This strategy is an APPROXIMATE
    spatial association; it is not an exact edit transfer and never claims to be.
    """

    VALIDATION = 'validated_approximate'

    def __init__(self, world, retained, sampling_resolution_m, submaps, association_spacing_multiplier,
                 association_radius_m=None, retained_world=None):
        self.world = world
        self.retained = retained
        self.sampling_resolution_m = sampling_resolution_m
        self.submaps = submaps
        self.association_spacing_multiplier = association_spacing_multiplier
        self.association_radius_m = association_radius_m
        self._retained_world = retained_world
        self._tree_all = None
        self._tree_retained = None
        self._tree_deleted = None

    @property
    def retained_points(self):
        if self._retained_world is None:
            self._retained_world = self.world[self.retained]
        return self._retained_world

    def trees(self):
        from scipy.spatial import cKDTree
        if self._tree_all is None:
            self._tree_all = cKDTree(self.world, compact_nodes=True, balanced_tree=True)
        if self._tree_retained is None:
            self._tree_retained = cKDTree(self.retained_points, compact_nodes=True, balanced_tree=True)
        if self._tree_deleted is None and np.any(~self.retained):
            self._tree_deleted = cKDTree(self.world[~self.retained], compact_nodes=True, balanced_tree=True)
        return self._tree_all, self._tree_retained, self._tree_deleted

    def resolve_association_radius(self, spacing_multiplier=None):
        """Derive the association radius from the measured reference sampling spacing.

        The radius is measured, never a fixed 5/10/20 cm guess, and is reported in
        the run metadata so a reader can judge the boundary uncertainty band.
        """
        if self.association_radius_m is not None:
            return self.association_radius_m
        multiplier = self.association_spacing_multiplier if spacing_multiplier is None else float(spacing_multiplier)
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError('Association spacing multiplier must be a positive, finite number')
        radius = multiplier * self.sampling_resolution_m
        self.association_radius_m = float(max(radius, MIN_ASSOCIATION_RADIUS_M))
        return self.association_radius_m

    CLASS_KEEP = 0
    CLASS_REMOVE = 1
    CLASS_UNSUPPORTED = 2
    CLASS_AMBIGUOUS = 3
    CLASS_LABELS = ('KEEP', 'REMOVE', 'UNSUPPORTED', 'AMBIGUOUS')

    def classify(self, points, association_radius_m, boundary_margin_m=0.0, ambiguity_band_m=None):
        """Classify world coordinates against verified kept and removed geometry.

        Returns a dict of boolean masks. A coordinate is kept when verified kept
        geometry is within the association radius and the nearest removed sample is
        not at least as close (minus ``boundary_margin_m``, which biases the edit
        boundary toward exclusion). Coordinates without kept support inside the
        radius are ``unsupported`` and are excluded by default rather than guessed.

        ``ambiguous`` marks coordinates whose nearest kept and nearest removed samples
        are both inside the radius and within ``ambiguity_band_m`` of each other: they
        are inside the documented boundary uncertainty band, so their membership is
        decided by the documented policy (kept unless removed geometry dominates) and
        reported instead of being presented as a confident classification. The band
        defaults to the measured reference sampling resolution, which is the resolution
        limit of the GLIM submap samples themselves; it never widens the association
        radius. ``class_of`` maps the mask dict to ``CLASS_*`` labels for reporting.
        """
        _, tree_retained, tree_deleted = self.trees()
        count = len(points)
        if not count:
            empty = np.zeros(0, dtype=bool)
            return dict(keep=empty, unsupported=empty, removed_dominated=empty, removed_support=empty,
                        ambiguous=empty)
        band = float(self.sampling_resolution_m if ambiguity_band_m is None else ambiguity_band_m)
        upper = np.nextafter(float(association_radius_m), math.inf)
        near_retained, _ = tree_retained.query(points, k=1, distance_upper_bound=upper, workers=1)
        retained_support = np.isfinite(near_retained) & (near_retained <= association_radius_m)
        unsupported = ~retained_support
        if tree_deleted is None:
            return dict(keep=retained_support, unsupported=unsupported,
                        removed_dominated=np.zeros(count, dtype=bool),
                        removed_support=np.zeros(count, dtype=bool),
                        ambiguous=np.zeros(count, dtype=bool))
        near_removed, _ = tree_deleted.query(points, k=1, distance_upper_bound=upper, workers=1)
        removed_support = np.isfinite(near_removed) & (near_removed <= association_radius_m)
        removed_dominated = retained_support & removed_support & (near_retained + float(boundary_margin_m) >= near_removed)
        # Inside the documented uncertainty band, and not already decided as removed. The
        # difference is only evaluated where both neighbours exist, so unsupported
        # coordinates never produce a meaningless infinity-minus-infinity.
        both = retained_support & removed_support
        margin = np.full(count, np.inf, dtype=np.float64)
        np.subtract(near_removed, near_retained, out=margin, where=both)
        ambiguous = both & ~removed_dominated & (margin <= band)
        return dict(keep=retained_support & ~removed_dominated, unsupported=unsupported,
                    removed_dominated=removed_dominated, removed_support=removed_support,
                    ambiguous=ambiguous)

    @classmethod
    def class_of(cls, classified):
        """Label each classified coordinate as KEEP, REMOVE, UNSUPPORTED or AMBIGUOUS."""
        labels = np.full(len(classified['keep']), cls.CLASS_KEEP, dtype=np.int8)
        labels[classified['removed_dominated']] = cls.CLASS_REMOVE
        labels[classified['unsupported']] = cls.CLASS_UNSUPPORTED
        labels[classified['ambiguous'] & classified['keep']] = cls.CLASS_AMBIGUOUS
        return labels


SUBMAP_ROW_BYTES = 12
# Rotation blocks from GLIM are text-parsed; this tolerance only sanity-checks the
# frame, it is never used to establish source identity (that is bit-exact).
SUBMAP_FRAME_TOLERANCE = 1e-6


class SavedSubmapError(ValueError):
    """Saved cleanup geometry is not a provably valid subset of its pre-edit source.

    The pre-edit (``map_01``) and saved (``saved_map``) submaps are compared as
    multisets of exact float32 rows. A point the pre-edit submap never contained, or
    a saved multiplicity above the original multiplicity, is rejected outright rather
    than silently reinterpreted as removed geometry.

    Measurements on the real saved cleanup ``edit_ef16d03e7f86`` (57 submaps):
    ``data.txt`` is byte-identical between ``map_01`` and ``saved_map``, every saved
    submap is an exact multiplicity-valid subset of its pre-edit submap, and every
    saved submap preserves the original point order. The contract also holds that the
    first ``id:`` line of ``data.txt`` equals the submap directory index.
    """

    code = 'INVALID_SAVED_SUBMAP'

    def __init__(self, edit_id, submap_id, mismatch, invalid_points, action=None):
        self.edit_id = edit_id
        self.submap_id = submap_id
        self.mismatch = mismatch
        self.invalid_points = int(invalid_points)
        self.action = action or ('Re-export the Clean Map workspace from the native map editor so that '
                                 'saved_map is a strict subset of map_01, then prepare the job again.')
        super().__init__(
            f'{self.code}: edit {edit_id or "unknown"} submap {submap_id or "unknown"}: {mismatch} '
            f'({self.invalid_points:,} invalid saved points). {self.action}')


def submap_block_bytes(path):
    """Read one ``points_compact.bin`` as its exact bytes.

    Identity is compared on file bytes, so a saved point can only ever match the
    pre-edit point it was copied from. ``'<f4'`` is used explicitly for values, which
    keeps the decoded coordinates correct on any host endianness.
    """
    try:
        raw = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise ValueError(f'Saved cleanup submap {Path(path).parent.name} has no readable point block') from error
    if raw.size == 0 or raw.size % SUBMAP_ROW_BYTES:
        raise ValueError(f'Saved cleanup submap {Path(path).parent.name} has an invalid point block '
                         f'({raw.size} bytes is not a whole number of 3 x float32 rows)')
    return raw


def submap_rows(raw):
    """Float32 coordinates and exact byte keys for one raw point block."""
    return raw.view('<f4').reshape(-1, 3), raw.view(np.dtype((np.void, SUBMAP_ROW_BYTES))).ravel()


def _read_submap_points(path):
    """Backwards-compatible coordinate reader used by tests and diagnostics."""
    rows, _ = submap_rows(submap_block_bytes(path))
    if not np.isfinite(rows).all():
        raise ValueError(f'Saved cleanup submap {Path(path).parent.name} contains non-finite points')
    return rows


def _read_submap_origin(data_txt, submap_id=None):
    """Parse the submap frame: the leading ``id:`` and ``T_world_origin``.

    The frame is validated structurally (rigid rotation, bottom row) so an
    inconsistent or corrupt transform is reported instead of being used.
    """
    try:
        lines = Path(data_txt).read_text().splitlines()
    except OSError as error:
        raise ValueError(f'Saved cleanup submap {submap_id or "?"} has no readable data.txt') from error
    declared = None
    for line in lines[:4]:
        match = re.match(r'^id:\s*(\d+)\s*$', line)
        if match:
            declared = int(match.group(1))
            break
    matrix = None
    for index, line in enumerate(lines):
        if line.startswith('T_world_origin'):
            try:
                rows = [np.fromstring(lines[index + 1 + k], sep=' ') for k in range(4)]
            except IndexError as error:
                raise ValueError(f'Saved cleanup submap {submap_id or "?"} has a truncated T_world_origin') from error
            matrix = np.vstack(rows)
            break
    if matrix is None:
        raise ValueError(f'Saved cleanup submap {submap_id or "?"} has no T_world_origin')
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f'Saved cleanup submap {submap_id or "?"} has an invalid T_world_origin')
    if submap_id is not None and declared is not None:
        try:
            expected = int(submap_id)
        except (TypeError, ValueError):
            expected = None
        if expected is not None and declared != expected:
            raise ValueError(
                f'Saved cleanup submap {submap_id} declares id {declared} in data.txt; the submap set is inconsistent')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=SUBMAP_FRAME_TOLERANCE) or \
            not np.isclose(np.linalg.det(rotation), 1.0, atol=SUBMAP_FRAME_TOLERANCE):
        raise ValueError(f'Saved cleanup submap {submap_id or "?"} has a non-rigid T_world_origin rotation')
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=SUBMAP_FRAME_TOLERANCE):
        raise ValueError(f'Saved cleanup submap {submap_id or "?"} has an invalid T_world_origin bottom row')
    return matrix


def validate_submap_pair(original_raw, saved_raw, edit_id, submap_id):
    """Prove saved geometry is a multiplicity-valid subset of the pre-edit submap.

    Returns the per-original-row retained mask and integrity statistics. Multiplicity
    is compared exactly, so ``original = A, A, B`` with ``saved = A, A, A`` is
    rejected instead of accepting every saved ``A`` by membership alone.
    """
    original_rows, original_keys = submap_rows(original_raw)
    saved_rows, saved_keys = submap_rows(saved_raw)
    if not np.isfinite(original_rows).all():
        raise ValueError(f'Pre-edit submap {submap_id} contains non-finite points')
    if not np.isfinite(saved_rows).all():
        raise ValueError(f'Saved cleanup submap {submap_id} contains non-finite points')
    if not len(saved_rows):
        return np.zeros(len(original_rows), dtype=bool), dict(
            saved_points=0, unknown_saved_points=0, multiplicity_excess_points=0, max_multiplicity_excess=0)

    original_unique, original_counts = np.unique(original_keys, return_counts=True)
    saved_unique, saved_counts = np.unique(saved_keys, return_counts=True)
    position = np.searchsorted(original_unique, saved_unique)
    clipped = np.clip(position, 0, len(original_unique) - 1)
    found = (position < len(original_unique)) & (original_unique[clipped] == saved_unique)
    unknown = int(saved_counts[~found].sum())
    if unknown:
        raise SavedSubmapError(edit_id, submap_id,
                               'saved geometry contains points that the pre-edit submap never contained',
                               unknown,
                               'The saved map is not a subset of this cleanup working copy. Do not copy points '
                               'between maps; re-save the Clean Map workspace and export it again.')
    excess = saved_counts - original_counts[clipped]
    excess_points = int(excess[excess > 0].sum())
    if excess_points:
        raise SavedSubmapError(edit_id, submap_id,
                               'saved geometry repeats points more often than the pre-edit submap',
                               excess_points,
                               'Saved multiplicities exceed the pre-edit multiplicities, so retained/removed '
                               'membership cannot be proven. Export the Clean Map workspace again without '
                               'duplicating points.')
    positions = np.clip(np.searchsorted(saved_unique, original_keys), 0, len(saved_unique) - 1)
    retained = saved_unique[positions] == original_keys
    stats = dict(saved_points=int(len(saved_rows)), unknown_saved_points=0,
                 multiplicity_excess_points=0,
                 max_multiplicity_excess=int(excess.max()) if len(excess) else 0,
                 original_unique_keys=int(len(original_unique)), saved_unique_keys=int(len(saved_unique)))
    return retained, stats


def load_validated_submap_pair(original_dir, saved_dir, submap_id, edit_id):
    """Validate one pre-edit/saved submap pair and return its retained mask.

    Both point blocks and both ``data.txt`` frames are read and checked: the block must be
    a whole number of float32 rows and finite, the frame must be rigid and its declared id
    must match the submap directory, the two frames must agree exactly, and the saved rows
    must be a multiplicity-valid subset of the pre-edit rows. Any failure raises
    :class:`SavedSubmapError` naming the edit, the submap and the mismatch.
    """
    name = Path(original_dir).name
    original_raw = submap_block_bytes(Path(original_dir) / submap_id / 'points_compact.bin')
    saved_raw = submap_block_bytes(Path(saved_dir) / submap_id / 'points_compact.bin')
    original_matrix = _read_submap_origin(Path(original_dir) / submap_id / 'data.txt', submap_id)
    saved_matrix = _read_submap_origin(Path(saved_dir) / submap_id / 'data.txt', submap_id)
    if not np.array_equal(original_matrix, saved_matrix):
        raise SavedSubmapError(
            edit_id=edit_id, submap_id=submap_id,
            mismatch='saved and pre-edit submaps declare different T_world_origin transforms',
            invalid_points=len(saved_raw) // SUBMAP_ROW_BYTES,
            action='The saved map was written in a different frame. Archive this cleanup and export the Clean Map '
                   'workspace again against the same session.')
    keep, integrity = validate_submap_pair(original_raw, saved_raw, edit_id, submap_id)
    original_rows, _ = submap_rows(original_raw)
    del original_raw, saved_raw
    return keep, integrity, original_matrix, original_rows


def load_edited_reference(workspace, association_radius_m=None, association_spacing_multiplier=None,
                          spacing_sample=DEFAULT_ASSOCIATION_SPACING_SAMPLE):
    """Load and verify the retained/removed reference geometry of a saved cleanup.

    The pre-edit (``map_01``) and saved (``saved_map``) submaps must share the same
    ids and the same ``T_world_origin``, and every saved point must be a byte-exact
    member of the pre-edit submap it came from, counted with multiplicity. The saved
    submap frame is validated rather than ignored, so a cleanup written in a different
    coordinate frame is rejected instead of silently producing wrong world geometry.
    Anything that fails is reported as ``INVALID_SAVED_SUBMAP`` with the edit id,
    submap id, mismatch type and invalid point count; nothing is guessed.
    """
    workspace = Path(workspace)
    original_dir = workspace / 'map_01'
    saved_dir = workspace / 'saved_map'
    for directory in (original_dir, saved_dir):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError('Saved cleanup must contain plain map_01 and saved_map directories')
    original_ids = sorted(p.name for p in original_dir.iterdir() if p.is_dir() and p.name.isdigit())
    saved_ids = sorted(p.name for p in saved_dir.iterdir() if p.is_dir() and p.name.isdigit())
    if not original_ids:
        raise ValueError('Saved cleanup has no pre-edit submaps to compare against')
    if original_ids != saved_ids:
        raise ValueError('Saved cleanup submaps do not match the pre-edit submaps; export the cleanup again')

    # One pass over the submap pair to class every pre-edit point as kept or removed.
    # Each pair is validated as an exact multiset subset before it is trusted, and the
    # saved frame is compared with the pre-edit frame instead of being ignored.
    all_blocks, labels, submaps = [], [], []
    retained_total = original_total = retained_bytes = 0
    for name in original_ids:
        try:
            keep, integrity, original_matrix, original_rows = load_validated_submap_pair(
                original_dir, saved_dir, name, workspace.name)
        except SavedSubmapError:
            raise
        except ValueError as error:
            raise SavedSubmapError(edit_id=workspace.name, submap_id=name, mismatch=str(error), invalid_points=0,
                                   action='The saved cleanup submap is corrupt. Export the Clean Map workspace '
                                          'again, then prepare the job again.') from error
        rotation = original_matrix[:3, :3].astype(np.float64)
        translation = original_matrix[:3, 3].astype(np.float64)
        world = original_rows.astype(np.float64) @ rotation.T + translation
        del original_rows, rotation, translation
        all_blocks.append(world)
        labels.append(keep)
        retained_total += int(keep.sum())
        original_total += len(world)
        retained_bytes += int(keep.sum()) * 24
        submaps.append(dict(id=name, original_points=int(len(world)), retained_points=int(keep.sum()),
                            saved_points=integrity['saved_points'],
                            max_multiplicity_excess=integrity['max_multiplicity_excess']))
    if not retained_total:
        raise ValueError('Saved cleanup retains no submaps; export a cleanup that keeps geometry')
    world = np.concatenate(all_blocks)
    del all_blocks
    retained = np.concatenate(labels)
    del labels
    estimate = int(world.nbytes * 3 + retained.nbytes * 4)
    if estimate > MAX_REFERENCE_INDEX_BYTES:
        raise ValueError(
            f'Saved cleanup has {original_total:,} pre-edit points; its spatial index would need about '
            f'{estimate / 1024 ** 3:.1f} GiB, exceeding the {MAX_REFERENCE_INDEX_BYTES / 1024 ** 3:.0f} GiB limit. '
            'Export a smaller cleanup region before filtering.')
    spacing = measure_sampling_resolution(world[retained], spacing_sample)
    reference = EditedGeometryReference(
        world=world, retained=retained,
        sampling_resolution_m=spacing if spacing else DEFAULT_REFERENCE_SAMPLING_M,
        submaps=submaps,
        association_spacing_multiplier=(DEFAULT_ASSOCIATION_SPACING_MULTIPLIER if association_spacing_multiplier is None
                                        else float(association_spacing_multiplier)),
        association_radius_m=association_radius_m, retained_world=None)
    reference.metadata = dict(
        reference_points=int(original_total), retained_points=int(retained_total),
        removed_points=int(original_total - retained_total),
        retained_ratio=(retained_total / original_total) if original_total else 0.0,
        sampling_resolution_m=float(reference.sampling_resolution_m),
        retained_bytes=int(retained_bytes), estimate_index_bytes=estimate,
        submaps=submaps, association_radius_m=reference.resolve_association_radius(),
        association_spacing_multiplier=reference.association_spacing_multiplier,
        method='nearest_kept_within_measured_sampling_radius_with_nearest_removed_exclusion',
        accuracy=EditedGeometryReference.VALIDATION,
        integrity=dict(validation='strict_multiset_subset', submaps_checked=len(submaps),
                       saved_points=sum(int(entry['saved_points']) for entry in submaps),
                       unknown_saved_points=0, multiplicity_violations=0,
                       transform_identity_verified=True),
        limitation='GLIM stores submaps as voxel samples, so kept/removed membership cannot be traced back to '
                   'individual raw observations. Edited boundaries therefore carry a documented uncertainty band '
                   'equal to the reported association radius.')
    return reference


def measure_sampling_resolution(points, sample=DEFAULT_ASSOCIATION_SPACING_SAMPLE):
    """Median nearest-neighbour spacing of a reference point set, in metres."""
    if len(points) < 2:
        return 0.0
    from scipy.spatial import cKDTree
    tree = cKDTree(points, compact_nodes=True, balanced_tree=True)
    if len(points) > sample:
        pick = np.random.default_rng(0).choice(len(points), int(sample), replace=False)
        query = points[pick]
    else:
        query = points
    distances, _ = tree.query(query, k=2, workers=1)
    spacing = float(np.median(distances[:, 1]))
    if not math.isfinite(spacing) or spacing <= 0:
        return 0.0
    return spacing


def truncation_influence_voxels(voxel_size_m, sdf_trunc_m):
    """Voxels one observation can influence: the truncation band around its cell."""
    half = int(math.ceil(float(sdf_trunc_m) / float(voxel_size_m)))
    return (2 * half + 1) ** 3


def surface_triangles_estimate(touched_voxels, band_thickness):
    """Triangles a marching-cubes isosurface produces from a touched voxel count.

    A surface band is one voxel thick, so the traced surface voxels are the touched
    voxels divided by the truncation band thickness, and each surface voxel yields about
    two triangles. Calibrated against the recorded real-engine benchmark (a 20 mm run over
    the ROI that touched about 14 M voxels produced 3.99 M triangles).
    """
    return 2.0 * float(touched_voxels) / max(int(band_thickness), 1)


def memory_estimate(settings, observed_points=None, observed_bbox=None, occupancy=None,
                    batch_points=None, reference_points=None):
    """Component-wise memory estimate, with every heuristic labelled.

    Components are reported separately because they fail for different reasons: the TSDF
    grows with the touched surface, the reference index grows with the export size, the
    input batches grow with ``batch_points``, and extraction/masking/output grow with the
    triangle count.

    The *primary* estimate for the touched surface is the observation count times the
    measured voxels-per-observation when a bag sample is available, and the observation
    count itself otherwise (one voxel per surface observation). The audit's finding is
    still fixed and reported explicitly: the previous code used ``min(dense_band, N)``,
    which understates the touched surface by up to ``truncation_influence_voxels()`` (343x
    at 20 mm voxels with 60 mm truncation) because one observation can influence many
    voxels. That union bound, and the dense truncation band, are now computed and reported
    as ``upper_bound`` figures next to the primary estimate instead of being hidden inside
    a single number.
    """
    voxel = float(settings['voxel_size_m'])
    trunc = float(settings['sdf_trunc_m'])
    influence = truncation_influence_voxels(voxel, trunc)
    band_thickness = 2 * int(math.ceil(trunc / voxel)) + 1
    batch_points = int(batch_points or settings.get('batch_points') or DEFAULT_BATCH_POINTS)
    components = {}
    basis = []
    extent = dense_voxels = upper_voxels = None
    if observed_bbox is not None:
        lower = np.asarray(observed_bbox[0], dtype=np.float64)
        upper = np.asarray(observed_bbox[1], dtype=np.float64)
        sizes = np.ceil(np.maximum(upper - lower, voxel) / voxel) + 2 * math.ceil(trunc / voxel)
        extent = (upper - lower).tolist()
        dense_voxels = float(np.prod(np.maximum(sizes, 1)))
        components['tsdf_dense_band_voxels'] = dense_voxels
        components['tsdf_dense_band_bytes'] = int(dense_voxels * TSDF_BYTES_PER_TOUCHED_VOXEL)
        basis.append(f'dense truncation band over the sampled extent: {dense_voxels:,.0f} voxels (geometric '
                     'ceiling, assumes no sparse surface)')
    if occupancy and occupancy.get('observations'):
        density = float(occupancy['voxels']) / float(occupancy['observations'])
        basis.append(f'measured occupancy from a bag sample: {density:.2f} voxels per observation at '
                     f'{occupancy.get("voxel_size_m", voxel) * 1000:.0f} mm')
    else:
        density = 1.0
        basis.append('no bag sample was available, so one TSDF voxel per surface observation is assumed')
    primary_voxels = None
    if observed_points:
        primary_voxels = float(observed_points) * density
        basis.append(f'{int(observed_points):,} estimated observations x {density:.2f} voxels')
    elif dense_voxels:
        primary_voxels = dense_voxels
        basis.append('no observation count was available, so the dense band is used')
    if observed_points and observed_bbox is not None:
        upper_voxels = min(dense_voxels, float(observed_points) * influence)
    elif observed_points:
        upper_voxels = float(observed_points) * influence
    if primary_voxels is None:
        return dict(components={}, total_peak_bytes=None, basis=basis, heuristics={},
                    safety_margin=PREFLIGHT_SAFETY_MARGIN, observed_extent_m=extent,
                    observed_points=None if observed_points is None else int(observed_points),
                    band_voxels_upper_bound=dense_voxels, estimated_tsdf_voxels=None,
                    estimated_tsdf_bytes=None, truncation_influence_voxels=influence, density=None,
                    upper_bound_voxels=None, upper_bound_bytes=None, band_thickness_voxels=band_thickness)
    if dense_voxels and primary_voxels > dense_voxels:
        primary_voxels = dense_voxels
    touched = min(primary_voxels, upper_voxels) if upper_voxels else primary_voxels
    components['tsdf_estimated_bytes'] = int(touched * TSDF_BYTES_PER_TOUCHED_VOXEL)
    upper_bound_bytes = int(upper_voxels * TSDF_BYTES_PER_TOUCHED_VOXEL) if upper_voxels else None
    components['tsdf_upper_bound_bytes'] = upper_bound_bytes
    basis.append(f'truncation influence: {influence} voxels per observation, {band_thickness} voxels thick')
    triangles = surface_triangles_estimate(touched, band_thickness)
    components['mesh_triangles_estimate'] = int(triangles)
    components['mesh_extraction_bytes'] = int(triangles * MESH_BYTES_PER_TRIANGLE)
    components['mesh_masking_bytes'] = int(min(MASK_FACE_BATCH, triangles) * MASK_PROBE_BYTES_PER_TRIANGLE)
    components['mesh_output_bytes'] = int(triangles * PLY_BYTES_PER_TRIANGLE)
    components['mesh_output_transient_bytes'] = int(triangles * PLY_BYTES_PER_TRIANGLE * PLY_TRANSIENT_FACTOR)
    components['input_batch_bytes'] = int(batch_points * BATCH_BYTES_PER_OBSERVATION)
    if reference_points:
        components['reference_bytes'] = int(float(reference_points) * REFERENCE_BYTES_PER_POINT)
        basis.append(f'edited-geometry reference: {int(reference_points):,} pre-edit samples')
    else:
        components['reference_bytes'] = 0
    peak = (components['tsdf_estimated_bytes'] + components['input_batch_bytes'] +
            components['reference_bytes'] + components['mesh_extraction_bytes'] +
            components['mesh_output_transient_bytes'])
    heuristics = dict(mesh_bytes_per_triangle=MESH_BYTES_PER_TRIANGLE, ply_bytes_per_triangle=PLY_BYTES_PER_TRIANGLE,
                      ply_transient_factor=PLY_TRANSIENT_FACTOR, reference_bytes_per_point=REFERENCE_BYTES_PER_POINT,
                      batch_bytes_per_observation=BATCH_BYTES_PER_OBSERVATION,
                      bytes_per_touched_voxel=TSDF_BYTES_PER_TOUCHED_VOXEL,
                      triangles_per_surface_voxel=2.0, band_thickness_voxels=band_thickness)
    return dict(components=components, total_peak_bytes=int(peak * PREFLIGHT_SAFETY_MARGIN), basis=basis,
                heuristics=heuristics, safety_margin=PREFLIGHT_SAFETY_MARGIN, observed_extent_m=extent,
                observed_points=None if observed_points is None else int(observed_points),
                band_voxels_upper_bound=dense_voxels, estimated_tsdf_voxels=touched,
                estimated_tsdf_bytes=components['tsdf_estimated_bytes'],
                truncation_influence_voxels=influence, density=density, upper_bound_voxels=upper_voxels,
                upper_bound_bytes=upper_bound_bytes, band_thickness_voxels=band_thickness)


def _mitigations(voxel_size_m, target_bytes, extent_m, free_ram):
    """Concrete ways to fit a run into the available memory, never a silent downgrade."""
    suggestions = []
    if free_ram > 0:
        suggestions.append(f'Free memory or reduce concurrent work so about {target_bytes / 1024 ** 3:.1f} GiB is '
                           f'available (currently {free_ram / 1024 ** 3:.1f} GiB).')
    if extent_m:
        suggestions.append('Use a region of interest (ROI) covering only the edited area instead of the whole '
                           f'sampled {np.round(extent_m, 1).tolist()} m extent.')
    suggestions.append(f'Use a larger voxel size than {voxel_size_m * 1000:.0f} mm and re-prepare: the request is '
                       'never downgraded silently, so the configured resolution is what runs.')
    return suggestions


def preflight(root, settings, observed_points=None, observed_bbox=None, bag_bytes=None, occupancy=None,
              batch_points=None, reference_points=None, memory_budget_bytes=None):
    """Resource preflight: measured free RAM/disk and a component-wise memory estimate.

    Streaming bounds the *input* memory, not the TSDF: a fine sparse volume over a whole
    factory still grows with the touched surface, and extraction, masking and PLY writing
    add their own peaks after integration. Every estimate is an order-of-magnitude figure
    with an explicit safety margin, reported per component so a user can see which part
    is under pressure before committing to a long run.

    ``ok`` is False when the estimated peak cannot fit inside
    ``min(configured budget, available RAM - reserve)``. When it is False the run must be
    refused with ``RESOURCE_PREFLIGHT_FAILED`` and mitigation advice; the requested
    resolution is never silently downgraded.
    """
    import psutil
    free_ram = psutil.virtual_memory().available
    total_ram = psutil.virtual_memory().total
    target = Path(root)
    while not target.exists() and target.parent != target:
        target = target.parent
    disk = psutil.disk_usage(str(target))
    settings = dict(settings)
    if batch_points is not None:
        settings['batch_points'] = batch_points
    estimate = memory_estimate(settings, observed_points=observed_points, observed_bbox=observed_bbox,
                               occupancy=occupancy, batch_points=batch_points, reference_points=reference_points)
    reserve = max(int(total_ram * PREFLIGHT_RAM_RESERVE_FRACTION), PREFLIGHT_MIN_RAM_RESERVE_BYTES)
    budget = resolve_memory_budget_bytes(settings, override=memory_budget_bytes)
    usable = max(min(budget, free_ram - reserve), 0)
    total = estimate['total_peak_bytes']
    upper_bound = estimate.get('upper_bound_bytes')
    warnings, failures = [], []
    ok = True
    # Compared as byte values, never by truthiness: zero usable memory used to leave ``ok``
    # True and report a run that cannot fit at all as passing.
    if total is not None and total > usable:
        ok = False
        failures.append(
            f'RESOURCE_PREFLIGHT_FAILED: the estimated peak of {total / 1024 ** 3:.1f} GiB '
            f'({estimate["safety_margin"]:.2f}x safety margin over the component estimate) exceeds the usable '
            f'{usable / 1024 ** 3:.1f} GiB (budget {budget / 1024 ** 3:.1f} GiB, free RAM '
            f'{free_ram / 1024 ** 3:.1f} GiB minus {reserve / 1024 ** 3:.1f} GiB reserve).'
            + (' Free RAM does not even cover the reserve, so no VDBFusion run fits on this workstation right now.'
               if usable <= 0 else ''))
    elif total is not None and usable > 0 and total > usable * 0.75:
        warnings.append(
            f'The estimated peak of {total / 1024 ** 3:.1f} GiB is close to the usable {usable / 1024 ** 3:.1f} GiB; '
            'keep the workstation free of other large jobs and watch the run.')
    if upper_bound is not None and usable and upper_bound > usable:
        warnings.append(
            f'Worst case, every observation could influence {estimate["truncation_influence_voxels"]} voxels, which '
            f'would need up to {upper_bound / 1024 ** 3:.1f} GiB of TSDF against the usable '
            f'{usable / 1024 ** 3:.1f} GiB. The estimate above assumes a sparse surface; watch the run, and prefer an '
            'ROI or a larger voxel size if memory pressure appears.')
    voxel = settings['voxel_size_m']
    if estimate['band_voxels_upper_bound'] and estimate['observed_extent_m']:
        span = max(estimate['observed_extent_m'])
        if voxel <= 0.01 and span >= 40.0:
            warnings.append(
                f'A {voxel * 1000:.0f} mm TSDF over a {span:.0f} m scan is memory intensive: the dense truncation '
                f'band alone is {estimate["band_voxels_upper_bound"]:,.0f} voxels. Prefer 20 mm, or restrict the '
                'run with an ROI, then re-prepare.')
    if disk.free < 2 * 1024 ** 3:
        failures.append(f'Only {disk.free / 1024 ** 3:.1f} GiB of free disk space on {target}.')
        ok = False
    return dict(ok=ok, code=None if ok else 'RESOURCE_PREFLIGHT_FAILED',
                free_ram_bytes=int(free_ram), total_ram_bytes=int(total_ram),
                free_disk_bytes=int(disk.free), disk_path=str(target),
                voxel_size_m=voxel, sdf_trunc_m=settings['sdf_trunc_m'], observed_extent_m=estimate['observed_extent_m'],
                estimated_tsdf_voxels=estimate['estimated_tsdf_voxels'],
                band_voxels_upper_bound=estimate['band_voxels_upper_bound'],
                estimated_tsdf_bytes=estimate['estimated_tsdf_bytes'],
                truncation_influence_voxels=estimate['truncation_influence_voxels'],
                density=estimate['density'], components=estimate['components'],
                estimated_tsdf_upper_bound_bytes=upper_bound,
                band_thickness_voxels=estimate['band_thickness_voxels'],
                estimated_peak_bytes=total, safety_margin=estimate['safety_margin'],
                usable_bytes=int(usable), ram_reserve_bytes=int(reserve), memory_budget_bytes=int(budget),
                bytes_per_touched_voxel=TSDF_BYTES_PER_TOUCHED_VOXEL,
                estimate_note='Component-wise order-of-magnitude estimate with an explicit safety margin. Streaming '
                              'bounds input memory only; the sparse TSDF itself is not memory bounded.',
                estimate_basis=estimate['basis'], heuristics=estimate['heuristics'],
                observed_points=None if observed_points is None else int(observed_points),
                bag_bytes=None if bag_bytes is None else int(bag_bytes),
                warnings=warnings, failures=failures,
                suggestions=[] if ok else _mitigations(voxel, total or 0, estimate['observed_extent_m'], free_ram))


def _format_bytes(value):
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if abs(value) < 1024 or unit == 'TiB':
            return f'{value:.1f} {unit}'
        value /= 1024


class MemoryBudgetExceeded(RuntimeError):
    """The configured soft memory budget was reached; fail cleanly, never re-resolve."""


def check_memory_budget(rss_bytes, peak_limit_bytes, budget_bytes):
    """Return the failure message when RSS exceeds the soft budget, otherwise ``None``.

    A separate function because this is the runtime half of memory safety: the preflight
    refuses a run that cannot fit, and this check fails a run that grows past its budget
    *while* integrating. Neither ever lowers the requested resolution.
    """
    if peak_limit_bytes is not None and rss_bytes > peak_limit_bytes:
        return (f'Worker RSS {_format_bytes(rss_bytes)} exceeded the configured memory budget '
                f'{_format_bytes(peak_limit_bytes)} while integrating. Reduce the voxel size request or the scan '
                'extent; the requested resolution was not changed.')
    if peak_limit_bytes is None and rss_bytes > budget_bytes:
        return (f'Worker RSS {_format_bytes(rss_bytes)} exceeded the default memory budget '
                f'{_format_bytes(budget_bytes)} while integrating. Reduce the voxel size request or the scan '
                'extent; the requested resolution was not changed.')
    return None


def integrate_bag(bag, trajectory, topic, settings, output=None, edit_reference=None,
                  boundary_margin_m=0.0, unsupported_policy='exclude', memory_budget_bytes=None,
                  event=None, cancel=None, peak_limit_bytes=None):
    """Stream a bag into one fused TSDF, then return the volume and its statistics.

    The whole recording is never materialised: batches are read, filtered,
    motion-grouped and handed to the native integrator one bounded buffer at a
    time. ``peak_limit_bytes`` is a soft RSS budget checked between batches; when
    it is exceeded the run fails cleanly instead of silently lowering resolution.
    """
    import vdbfusion
    import psutil

    event = event or (lambda stage, **fields: None)
    process = psutil.Process()
    budget = resolve_memory_budget_bytes(settings, override=memory_budget_bytes)
    volume = vdbfusion.VDBVolume(settings['voxel_size_m'], settings['sdf_trunc_m'], settings['space_carving'])
    stats = dict(
        raw_frames=0, raw_observations=0, integrated_observations=0, origin_groups=0,
        max_origin_error_m=0.0, origin_error_budget_m=settings['origin_error_budget_m'],
        batches=0, edit_filter_enabled=edit_reference is not None,
        points_before_filter=0, points_retained=0, points_dropped_removed_support=0,
        points_dropped_unsupported=0, filter_retention_ratio=None,
        space_carving=settings['space_carving'], voxel_size_m=settings['voxel_size_m'],
        sdf_trunc_m=settings['sdf_trunc_m'], units='meters', coordinate_system='GLIM_world',
        roi_min_m=settings.get('roi_min_m'), roi_max_m=settings.get('roi_max_m'),
        roi_enabled=settings.get('roi_min_m') is not None, points_dropped_outside_roi=0,
    )
    if edit_reference is not None:
        stats['association_radius_m'] = edit_reference.resolve_association_radius()
        stats['edit_reference'] = dict(edit_reference.metadata)
        stats['edit_filter_accuracy'] = EditedGeometryReference.VALIDATION
        stats['boundary_margin_m'] = float(boundary_margin_m)
        stats['unsupported_observations'] = unsupported_policy
    started = time.monotonic()
    peak_rss = process.memory_info().rss
    counters = dict(frames=0, observations=0, outside_trajectory=0)
    timings = dict(bag_read_seconds=0.0, decode_seconds=0.0, transform_seconds=0.0, batch_seconds=0.0,
                   filter_seconds=0.0, group_seconds=0.0, integrate_seconds=0.0)
    radius = stats.get('association_radius_m')
    roi_min = np.asarray(settings['roi_min_m'], dtype=np.float64) if settings.get('roi_min_m') else None
    roi_max = np.asarray(settings['roi_max_m'], dtype=np.float64) if settings.get('roi_max_m') else None
    # The bag reader opens lazily, so the stage is set before the first iteration.
    event('READING_BAG')

    def timed(inner):
        """Attribute the time spent inside the batch generator to input reading."""
        start = time.perf_counter()
        for item in inner:
            timings['read_seconds'] = timings.get('read_seconds', 0.0) + time.perf_counter() - start
            yield item
            start = time.perf_counter()

    for batch in timed(iter_world_batches(bag, trajectory, topic, settings['batch_points'],
                                          counters=counters, cancel=cancel, timings=timings)):
        stats['batches'] += 1
        stats['raw_observations'] = counters['observations']
        stats['raw_frames'] = counters['frames']
        stats['outside_trajectory'] = counters['outside_trajectory']
        points, origins = batch['points'], batch['origins']
        if roi_min is not None:
            inside = np.all((points >= roi_min) & (points <= roi_max), axis=1)
            stats['points_dropped_outside_roi'] += int((~inside).sum())
            points, origins = points[inside], origins[inside]
        if edit_reference is not None:
            started_filter = time.perf_counter()
            stats['points_before_filter'] += len(points)
            classified = edit_reference.classify(points, radius, boundary_margin_m)
            keep = classified['keep']
            stats['points_dropped_removed_support'] += int(classified['removed_dominated'].sum())
            stats['points_dropped_unsupported'] += int(classified['unsupported'].sum())
            if unsupported_policy == 'include':
                keep = keep | classified['unsupported']
            points, origins = points[keep], origins[keep]
            stats['points_retained'] += len(points)
            timings['filter_seconds'] += time.perf_counter() - started_filter
        if not len(points):
            del batch
            continue
        started_group = time.perf_counter()
        groups = list(motion_aware_groups(origins, settings['origin_error_budget_m']))
        timings['group_seconds'] += time.perf_counter() - started_group
        started_integrate = time.perf_counter()
        for start, stop, origin, error in groups:
            group = np.ascontiguousarray(points[start:stop], dtype=np.float64)
            if not len(group):
                continue
            volume.integrate(group, np.ascontiguousarray(origin, dtype=np.float64))
            stats['origin_groups'] += 1
            stats['max_origin_error_m'] = max(stats['max_origin_error_m'], error)
        timings['integrate_seconds'] += time.perf_counter() - started_integrate
        stats['integrated_observations'] += len(points)
        del batch, points, origins, groups
        rss = process.memory_info().rss
        peak_rss = max(peak_rss, rss)
        if exceed := check_memory_budget(rss, peak_limit_bytes, budget):
            raise MemoryBudgetExceeded(exceed)
        event('INTEGRATING', batches=stats['batches'], origin_groups=stats['origin_groups'],
              integrated_observations=stats['integrated_observations'], rss_bytes=int(rss),
              message=(f'Integrated {stats["integrated_observations"]:,} observations in '
                       f'{stats["origin_groups"]:,} motion-aware groups'))
    if edit_reference is not None and stats['points_before_filter']:
        stats['filter_retention_ratio'] = stats['points_retained'] / stats['points_before_filter']
    stats['peak_rss_bytes'] = int(peak_rss)
    stats['integration_seconds'] = time.monotonic() - started
    timings['integration_total_seconds'] = stats['integration_seconds']
    # Batching is whatever input reading is not decoding, interpolating or buffering.
    timings['batch_seconds'] = max(timings['batch_seconds'], 0.0)
    stats['timings'] = {key: round(value, 4) for key, value in timings.items()}
    if not stats['integrated_observations']:
        raise ValueError('No observations were integrated; verify the topic, trajectory times and edit filter')
    return volume, stats


MASK_FACE_BATCH = 250_000
MAX_EDGE_INTERIOR_SAMPLES = 8
# Interior probes are allocated in chunks so an unusual mesh with many long edges
# cannot spike memory while it is being validated.
MASK_PROBE_CHUNK = 50_000
# Exact topology (edge orientation and connected components) is computed up to this many
# faces; above it the topology is measured on a bounded, deterministic sample so a
# 50-million-triangle factory mesh cannot blow up the audit's memory.
AUDIT_TOPOLOGY_FACE_LIMIT = 1_500_000
# Edge-length statistics keep at most this many evenly spread measurements, so a
# 17-million-triangle mesh (52 million edges) cannot allocate a 400 MiB array just to
# compute a median. The count is reported alongside the sample size.
AUDIT_EDGE_SAMPLE = 600_000


def vertex_and_midpoint_probes(vertices, faces):
    """The three vertices and three edge midpoints of every given triangle."""
    corners = vertices[faces]
    probes = np.empty((len(faces), 6, 3), dtype=np.float64)
    probes[:, :3] = corners
    probes[:, 3] = 0.5 * (corners[:, 0] + corners[:, 1])
    probes[:, 4] = 0.5 * (corners[:, 1] + corners[:, 2])
    probes[:, 5] = 0.5 * (corners[:, 2] + corners[:, 0])
    return probes


def interior_edge_probes(vertices, faces, spacing_m, max_interior=MAX_EDGE_INTERIOR_SAMPLES):
    """Adaptive interior samples along edges longer than the reference spacing.

    Returns ``(points, valid, capped)``: a ``(len(faces), 3 * max_interior, 3)`` array
    with a validity mask, and how many triangles needed more interior samples than the
    documented cap allows. Endpoints and midpoints are excluded, so no probe duplicates
    the vertex/midpoint pass.
    """
    corners = vertices[faces]
    edges = ((0, 1), (1, 2), (2, 0))
    slots = 3 * max_interior
    points = np.zeros((len(faces), slots, 3), dtype=np.float64)
    valid = np.zeros((len(faces), slots), dtype=bool)
    capped = 0
    limit = max(float(spacing_m), 1e-9)
    for index, (first, second) in enumerate(edges):
        start, end = corners[:, first], corners[:, second]
        length = np.linalg.norm(end - start, axis=1)
        needed = np.ceil(length / (0.5 * limit)) - 1.0
        capped += int(np.count_nonzero(needed > max_interior))
        counts = np.clip(np.nan_to_num(needed, nan=0.0), 0, max_interior).astype(np.int64)
        for slot in range(max_interior):
            active = np.flatnonzero(counts > slot)
            if not len(active):
                continue
            fraction = ((slot + 1) / (counts[active] + 1)).reshape(-1, 1)
            points[active, index * max_interior + slot] = start[active] + (end[active] - start[active]) * fraction
            valid[active, index * max_interior + slot] = True
    return points, valid, capped


def probe_removed_geometry(vertices, faces, reference, association_radius_m, boundary_margin_m,
                           ambiguity_band_m, spacing_m):
    """Conservative staged probe of candidate triangles against removed geometry.

    Returns ``(removed, ambiguous, stats)``. ``removed`` is True for a triangle that any
    probe showed to be dominated by removed reference geometry: deleted geometry must not
    be glued back into the mesh, so one hit removes the whole triangle.
    """
    removed = np.zeros(len(faces), dtype=bool)
    ambiguous = np.zeros(len(faces), dtype=bool)
    stats = dict(probe_triangles=0, interior_probe_triangles=0, long_edge_cap_hits=0,
                 ambiguous_triangles=0, max_probe_samples_per_triangle=0)
    if not len(faces):
        return removed, ambiguous, stats
    classify = lambda coordinates: reference.classify(  # noqa: E731
        coordinates, association_radius_m, boundary_margin_m, ambiguity_band_m)
    probes = vertex_and_midpoint_probes(vertices, faces)
    flat = probes.reshape(-1, 3)
    classified = classify(flat) if len(flat) else None
    if classified is not None:
        removed = classified['removed_dominated'].reshape(len(faces), 6).any(axis=1)
        ambiguous = (~removed) & classified['ambiguous'].reshape(len(faces), 6).any(axis=1)
    stats['probe_triangles'] = int(len(faces))
    stats['max_probe_samples_per_triangle'] = 6
    stats['ambiguous_triangles'] = int(ambiguous.sum())
    long_candidates = np.flatnonzero(~removed)
    if not len(long_candidates):
        return removed, ambiguous, stats
    for start in range(0, len(long_candidates), MASK_PROBE_CHUNK):
        chunk = long_candidates[start:start + MASK_PROBE_CHUNK]
        points, valid, capped = interior_edge_probes(vertices, faces[chunk], spacing_m)
        stats['long_edge_cap_hits'] += capped
        flat_valid = valid.reshape(-1)
        if not flat_valid.any():
            continue
        inner = classify(points.reshape(-1, 3)[flat_valid])
        removed_full = np.zeros(flat_valid.size, dtype=bool)
        removed_full[flat_valid] = inner['removed_dominated']
        ambiguous_full = np.zeros(flat_valid.size, dtype=bool)
        ambiguous_full[flat_valid] = inner['ambiguous']
        hit = removed_full.reshape(len(chunk), -1).any(axis=1)
        removed[chunk] |= hit
        inner_ambiguous = (~hit) & ambiguous_full.reshape(len(chunk), -1).any(axis=1)
        ambiguous[chunk] |= inner_ambiguous
        stats['interior_probe_triangles'] += int(len(chunk))
        stats['max_probe_samples_per_triangle'] = max(
            stats['max_probe_samples_per_triangle'], 6 + 3 * MAX_EDGE_INTERIOR_SAMPLES)
    stats['ambiguous_triangles'] = int(ambiguous.sum())
    return removed, ambiguous, stats


def extract_and_mask(volume, edit_reference=None, association_radius_m=None, boundary_margin_m=0.0,
                     event=None, mask_deleted_triangles=True, cancel=None, ambiguity_band_m=None):
    """Extract the fused triangle mesh and mask triangles that touch removed geometry.

    Removing observations is necessary but not sufficient: a retained ray can still
    bridge a deleted gap, and a large triangle can cross a deleted strip while its
    centroid sits in kept geometry. Masking is therefore staged and conservative:

    1. triangles whose centroid is dominated by removed reference geometry are dropped;
    2. the survivors are probed at all three vertices and all three edge midpoints, and a
       triangle touched by removed geometry is dropped;
    3. survivors with edges longer than the measured reference sampling resolution get
       interior samples along those edges, bounded by ``MAX_EDGE_INTERIOR_SAMPLES`` per
       edge (hits on that cap are reported, never silently ignored).

    One hit on removed geometry removes the whole triangle, because gluing deleted
    geometry back into the mesh is exactly the failure this filter prevents. Probes inside
    the documented ambiguity band are counted and reported, not used to remove geometry.
    Face batches are bounded, cancellation is polled while masking, and winding and vertex
    connectivity are preserved when vertices are compacted.
    """
    event = event or (lambda stage, **fields: None)
    started = time.monotonic()
    check_cancellation(cancel)
    vertices, faces = volume.extract_triangle_mesh()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    stats = dict(extract_seconds=time.monotonic() - started, vertex_count=len(vertices), face_count=len(faces),
                 masked_triangles=0, mask_enabled=False)
    if not len(vertices) or not len(faces):
        return vertices, faces, stats
    if edit_reference is not None and mask_deleted_triangles and len(faces):
        stats['mask_enabled'] = True
        started_mask = time.monotonic()
        radius = association_radius_m if association_radius_m is not None else \
            edit_reference.resolve_association_radius()
        band = ambiguity_band_m if ambiguity_band_m is not None else edit_reference.sampling_resolution_m
        keep = np.ones(len(faces), dtype=bool)
        by_centroid = by_probe = by_unsupported = ambiguous_total = 0
        for start in range(0, len(faces), MASK_FACE_BATCH):
            check_cancellation(cancel)
            stop = min(start + MASK_FACE_BATCH, len(faces))
            window = faces[start:stop]
            corners = vertices[window]
            centroids = corners.mean(axis=1)
            classified = edit_reference.classify(centroids, radius, boundary_margin_m, band)
            # The centroid stage keeps the established edit-filter semantics: a triangle
            # without kept support inside the radius was dropped before this audit and is
            # dropped here as well, so the fix cannot silently start retaining geometry.
            removed = ~classified['keep']
            by_centroid += int(classified['removed_dominated'].sum())
            by_unsupported += int(classified['unsupported'].sum())
            # Probing only applies to triangles that would otherwise be retained: the
            # possible false-retention set. A probe can only remove extra triangles when it
            # finds removed geometry, never resurrect one.
            candidates = np.flatnonzero(classified['keep'])
            if len(candidates):
                hit, ambiguous, probe_stats = probe_removed_geometry(
                    vertices, window[candidates], edit_reference, radius, boundary_margin_m, band,
                    edit_reference.sampling_resolution_m)
                removed[candidates[hit]] = True
                ambiguous_total += int(ambiguous.sum())
                by_probe += int(hit.sum())
                stats['long_edge_cap_hits'] = stats.get('long_edge_cap_hits', 0) + probe_stats['long_edge_cap_hits']
                stats['interior_probe_triangles'] = stats.get('interior_probe_triangles', 0) + \
                    probe_stats['interior_probe_triangles']
                stats['max_probe_samples_per_triangle'] = max(
                    stats.get('max_probe_samples_per_triangle', 0),
                    probe_stats['max_probe_samples_per_triangle'])
            keep[start:stop] = ~removed
            del corners, centroids, classified, candidates, removed, window
            event('VALIDATING_MESH', masked_faces=by_centroid + by_probe,
                  message=f'Checking triangles against the removed regions: {stop:,}/{len(faces):,}')
        stats['mask_seconds'] = time.monotonic() - started_mask
        stats['triangles_before'] = int(len(faces))
        stats['triangles_removed_by_centroid'] = int(by_centroid)
        stats['triangles_removed_by_probe'] = int(by_probe)
        stats['triangles_dropped_unsupported'] = int(by_unsupported)
        stats['ambiguous_triangles'] = int(ambiguous_total)
        stats['ambiguity_band_m'] = float(band)
        stats['masked_triangles'] = int(len(faces) - np.count_nonzero(keep))
        faces = faces[keep]
        if len(faces):
            used = np.unique(faces)
            remap = np.full(len(vertices), -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
            vertices = vertices[used]
            faces = remap[faces]
        else:
            vertices = np.zeros((0, 3), dtype=np.float64)
            faces = np.zeros((0, 3), dtype=np.int64)
        stats.update(vertex_count=len(vertices), face_count=len(faces))
    return vertices, faces, stats


def audit_mesh(vertices, faces, settings, observed_bbox=None, batch=MASK_FACE_BATCH):
    """Independent structural audit of a triangle mesh.

    Everything here is *reported*, never used to delete geometry: a legitimate factory
    mesh may legitimately have several disconnected components or long triangles, so the
    audit surfaces the numbers and warnings and leaves the decision to a human. Hard
    corruption (non-finite vertices, out-of-range indices) is raised by
    :func:`validate_and_report` instead.

    The audit is batched over faces and every per-edge statistic is computed with
    ``bincount`` over a single ``unique`` pass, so a four-million-triangle mesh costs
    seconds and a bounded working set rather than a Python loop over millions of edges.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as scipy_components
    # Pass 1: exact totals, running maxima and an evenly spread edge-length sample.
    stride = max(1, int(math.ceil(3.0 * len(faces) / AUDIT_EDGE_SAMPLE)))
    degenerate = 0
    total_area = 0.0
    longest_edge = 0.0
    sampled_lengths = []
    for start in range(0, len(faces), batch):
        window = faces[start:start + batch]
        corners = vertices[window]
        first = corners[:, 1] - corners[:, 0]
        second = corners[:, 2] - corners[:, 0]
        areas = 0.5 * np.linalg.norm(np.cross(first, second), axis=1)
        degenerate += int(np.count_nonzero(areas <= 0.0))
        total_area += float(areas.sum())
        lengths = np.concatenate([np.linalg.norm(first, axis=1),
                                 np.linalg.norm(corners[:, 2] - corners[:, 1], axis=1),
                                 np.linalg.norm(corners[:, 0] - corners[:, 2], axis=1)])
        longest_edge = max(longest_edge, float(lengths.max()) if len(lengths) else 0.0)
        if stride == 1:
            sampled_lengths.append(lengths)
        else:
            sampled_lengths.append(lengths[::stride])
        del corners, first, second, areas, window
    sampled = np.concatenate(sampled_lengths) if sampled_lengths else np.zeros(0)
    del sampled_lengths
    median_edge = float(np.median(sampled)) if len(sampled) else 0.0
    threshold = max(8.0 * median_edge, float(settings['sdf_trunc_m']) * 8.0) if median_edge else 0.0
    # The extreme-edge count uses the measured sample and the reported median, so it is
    # labelled with the same scope as the median rather than costing a second full pass.
    extreme_edges = int(np.count_nonzero(sampled > threshold)) if threshold else 0
    # Orientation conflicts: a consistently wound manifold uses every shared edge once in
    # each direction, so an edge seen twice in the *same* direction is a conflict. Both the
    # unique pass and the per-edge tally are vectorised, and the 2 x int64 edge keys are
    # viewed as one 16-byte key so the sort runs on a flat array instead of a structured one.
    topological = faces if len(faces) <= AUDIT_TOPOLOGY_FACE_LIMIT else \
        faces[np.random.default_rng(0).choice(len(faces), AUDIT_TOPOLOGY_FACE_LIMIT, replace=False)]
    ordered = np.concatenate([topological[:, [0, 1]], topological[:, [1, 2]], topological[:, [2, 0]]])
    keys = np.ascontiguousarray(np.sort(ordered, axis=1))
    edge_keys = keys.view(np.dtype((np.void, 2 * keys.dtype.itemsize))).ravel()
    _, inverse, counts = np.unique(edge_keys, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).ravel()
    # One representative occurrence per unique edge, without a second sort.
    representative = np.empty(len(counts), dtype=np.int64)
    representative[inverse] = np.arange(len(inverse), dtype=np.int64)
    forward = np.bincount(inverse, weights=(ordered[:, 0] < ordered[:, 1]).astype(np.float64),
                          minlength=len(counts))
    backward = counts - forward
    conflicts = int(np.count_nonzero((forward >= 2) | (backward >= 2)))
    graph_edges = keys[representative]
    del inverse, forward, backward, counts, representative, edge_keys, keys, ordered
    if len(graph_edges) and len(vertices) < np.iinfo(np.int32).max:
        graph_edges = graph_edges.astype(np.int32)
    exact_topology = len(topological) == len(faces)
    used_components = largest = None
    if exact_topology:
        graph = coo_matrix((np.ones(len(graph_edges), dtype=np.int8), (graph_edges[:, 0], graph_edges[:, 1])),
                           shape=(len(vertices), len(vertices)))
        component_count, labels = scipy_components(graph, directed=False)
        per_component = np.bincount(labels[topological[:, 0]], minlength=component_count)
        used_components = int(np.count_nonzero(per_component))
        largest = int(per_component.max()) if len(per_component) else 0
    del graph_edges
    report = dict(
        degenerate_triangles=degenerate,
        degenerate_ratio=float(degenerate / len(faces)),
        total_surface_area_m2=total_area,
        median_edge_m=median_edge,
        longest_edge_m=longest_edge,
        extreme_edges=extreme_edges,
        extreme_edge_threshold_m=threshold,
        edge_lengths_measured=int(len(sampled)),
        edge_statistics_scope='exact' if len(sampled) == 3 * len(faces) else 'sampled',
        orientation_conflict_edges=conflicts,
        orientation_consistent=bool(conflicts == 0),
        connected_components=used_components,
        largest_component_ratio=(float(largest / len(topological)) if largest is not None and len(topological)
                                else None),
        topology_scope='exact' if exact_topology else 'sampled',
        topology_faces_measured=int(len(topological)),
        orientation_conflicts_are_lower_bound=not exact_topology,
        connected_components_note=(None if exact_topology else
                                  'Not measurable from a face sample: sampling an arbitrary face subset cuts the '
                                  'edge adjacency, so a component count on the sample would be a fragmentation '
                                  'artefact, not a property of the mesh.'),
        audit_note='Reported, never used to delete geometry: disconnected components and long triangles are normal '
                   'in factory scans. Above the documented face limit the topology is measured on a bounded sample '
                   'and labelled as such.',
    )
    if observed_bbox is not None:
        lower = np.asarray(observed_bbox[0], dtype=np.float64)
        upper = np.asarray(observed_bbox[1], dtype=np.float64)
        margin = 2.0 * float(settings['sdf_trunc_m'])
        mesh_lower, mesh_upper = vertices.min(axis=0), vertices.max(axis=0)
        report['mesh_outside_observed_bbox_m'] = float(max(
            np.max(lower - margin - mesh_lower), np.max(mesh_upper - (upper + margin)), 0.0))
        report['mesh_within_observed_bbox'] = bool(report['mesh_outside_observed_bbox_m'] <= 0.0)
    return report


def validate_and_report(vertices, faces, settings, extra=None, observed_bbox=None):
    """Independent mesh validation through the shared NKSR-era mesh reader."""
    if not len(vertices) or not len(faces):
        raise ValueError('TSDF produced an empty mesh; verify the observations, trajectory and TSDF settings')
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if not np.isfinite(vertices).all():
        raise ValueError('TSDF produced non-finite vertices')
    if faces.dtype.kind not in 'iu':
        raise ValueError('TSDF produced non-integer triangle indices')
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('TSDF produced triangle indices outside the vertex range')
    report = dict(vertex_count=int(len(vertices)), face_count=int(len(faces)),
                  bounding_box_min=vertices.min(axis=0).tolist(),
                  bounding_box_max=vertices.max(axis=0).tolist(),
                  units='meters', coordinate_system='GLIM_world', voxel_size_m=settings['voxel_size_m'],
                  sdf_trunc_m=settings['sdf_trunc_m'], space_carving=settings['space_carving'],
                  geometry_audit=audit_mesh(vertices, faces, settings, observed_bbox))
    if extra:
        report.update(extra)
    return report


def write_mesh_ply(path, vertices, faces):
    from .nksr_mesh import write_mesh
    return write_mesh(path, vertices, faces)


PUBLISH_MANIFEST = 'publish_manifest.json'
# A published artifact larger than this is recorded by size only: hashing a multi-gigabyte
# mesh would cost minutes for no operational benefit.
PUBLISH_HASH_LIMIT_BYTES = 256 * 1024 ** 2
PUBLISH_BACKUP_SUFFIX = 'previous-'
PUBLISH_TRANSACTION_NOTE = (
    'Each artifact is replaced atomically on its own: a file is swapped with one rename that '
    'never unlinks the previous version first, and a directory is moved aside before the new '
    'one is moved in, with the previous one restored if the swap fails. The set as a whole is '
    'therefore not a single transaction - between the two renames a reader can see the new '
    'mesh next to the previous chunk directory - and the previous artifacts are only deleted '
    'after every rename succeeded. This manifest, written last, records the artifacts that '
    'belong to the completed run.')


def output_artifact_names(mode):
    """The artifacts a mesh output mode must publish, directories first."""
    if mode not in MESH_OUTPUT_MODES:
        raise ValueError(f'Mesh output mode must be one of {", ".join(MESH_OUTPUT_MODES)}')
    names = []
    if mode != 'merged':
        names.append('mesh_chunks')
    if mode != 'chunks':
        names.append('mesh.ply')
    return names


def artifact_size_bytes(path):
    """Content size of an artifact: a file's size, or the sum of a directory's files.

    ``Path.stat().st_size`` on a directory is the size of its inode entry, not of its
    contents, so directory artifacts must be measured by summing their files on both sides of
    the comparison.
    """
    path = Path(path)
    if path.is_dir():
        return int(sum(item.stat().st_size for item in path.rglob('*') if item.is_file()))
    return int(path.stat().st_size)


def _artifact_digest(path):
    """Size, file count and - for reasonably sized files - a content hash."""
    path = Path(path)
    if path.is_dir():
        files = sorted(item for item in path.rglob('*') if item.is_file())
        return dict(kind='directory', bytes=artifact_size_bytes(path),
                    file_count=len(files), sha256=None)
    size = path.stat().st_size
    digest = None
    if size <= PUBLISH_HASH_LIMIT_BYTES:
        hasher = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    return dict(kind='file', bytes=int(size), file_count=1, sha256=digest)


def validate_staged_outputs(staging, mode, expected=None, metadata=None):
    """Validate every staged artifact before anything is published.

    A corrupt or incomplete staged set must never replace a valid published one, so the whole
    set is validated first: the merged PLY is re-read with the independent validator and the
    chunk set is validated through the shared chunk-manifest validator. ``expected`` optionally
    pins the merged mesh counts from the in-memory mesh that produced it.
    """
    staging = Path(staging)
    if not staging.is_dir():
        raise ValueError('Staged output directory is missing; nothing to publish')
    names = output_artifact_names(mode)
    missing = [name for name in names if not (staging/name).exists()]
    if missing:
        raise ValueError(f'Staged output set is incomplete: {", ".join(missing)} missing')
    stats = {}
    for name in names:
        if name == 'mesh_chunks':
            from .nksr_jobs import validate_chunks_completed
            # The shared validator also compares the manifest against the worker metadata, so
            # the same metadata that described the export is used here.
            merged = dict(metadata or {})
            merged.setdefault('mesh_output_mode', mode)
            stats['mesh_chunks'] = dict(validate_chunks_completed(staging, merged) or {})
        else:
            stats['mesh.ply'] = validate_written_mesh(staging/name, expected) if expected else \
                inspect_mesh_shared(staging/name)
    return stats


def inspect_mesh_shared(path):
    from .nksr_mesh import inspect_mesh
    return inspect_mesh(path)


def publish_output_set(staging, destination, mode='merged', expected=None, metadata=None):
    """Publish a validated staged output set, then record a completion manifest.

    Order and guarantees:

    1. every artifact of the requested mode must be present and validate while still staged;
    2. directories are moved aside and files are replaced with a single atomic rename, so a
       previous valid artifact is never unlinked before its replacement exists;
    3. on any failure the previous artifacts are restored and the staged set is kept for retry;
    4. only after every rename succeeded are the backups deleted and the manifest written.

    See :data:`PUBLISH_TRANSACTION_NOTE` for the documented cross-artifact limitation.
    """
    staging = Path(staging)
    destination = Path(destination)
    names = output_artifact_names(mode)
    validate_staged_outputs(staging, mode, expected, metadata)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    backups = []
    published = []

    def rollback():
        """Put the previous artifacts back; a failed publish must not lose them.

        Returns the backups that could not be restored, so an unrecoverable swap is reported
        with the exact path an operator can recover from instead of failing silently.
        """
        stranded = []
        for target, backup in reversed(backups):
            try:
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True) if target.is_dir() else target.unlink()
                os.replace(backup, target)
            except OSError:
                stranded.append((str(target), str(backup)))
        return stranded

    try:
        for name in names:
            source = staging/name
            target = destination/name
            artifacts[name] = _artifact_digest(source)
            if source.is_dir():
                backup = destination/f'{name}.{PUBLISH_BACKUP_SUFFIX}{uuid.uuid4().hex[:12]}'
                if target.exists():
                    os.replace(target, backup)
                    backups.append((target, backup))
                os.replace(source, target)
            else:
                # A single atomic rename: the previous file is replaced in place, never
                # unlinked first, so a failure leaves it exactly as it was.
                os.replace(source, target)
            published.append(name)
    except OSError as error:
        stranded = rollback()
        detail = (f'; the previous output could not be restored from {", ".join(backup for _, backup in stranded)}'
                  if stranded else '')
        raise OSError(error.errno, f'Publishing the mesh output set failed: {error.strerror}{detail}') from error
    except Exception:
        stranded = rollback()
        if stranded:
            raise RuntimeError(
                'Publishing failed and the previous output could not be restored from '
                + ', '.join(backup for _, backup in stranded)) from None
        raise
    for _, backup in backups:
        if backup.is_dir():
            shutil.rmtree(backup, ignore_errors=True)
        else:
            backup.unlink(missing_ok=True)
    shutil.rmtree(staging, ignore_errors=True)
    manifest = dict(mode=mode, published_at=time.time(), artifacts=artifacts, published=published,
                    transaction_note=PUBLISH_TRANSACTION_NOTE,
                    hash_limit_bytes=PUBLISH_HASH_LIMIT_BYTES)
    _write_manifest(destination/PUBLISH_MANIFEST, manifest)
    return manifest


def _write_manifest(path, manifest):
    """Write the manifest atomically: a reader never sees a half-written manifest."""
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.partial')
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(temporary, path)


def validate_written_mesh(path, expected):
    """Re-read a written PLY with the independent validator and compare counts."""
    from .nksr_mesh import inspect_mesh
    stats = inspect_mesh(path)
    if stats['vertex_count'] != expected['vertex_count'] or stats['face_count'] != expected['face_count']:
        raise ValueError('Written mesh does not match the validated mesh counts')
    for limit in range(3):
        if not math.isfinite(stats['bounding_box_min'][limit]) or not math.isfinite(stats['bounding_box_max'][limit]):
            raise ValueError('Written mesh has non-finite bounds')
    return stats
