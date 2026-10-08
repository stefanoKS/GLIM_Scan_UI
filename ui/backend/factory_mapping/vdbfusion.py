"""VDBFusion TSDF surface reconstruction: bounded streaming, real sensor origins.

This module is the single source of truth for the VDBFusion engine. It imports
NumPy and the standard library only; ROS bag reading, the native VDBFusion
library, SciPy and PLY writing are imported lazily so the module stays importable
from the web backend, the preparation subprocess and the isolated worker.

Nothing here touches NKSR, Torch, CUDA or any existing GLIM export. World
coordinates are always metres and every sensor origin comes from the GLIM
trajectory interpolation in :mod:`factory_mapping.reconstruction`.
"""
import math
import time
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
    Bounded by ``batch_points`` per yield: the caller never holds the whole
    recording. Observations outside the trajectory time range are counted in
    ``counters`` and never silently clamped. ``timings`` accumulates per-phase
    seconds, which is what the benchmark CLI reports.
    """
    from .reconstruction import transform_points
    batch_points = validate_batch_points(batch_points)
    counters = counters if counters is not None else {}
    timings = timings if timings is not None else {}
    for key in ('frames', 'observations', 'outside_trajectory'):
        counters.setdefault(key, 0)
    for key in ('transform_seconds', 'batch_seconds'):
        timings.setdefault(key, 0.0)
    buffers = {key: [] for key in ('points', 'origins', 'intensity', 'timestamps')}
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
            for key, values in zip(buffers, (points, origins, intensity[valid], timestamps[valid])):
                buffers[key].append(values)
            pending += len(points)
            counters['observations'] += len(points)
        if pending >= batch_points:
            started = time.perf_counter()
            yield {key: np.concatenate(values) for key, values in buffers.items()}
            timings['batch_seconds'] += time.perf_counter() - started
            for values in buffers.values():
                values.clear()
            pending = 0
        if counters['frames'] % 200 == 0 and progress is not None:
            progress(f'Reading raw bag: {counters["frames"]:,} LiDAR frames, '
                     f'{counters["observations"]:,} world observations')
    if pending:
        yield {key: np.concatenate(values) for key, values in buffers.items()}
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

    def classify(self, points, association_radius_m, boundary_margin_m=0.0):
        """Classify world coordinates against verified kept and removed geometry.

        Returns a dict of boolean masks. A coordinate is kept when verified kept
        geometry is within the association radius and the nearest removed sample is
        not at least as close (minus ``boundary_margin_m``, which biases the edit
        boundary toward exclusion). Coordinates without kept support inside the
        radius are ``unsupported`` and are excluded by default rather than guessed.
        """
        _, tree_retained, tree_deleted = self.trees()
        count = len(points)
        if not count:
            empty = np.zeros(0, dtype=bool)
            return dict(keep=empty, unsupported=empty, removed_dominated=empty, removed_support=empty)
        upper = np.nextafter(float(association_radius_m), math.inf)
        near_retained, _ = tree_retained.query(points, k=1, distance_upper_bound=upper, workers=1)
        retained_support = np.isfinite(near_retained) & (near_retained <= association_radius_m)
        unsupported = ~retained_support
        if tree_deleted is None:
            return dict(keep=retained_support, unsupported=unsupported,
                        removed_dominated=np.zeros(count, dtype=bool),
                        removed_support=np.zeros(count, dtype=bool))
        near_removed, _ = tree_deleted.query(points, k=1, distance_upper_bound=upper, workers=1)
        removed_support = np.isfinite(near_removed) & (near_removed <= association_radius_m)
        removed_dominated = retained_support & removed_support & (near_retained + float(boundary_margin_m) >= near_removed)
        return dict(keep=retained_support & ~removed_dominated, unsupported=unsupported,
                    removed_dominated=removed_dominated, removed_support=removed_support)


def _read_submap_points(path):
    try:
        raw = np.fromfile(path, dtype='<f4')
    except OSError as error:
        raise ValueError(f'Saved cleanup submap {Path(path).parent.name} has no readable point block') from error
    if raw.size == 0 or raw.size % 3:
        raise ValueError(f'Saved cleanup submap {Path(path).parent.name} has an invalid point block')
    return raw.reshape(-1, 3)


def _read_submap_origin(data_txt):
    lines = data_txt.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith('T_world_origin'):
            rows = [np.fromstring(lines[index + 1 + k], sep=' ') for k in range(4)]
            matrix = np.vstack(rows)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError('Saved cleanup submap has an invalid T_world_origin')
            return matrix
    raise ValueError('Saved cleanup submap has no T_world_origin')


def load_edited_reference(workspace, association_radius_m=None, association_spacing_multiplier=None,
                          spacing_sample=DEFAULT_ASSOCIATION_SPACING_SAMPLE):
    """Load and verify the retained/removed reference geometry of a saved cleanup.

    The pre-edit (``map_01``) and saved (``saved_map``) submaps must share the
    same ids, and every retained point must be a byte-exact member of the pre-edit
    submap it came from. Anything else is reported as unsupported, never guessed.
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
    all_blocks, labels, submaps = [], [], []
    retained_total = original_total = retained_bytes = 0
    for name in original_ids:
        original = _read_submap_points(original_dir / name / 'points_compact.bin')
        saved = _read_submap_points(saved_dir / name / 'points_compact.bin')
        original_view = original.view([('x', '<f4'), ('y', '<f4'), ('z', '<f4')]).ravel()
        saved_view = saved.view([('x', '<f4'), ('y', '<f4'), ('z', '<f4')]).ravel()
        keep = np.isin(original_view, saved_view)
        matrix = _read_submap_origin(original_dir / name / 'data.txt')
        rotation = matrix[:3, :3].astype(np.float64)
        translation = matrix[:3, 3].astype(np.float64)
        world = original.astype(np.float64) @ rotation.T + translation
        del original, saved, original_view, saved_view
        all_blocks.append(world)
        labels.append(keep)
        retained_total += int(keep.sum())
        original_total += len(world)
        retained_bytes += int(keep.sum()) * 24
        submaps.append(dict(id=name, original_points=int(len(world)), retained_points=int(keep.sum())))
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


def preflight(root, settings, observed_points=None, observed_bbox=None, bag_bytes=None):
    """Resource preflight: free RAM, free disk, resolution and estimated pressure.

    Streaming bounds the *input* memory, not the TSDF: a fine sparse volume over a
    whole factory still grows with the touched surface. The estimate below is an
    order-of-magnitude figure, not a promise, and is reported so a user can see the
    pressure before committing to a long run.
    """
    import psutil
    free_ram = psutil.virtual_memory().available
    target = Path(root)
    while not target.exists() and target.parent != target:
        target = target.parent
    disk = psutil.disk_usage(str(target))
    voxel = settings['voxel_size_m']
    trunc = settings['sdf_trunc_m']
    extent = band_voxels = touched_voxels = None
    estimate = None
    if observed_bbox is not None:
        lower = np.asarray(observed_bbox[0], dtype=np.float64)
        upper = np.asarray(observed_bbox[1], dtype=np.float64)
        extents = np.maximum(upper - lower, voxel)
        extent = extents.tolist()
        sizes = np.ceil(extents / voxel) + 2 * math.ceil(trunc / voxel)
        band_voxels = float(np.prod(np.maximum(sizes, 1)))
        # Each observation touches at least one voxel, so the observation count caps
        # the touched surface even when the sampled extent is sparse.
        touched_voxels = band_voxels if observed_points is None else min(band_voxels, float(observed_points))
        estimate = int(touched_voxels * TSDF_BYTES_PER_TOUCHED_VOXEL)
    warnings = []
    if estimate is not None and estimate > free_ram:
        warnings.append(
            f'A {voxel * 1000:.0f} mm TSDF over the observed {np.round(extent, 1).tolist()} m extent could touch about '
            f'{estimate / 1024 ** 3:.1f} GiB of voxels, above the {free_ram / 1024 ** 3:.1f} GiB of free RAM. '
            'Reduce the scan extent, use a larger voxel size or use an ROI.')
    if disk.free < 2 * 1024 ** 3:
        warnings.append(f'Only {disk.free / 1024 ** 3:.1f} GiB of free disk space on {target}.')
    return dict(free_ram_bytes=int(free_ram), total_ram_bytes=int(psutil.virtual_memory().total),
                free_disk_bytes=int(disk.free), disk_path=str(target),
                voxel_size_m=voxel, sdf_trunc_m=trunc, observed_extent_m=extent,
                estimated_tsdf_voxels=touched_voxels,
                band_voxels_upper_bound=band_voxels,
                estimated_tsdf_bytes=estimate,
                bytes_per_touched_voxel=TSDF_BYTES_PER_TOUCHED_VOXEL,
                estimate_note='Order-of-magnitude estimate from the sampled extent and observation count. '
                              'Streaming bounds input memory only; the sparse TSDF itself is not memory bounded.',
                observed_points=None if observed_points is None else int(observed_points),
                bag_bytes=None if bag_bytes is None else int(bag_bytes), warnings=warnings)


def _format_bytes(value):
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if abs(value) < 1024 or unit == 'TiB':
            return f'{value:.1f} {unit}'
        value /= 1024


class MemoryBudgetExceeded(RuntimeError):
    """The configured soft memory budget was reached; fail cleanly, never re-resolve."""


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
    budget = int(memory_budget_bytes or DEFAULT_MEMORY_BUDGET_BYTES)
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
        if peak_limit_bytes is not None and rss > peak_limit_bytes:
            raise MemoryBudgetExceeded(
                f'Worker RSS {_format_bytes(rss)} exceeded the configured memory budget '
                f'{_format_bytes(peak_limit_bytes)} while integrating. Reduce the voxel size request or the scan '
                'extent; the requested resolution was not changed.')
        if peak_limit_bytes is None and rss > budget:
            raise MemoryBudgetExceeded(
                f'Worker RSS {_format_bytes(rss)} exceeded the default memory budget {_format_bytes(budget)}. '
                'Reduce the voxel size request or the scan extent.')
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


def extract_and_mask(volume, edit_reference=None, association_radius_m=None, boundary_margin_m=0.0,
                     event=None, mask_deleted_triangles=True):
    """Extract the fused triangle mesh and optionally mask removed regions.

    Removing observations is necessary but not sufficient: a retained ray can
    still bridge a small deleted gap. Triangle centroids are therefore classified
    with the same rule as the observations, so removed geometry is never
    recreated by meshing.
    """
    event = event or (lambda stage, **fields: None)
    started = time.monotonic()
    vertices, faces = volume.extract_triangle_mesh()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    stats = dict(extract_seconds=time.monotonic() - started, vertex_count=len(vertices), face_count=len(faces),
                 masked_triangles=0, mask_enabled=False)
    if not len(vertices) or not len(faces):
        return vertices, faces, stats
    if edit_reference is not None and mask_deleted_triangles and len(faces):
        stats['mask_enabled'] = True
        kept = []
        batch = 2_000_000
        for start in range(0, len(faces), batch):
            stop = min(start + batch, len(faces))
            centroids = vertices[faces[start:stop]].mean(axis=1)
            classified = edit_reference.classify(centroids, association_radius_m, boundary_margin_m)
            kept.append(np.flatnonzero(classified['keep']) + start)
            event('VALIDATING_MESH', masked_faces=stats['masked_triangles'],
                  message=f'Checking triangles against the removed regions: {stop:,}/{len(faces):,}')
        keep_all = np.concatenate(kept) if kept else np.zeros(0, dtype=np.int64)
        stats['masked_triangles'] = int(len(faces) - len(keep_all))
        faces = faces[keep_all]
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


def validate_and_report(vertices, faces, settings, extra=None):
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
                  sdf_trunc_m=settings['sdf_trunc_m'], space_carving=settings['space_carving'])
    if extra:
        report.update(extra)
    return report


def write_mesh_ply(path, vertices, faces):
    from .nksr_mesh import write_mesh
    return write_mesh(path, vertices, faces)


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
