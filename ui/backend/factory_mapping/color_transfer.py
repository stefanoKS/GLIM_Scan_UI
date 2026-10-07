"""Optional color transfer from the master colored point cloud.

Transfers RGB onto the exact GLIM PLY geometry and onto NKSR mesh vertices
without ever modifying positions or topology. New outputs only.

Both transfers use a SciPy ``cKDTree``; there is deliberately no quadratic
fallback. The search radius is explicit and conservative: large radii can mix
color across two sides of a thin wall or panel, because this is a Euclidean
nearest-neighbor transfer without normal awareness.
"""
from pathlib import Path

import numpy as np

FALLBACK_COLOR = (128, 128, 128)

# A clearly unrelated target is rejected before any color is written. The guard
# is deliberately loose: it catches mismatched runs and coordinate frames, not
# legitimate differences between an optimized map and the raw-colorized cloud.
SANITY_MIN_SAMPLE_COVERAGE = 0.01
SANITY_SAMPLE_LIMIT = 20_000


def _cKDTree():
    """Import SciPy's KD-tree, or explain exactly what is missing."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            'SciPy is required for color transfer (scipy.spatial.cKDTree). Install the pinned '
            'project requirements, for example ".venv/bin/pip install -r requirements.lock" on a '
            'desktop or scripts/bootstrap_jetson.sh on Jetson. No slower fallback is provided.'
        ) from error
    return cKDTree


def _sanity_check(query_points, src_points, index, radius, sanity):
    """Reject targets whose geometry is clearly unrelated to the colored cloud.

    Two cheap, deliberately loose checks: the bounding boxes must not be
    disjoint, and at least a small sample fraction of target vertices must have a
    colored neighbor inside the transfer radius. A legitimate optimized map or
    surface is almost entirely covered by a few centimetres; an unrelated run is
    covered by essentially nothing.
    """
    stats = dict(sanity_checked=bool(sanity), sanity_bounding_box_overlap=bool(sanity),
                 sanity_median_nearest_distance=0.0, sanity_sample_coverage=1.0,
                 sanity_min_sample_coverage=SANITY_MIN_SAMPLE_COVERAGE)
    if not sanity or not len(query_points):
        return stats
    lower, upper = src_points.min(axis=0), src_points.max(axis=0)
    q_lower, q_upper = query_points.min(axis=0), query_points.max(axis=0)
    overlap_min, overlap_max = np.maximum(lower, q_lower), np.minimum(upper, q_upper)
    # Degenerate axes (planar or linear targets) compare as touching, not as a gap.
    stats['sanity_bounding_box_overlap'] = bool(np.all(overlap_max >= overlap_min))
    sample = query_points[::max(1, len(query_points)//SANITY_SAMPLE_LIMIT)][:SANITY_SAMPLE_LIMIT]
    nearest, _ = index.query(sample, k=1, workers=1)
    coverage = float(np.mean(nearest <= radius)) if len(nearest) else 0.0
    stats['sanity_median_nearest_distance'] = float(np.median(nearest)) if len(nearest) else 0.0
    stats['sanity_sample_coverage'] = coverage
    if not stats['sanity_bounding_box_overlap']:
        raise ValueError('Target geometry does not overlap the colored master cloud; the run lineage '
                         'or coordinate frame differs. Transfer aborted before writing any output.')
    if not np.isfinite(stats['sanity_median_nearest_distance']) or coverage < SANITY_MIN_SAMPLE_COVERAGE:
        raise ValueError(
            f'Only {100*coverage:.2f}% of target vertices have a colored point within the {radius:g} m '
            f'transfer radius (median nearest distance {stats["sanity_median_nearest_distance"]:.3f} m); '
            'this target does not belong to the colorized run. Transfer aborted.')
    return stats


def _read_ply(path):
    from plyfile import PlyData
    mesh = PlyData.read(str(path))
    if 'vertex' not in mesh:
        raise ValueError('PLY has no vertex element')
    return mesh


def _validate_source(colored_points, cd, confidence):
    colored_points = np.asarray(colored_points, dtype=np.float64)
    cd = np.asarray(cd, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    if colored_points.ndim != 2 or colored_points.shape[1] != 3:
        raise ValueError('Colored points must be [N,3]')
    if cd.shape != (len(colored_points), 3) or confidence.shape != (len(colored_points),):
        raise ValueError('Cd and confidence must match the point count')
    if not np.isfinite(colored_points).all():
        raise ValueError('Colored points contain non-finite coordinates')
    return colored_points, cd, confidence


def transfer_colors_to_points(source_ply, colored_points, cd, confidence, output_ply,
                              radius=0.025, k=5, progress=print, fallback_color=FALLBACK_COLOR,
                              sanity=True):
    """Transfer RGB to a vertex PLY (e.g. the official GLIM export).

    GLIM geometry is preserved exactly; only ``red``/``green``/``blue`` are added.
    """
    mesh = _read_ply(source_ply)
    vertex = mesh['vertex'].data
    points = np.column_stack((vertex['x'], vertex['y'], vertex['z'])).astype(np.float64)
    colored_points, cd, confidence = _validate_source(colored_points, cd, confidence)
    output = Path(output_ply)
    if output.resolve() == Path(source_ply).resolve():
        raise ValueError('Refusing to overwrite the source PLY; choose a new output path')
    rgb_u8, stats = _transfer(points, colored_points, cd, confidence, radius, k, progress,
                              fallback_color, sanity)

    names = list(vertex.dtype.names) + ['red', 'green', 'blue']
    formats = [vertex.dtype[name] for name in vertex.dtype.names] + ['u1', 'u1', 'u1']
    out = np.empty(len(vertex), dtype=list(zip(names, formats)))
    for name in vertex.dtype.names:
        out[name] = vertex[name]
    out['red'] = rgb_u8[:, 0]
    out['green'] = rgb_u8[:, 1]
    out['blue'] = rgb_u8[:, 2]
    _assert_attributes_preserved(out, vertex)

    from plyfile import PlyData, PlyElement
    output.parent.mkdir(parents=True, exist_ok=True)
    elements = [PlyElement.describe(out, 'vertex')]
    for element in mesh.elements:
        if element.name != 'vertex':
            elements.append(element)
    PlyData(elements, text=False, byte_order='<').write(str(output))
    stats['output_file'] = str(output)
    stats['source_file'] = str(source_ply)
    stats['source_vertices_preserved'] = True
    return stats


def transfer_colors_to_mesh(mesh_ply, colored_points, cd, confidence, output_ply,
                            radius=0.025, k=5, progress=print, fallback_color=FALLBACK_COLOR,
                            sanity=True):
    """Transfer RGB to mesh vertices; topology is preserved exactly."""
    mesh = _read_ply(mesh_ply)
    vertex = mesh['vertex'].data
    points = np.column_stack((vertex['x'], vertex['y'], vertex['z'])).astype(np.float64)
    colored_points, cd, confidence = _validate_source(colored_points, cd, confidence)
    output = Path(output_ply)
    if output.resolve() == Path(mesh_ply).resolve():
        raise ValueError('Refusing to overwrite the source PLY; choose a new output path')
    rgb_u8, stats = _transfer(points, colored_points, cd, confidence, radius, k, progress,
                              fallback_color, sanity)

    names = list(vertex.dtype.names) + ['red', 'green', 'blue']
    formats = [vertex.dtype[name] for name in vertex.dtype.names] + ['u1', 'u1', 'u1']
    out = np.empty(len(vertex), dtype=list(zip(names, formats)))
    for name in vertex.dtype.names:
        out[name] = vertex[name]
    out['red'] = rgb_u8[:, 0]
    out['green'] = rgb_u8[:, 1]
    out['blue'] = rgb_u8[:, 2]
    _assert_attributes_preserved(out, vertex)

    from plyfile import PlyData, PlyElement
    output.parent.mkdir(parents=True, exist_ok=True)
    elements = [PlyElement.describe(out, 'vertex')]
    for element in mesh.elements:
        if element.name != 'vertex':
            elements.append(element)
    PlyData(elements, text=False, byte_order='<').write(str(output))
    stats['output_file'] = str(output)
    stats['source_file'] = str(mesh_ply)
    stats['source_vertices_preserved'] = True
    return stats


def _assert_attributes_preserved(out, vertex):
    """Fail loudly rather than write a PLY whose geometry moved."""
    for name in vertex.dtype.names:
        if not np.array_equal(np.asarray(out[name]), np.asarray(vertex[name])):
            raise RuntimeError(f'Color transfer would alter the source attribute {name!r}')


def _transfer(query_points, src_points, src_cd, src_confidence, radius, k, progress, fallback_color,
              sanity=True):
    cKDTree = _cKDTree()

    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Transfer radius must be a positive, finite distance in meters')
    if type(k) is not int or k <= 0:
        raise ValueError('Transfer k must be a positive integer')

    valid = np.isfinite(src_cd).all(axis=1) & (src_confidence > 0)
    if not valid.any():
        raise ValueError('No colored master points available for transfer')
    src_points = src_points[valid]
    src_cd = src_cd[valid]
    src_confidence = src_confidence[valid]

    progress(f'Building KD-tree over {len(src_points):,} colored master points')
    index = cKDTree(src_points, compact_nodes=True, balanced_tree=True)
    sanity_stats = _sanity_check(query_points, src_points, index, radius, sanity)

    neighbors = min(k, len(src_points))
    eps = 1e-12
    rgb = np.empty((len(query_points), 3), dtype=np.float32)
    color_count = np.zeros(len(query_points), dtype=np.int32)
    batch = 100_000
    for start in range(0, len(query_points), batch):
        stop = min(start + batch, len(query_points))
        distances, indices = index.query(query_points[start:stop], k=neighbors, workers=1)
        if neighbors == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        within = distances <= radius
        weights = src_confidence[indices] / np.maximum(distances ** 2, eps)
        weights[~within] = 0.0
        total = weights.sum(axis=1, keepdims=True)
        rgb[start:stop] = (weights[:, :, None] * src_cd[indices]).sum(axis=1) / np.maximum(total, eps)
        color_count[start:stop] = within.sum(axis=1)
        progress(f'Transferring color: {stop:,}/{len(query_points):,} vertices')

    colored = color_count > 0
    rgb_u8 = np.empty((len(query_points), 3), dtype=np.uint8)
    rgb_u8[colored] = np.clip(np.rint(rgb[colored] * 255.0), 0, 255)
    rgb_u8[~colored] = np.asarray(fallback_color, dtype=np.uint8)

    colored_count = int(colored.sum())
    total = len(query_points)
    distances_stats = {}
    if colored_count:
        # Approximate nearest-neighbor distances for colored vertices using one query.
        nearest, _ = index.query(query_points[colored], k=1, workers=1)
        mean_distance = float(nearest.mean())
        distances_stats = dict(neighbor_distance_min=float(nearest.min()),
                               neighbor_distance_max=float(nearest.max()),
                               neighbor_distance_mean=mean_distance,
                               mean_nearest_color_distance=mean_distance)
    stats = dict(vertices_total=total, vertices_colored=colored_count,
                 vertices_uncolored=total - colored_count,
                 percentage_colored=100.0 * colored_count / total if total else 0.0,
                 coverage_percent=100.0 * colored_count / total if total else 0.0,
                 transfer_radius=radius, transfer_k=k, max_transfer_radius=radius,
                 **distances_stats, **sanity_stats)
    return rgb_u8, stats
