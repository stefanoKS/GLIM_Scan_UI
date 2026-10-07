"""Optional color transfer from the master colored point cloud.

Transfers RGB onto the exact GLIM PLY geometry and onto NKSR mesh vertices
without ever modifying positions or topology. New outputs only.
"""
from pathlib import Path

import numpy as np

FALLBACK_COLOR = (128, 128, 128)


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
                              radius=0.025, k=5, progress=print, fallback_color=FALLBACK_COLOR):
    """Transfer RGB to a vertex PLY (e.g. the official GLIM export).

    GLIM geometry is preserved exactly; only ``red``/``green``/``blue`` are added.
    """
    mesh = _read_ply(source_ply)
    vertex = mesh['vertex'].data
    points = np.column_stack((vertex['x'], vertex['y'], vertex['z'])).astype(np.float64)
    colored_points, cd, confidence = _validate_source(colored_points, cd, confidence)
    rgb_u8, stats = _transfer(points, colored_points, cd, confidence, radius, k, progress, fallback_color)

    names = list(vertex.dtype.names) + ['red', 'green', 'blue']
    formats = [vertex.dtype[name] for name in vertex.dtype.names] + ['u1', 'u1', 'u1']
    out = np.empty(len(vertex), dtype=list(zip(names, formats)))
    for name in vertex.dtype.names:
        out[name] = vertex[name]
    out['red'] = rgb_u8[:, 0]
    out['green'] = rgb_u8[:, 1]
    out['blue'] = rgb_u8[:, 2]

    from plyfile import PlyData, PlyElement
    output = Path(output_ply)
    output.parent.mkdir(parents=True, exist_ok=True)
    elements = [PlyElement.describe(out, 'vertex')]
    for element in mesh.elements:
        if element.name != 'vertex':
            elements.append(element)
    PlyData(elements, text=False, byte_order='<').write(str(output))
    stats['output_file'] = str(output)
    stats['source_file'] = str(source_ply)
    return stats


def transfer_colors_to_mesh(mesh_ply, colored_points, cd, confidence, output_ply,
                            radius=0.025, k=5, progress=print, fallback_color=FALLBACK_COLOR):
    """Transfer RGB to mesh vertices; topology is preserved exactly."""
    mesh = _read_ply(mesh_ply)
    vertex = mesh['vertex'].data
    points = np.column_stack((vertex['x'], vertex['y'], vertex['z'])).astype(np.float64)
    colored_points, cd, confidence = _validate_source(colored_points, cd, confidence)
    rgb_u8, stats = _transfer(points, colored_points, cd, confidence, radius, k, progress, fallback_color)

    names = list(vertex.dtype.names) + ['red', 'green', 'blue']
    formats = [vertex.dtype[name] for name in vertex.dtype.names] + ['u1', 'u1', 'u1']
    out = np.empty(len(vertex), dtype=list(zip(names, formats)))
    for name in vertex.dtype.names:
        out[name] = vertex[name]
    out['red'] = rgb_u8[:, 0]
    out['green'] = rgb_u8[:, 1]
    out['blue'] = rgb_u8[:, 2]

    from plyfile import PlyData, PlyElement
    output = Path(output_ply)
    output.parent.mkdir(parents=True, exist_ok=True)
    elements = [PlyElement.describe(out, 'vertex')]
    for element in mesh.elements:
        if element.name != 'vertex':
            elements.append(element)
    PlyData(elements, text=False, byte_order='<').write(str(output))
    stats['output_file'] = str(output)
    stats['source_file'] = str(mesh_ply)
    return stats


def _transfer(query_points, src_points, src_cd, src_confidence, radius, k, progress, fallback_color):
    from scipy.spatial import cKDTree

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
        distances_stats = dict(neighbor_distance_min=float(nearest.min()),
                               neighbor_distance_max=float(nearest.max()),
                               neighbor_distance_mean=float(nearest.mean()))
    stats = dict(vertices_total=total, vertices_colored=colored_count,
                 vertices_uncolored=total - colored_count,
                 percentage_colored=100.0 * colored_count / total if total else 0.0,
                 transfer_radius=radius, transfer_k=k, **distances_stats)
    return rgb_u8, stats
