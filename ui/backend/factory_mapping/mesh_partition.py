"""Spatial partitioning of a finished fused mesh into world-aligned cubic export cells.

This is the standard export path for Full and Chunked reconstruction: NKSR produces
the final surface once and this module splits it into independent binary PLY files.
Cells are ``effective_chunk_size_m`` cubes on a grid anchored at world ``[0, 0, 0]``,
so the export cell matches the physical NKSR chunk size instead of the overlap
adjusted NKSR stride.

Ownership is decided by the triangle centroid, computed in float64:
``cell = floor(centroid / size)``. Each triangle belongs to exactly one cell and is
never cut, welded, decimated, resampled or retriangulated, so vertex positions,
winding and world metres survive unchanged. A triangle may therefore extend past its
nominal cell, and neighbouring tiles share duplicated vertex coordinates without
being topologically welded.

Memory stays bounded: faces are classified in batches, only the vertices a batch
references are widened to float64 (never the whole source array), per-cell face
references are spooled to disk through a bounded number of open handles, and one cell
at a time is materialized, compacted and written.
"""
from pathlib import Path
import tempfile
import numpy as np
from .nksr_mesh import validate_mesh, write_mesh
from .storage import atomic_json

EXPORT_STRATEGY = 'spatial_split_of_final_mesh'
MANIFEST_VERSION = 2
FACE_BATCH = 65536
SPOOL_HANDLE_LIMIT = 64
# Cells are ordered z-major, then y, then x, so filenames stay deterministic.
CELL_ORDER = '(z, y, x) ascending integer grid indices'

_CELL_DTYPE = np.dtype([('i', '<i8'), ('j', '<i8'), ('k', '<i8')])


def cell_keys(vertices, faces, size, start, stop):
    """Grid cell indices of the centroids of the faces in ``[start, stop)``.

    Only the vertices this batch references are gathered and widened to float64, so the
    peak allocation follows the bounded face batch instead of the whole source mesh.

    Centroids are summed in float64 so a centroid exactly on a grid line floors to
    the cell that starts there, and negative coordinates stay correct because floor
    (not int truncation) rounds towards minus infinity.
    """
    batch = np.asarray(faces)[start:stop]
    # copy=False keeps an already-float64 selection from being duplicated for nothing.
    centroids = np.asarray(vertices)[batch].astype(np.float64, copy=False).mean(axis=1)
    return np.floor(centroids / size).astype(np.int64)


def _groups(keys, start):
    """``(cell, face indices)`` pairs for one batch, in ascending cell order."""
    records = np.ascontiguousarray(keys, dtype='<i8').view(_CELL_DTYPE).reshape(-1)
    unique, inverse = np.unique(records, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    bounds = np.r_[0, np.cumsum(np.bincount(inverse, minlength=len(unique)))]
    return [((int(key['i']), int(key['j']), int(key['k'])), order[bounds[rank]:bounds[rank+1]] + start)
            for rank, key in enumerate(unique)]


def compact_tile(vertices, faces, references):
    """Referenced vertices only, renumbered from zero, with triangle order preserved."""
    triangles = np.asarray(faces)[references]
    used = np.unique(triangles)
    local = np.searchsorted(used, triangles.reshape(-1)).reshape(-1, 3)
    return np.asarray(vertices)[used], local


class _FaceSpool:
    """Append-only disk staging of per-cell face references with bounded handles."""

    def __init__(self, root):
        self.root = Path(root)
        self._handles = {}

    def append(self, index, references):
        handle = self._handles.get(index)
        if handle is None:
            if len(self._handles) >= SPOOL_HANDLE_LIMIT:
                self.close()
            handle = self._handles[index] = (self.path(index)).open('ab')
        handle.write(np.asarray(references, dtype='<i8').tobytes())

    def path(self, index):
        return self.root / f'faces_{index:06d}.bin'

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def _validate_size(size):
    if not (np.isfinite(size) and size > 0):
        raise ValueError(f'Effective chunk size must be a positive number of metres, not {size!r}')
    return float(size)


def partition_mesh(vertices, faces, size, output_dir, *, reconstruction_mode, requested_chunk_size_m=None,
                   chunk_size_source='user', event=None):
    """Write one binary PLY per world-aligned cubic export cell.

    Returns ``(manifest, totals)``. ``totals`` mirrors the flat worker metadata keys
    (``total_vertices``, ``total_faces``, ``union_bounds``, ``chunk_count``).
    """
    size = _validate_size(size)
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    validate_mesh(vertices, faces)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError('Chunk output exists; choose a new output directory')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    total = len(faces)
    if event: event('PARTITIONING_MESH', total_faces=total, effective_chunk_size_m=size,
                    message=f'Classifying {total:,} triangles into {size:g} m export cells')
    counts = {}
    for start in range(0, total, FACE_BATCH):
        stop = min(start + FACE_BATCH, total)
        for cell, references in _groups(cell_keys(vertices, faces, size, start, stop), start):
            counts[cell] = counts.get(cell, 0) + len(references)
    # Empty cells never appear here, so no empty PLY is ever written.
    order = sorted(counts, key=lambda cell: (cell[2], cell[1], cell[0]))
    rank = {cell: index for index, cell in enumerate(order)}
    output_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='nksr-partition-', dir=output_dir.parent) as scratch:
        spool = _FaceSpool(scratch)
        for start in range(0, total, FACE_BATCH):
            keys = cell_keys(vertices, faces, size, start, min(start + FACE_BATCH, total))
            for cell, references in _groups(keys, start):
                spool.append(rank[cell], references)
            if event: event('PARTITIONING_MESH', face_batches=min(start + FACE_BATCH, total),
                            total_faces=total, message=f'Classified {min(start + FACE_BATCH, total):,} / {total:,} triangles')
        spool.close()
        chunks = []
        total_vertices = total_faces = 0
        lower = np.full(3, np.inf)
        upper = np.full(3, -np.inf)
        for index, cell in enumerate(order):
            name = f'chunk_{index:04d}.ply'
            if event: event('SAVING_MESH_CHUNKS', chunk=index + 1, total_chunks=len(order),
                            message=f'Writing export cell {index + 1}/{len(order)}')
            references = np.fromfile(spool.path(index), dtype='<i8')
            if len(references) != counts[cell]:
                raise ValueError(f'Export cell {name} staged {len(references)} of {counts[cell]} triangles')
            tile_vertices, tile_faces = compact_tile(vertices, faces, references)
            if len(tile_vertices) > np.iinfo(np.int32).max:
                raise ValueError(f'Export cell {name} exceeds int32 PLY index capacity')
            stats = write_mesh(output_dir / name, tile_vertices, tile_faces)
            total_vertices += stats['vertex_count']
            total_faces += stats['face_count']
            lower = np.minimum(lower, np.asarray(stats['bounding_box_min']))
            upper = np.maximum(upper, np.asarray(stats['bounding_box_max']))
            nominal_min = (np.asarray(cell, dtype=np.float64) * size).tolist()
            chunks.append(dict(index=index, grid_index=list(cell), file=name,
                               nominal_bbox_min_m=nominal_min,
                               nominal_bbox_max_m=(np.asarray(cell, dtype=np.float64) * size + size).tolist(),
                               world_bbox_min=stats['bounding_box_min'],
                               world_bbox_max=stats['bounding_box_max'],
                               vertices=int(stats['vertex_count']), faces=int(stats['face_count']),
                               file_size_bytes=int(stats['mesh_file_size'])))
            del references, tile_vertices, tile_faces
    if total_faces != total:
        raise ValueError(f'Exported {total_faces} triangles but the source mesh has {total}')
    manifest = dict(version=MANIFEST_VERSION, export_strategy=EXPORT_STRATEGY,
                    reconstruction_mode=reconstruction_mode, coordinate_system='GLIM_world', units='meters',
                    tile_shape='xyz_cube', grid_origin_m=[0.0, 0.0, 0.0], cell_order=CELL_ORDER,
                    requested_chunk_size_m=requested_chunk_size_m, effective_chunk_size_m=size,
                    chunk_size_source=chunk_size_source, total_chunks=len(chunks),
                    source_faces=int(total), total_vertices=int(total_vertices), total_faces=int(total_faces),
                    chunks=chunks)
    atomic_json(output_dir / 'chunks.json', manifest)
    return manifest, dict(chunk_count=len(chunks), total_vertices=int(total_vertices), total_faces=int(total_faces),
                          union_bounds=(lower.tolist(), upper.tolist()))
