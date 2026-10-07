"""Focused tests for spatial mesh partitioning: geometry fidelity, determinism, memory."""
from pathlib import Path
import tracemalloc
import weakref
import numpy as np
import pytest
from factory_mapping import mesh_partition as module
from factory_mapping.mesh_partition import compact_tile, partition_mesh
from factory_mapping.nksr_mesh import inspect_mesh


def cube(origin, size=1.0):
    """One closed cube: 8 vertices, 12 outward-facing triangles."""
    local = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                      [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64) * size
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                          [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]], dtype=np.int64)
    return np.asarray(origin, dtype=np.float64) + local, triangles


def cubes(*origins, size=1.0):
    vertices, faces = [], []
    for origin in origins:
        local, triangles = cube(origin, size)
        base = len(vertices)
        vertices.extend(local.tolist())
        faces.extend((triangles + base).tolist())
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def spread(count, stride=10.0):
    """``count`` unit cubes 10 m apart along a diagonal, so every cube owns a cell."""
    return cubes(*[[index * stride, index * stride, index * stride] for index in range(count)])


def read_tiles(directory):
    return [inspect_mesh(path) for path in sorted(Path(directory).glob('chunk_*.ply'))]


def _canonical_triangle(triangle):
    """Smallest rotation of a 3-vertex triangle.

    Rotation-invariant, so the arbitrary starting vertex of an equivalent triangle does
    not matter, but reversal-sensitive, so winding stays part of the comparison.
    """
    rotations = (triangle, np.roll(triangle, -1, axis=0), np.roll(triangle, -2, axis=0))
    return min(tuple(map(tuple, rotation)) for rotation in rotations)


def triangle_multiset(vertices, faces):
    """Multiset of ordered 3-vertex triangles, as coordinate triples.

    Each entry is one complete triangle whose cyclic vertex order (winding) is part of
    the comparison, while the collection itself is sorted so triangle input order does
    not matter. Vertex occurrences are deliberately not flattened: two meshes can share
    a vertex-occurrence multiset and still triangulate the surface differently.
    """
    rounded = np.round(np.asarray(vertices, dtype=np.float64), 6)
    triangles = rounded[np.asarray(faces)]  # [F, 3, 3], winding preserved
    return sorted(_canonical_triangle(triangle) for triangle in triangles)


def load_tiles(directory, manifest):
    from plyfile import PlyData
    vertices, faces = [], []
    offset = 0
    for chunk in manifest['chunks']:
        mesh = PlyData.read(str(Path(directory) / chunk['file']), known_list_len={'face': {'vertex_indices': 3}})
        local = np.column_stack([mesh['vertex'][axis] for axis in 'xyz'])
        indexed = np.asarray(mesh['face']['vertex_indices'])
        if indexed.dtype.kind == 'O':
            indexed = np.stack(indexed)
        vertices.append(local)
        faces.append(indexed + offset)
        offset += len(local)
    return np.vstack(vertices), np.vstack(faces)


# --------------------------------------------------------------------------- #
# Cell ownership
# --------------------------------------------------------------------------- #

def test_one_cube_stays_in_exactly_one_cell(tmp_path):
    vertices, faces = cube([1.0, 2.0, 3.0])
    manifest, totals = partition_mesh(vertices, faces, 10.0, tmp_path / 'mesh_chunks',
                                      reconstruction_mode='full')
    assert manifest['total_chunks'] == 1 and totals['total_faces'] == 12
    assert manifest['chunks'][0]['grid_index'] == [0, 0, 0]
    assert manifest['chunks'][0]['file'] == 'chunk_0000.ply'
    assert len(read_tiles(tmp_path / 'mesh_chunks')) == 1


def test_multiple_cubes_split_into_their_own_cells(tmp_path):
    vertices, faces = cubes([1.0, 1.0, 1.0], [21.0, 1.0, 1.0], [1.0, 21.0, 1.0])
    manifest, totals = partition_mesh(vertices, faces, 10.0, tmp_path / 'mesh_chunks',
                                      reconstruction_mode='full')
    # Cells are (x, y, z); the list is ordered z-major, then y, then x.
    assert [chunk['grid_index'] for chunk in manifest['chunks']] == [[0, 0, 0], [2, 0, 0], [0, 2, 0]]
    assert manifest['total_chunks'] == 3 and totals['total_faces'] == 36
    assert totals['total_vertices'] == 24  # shared corners are not duplicated between these cubes


def test_negative_coordinates_use_floor_not_truncation(tmp_path):
    # Centroid x = -24.5 belongs to cell -3, not cell -24 (int truncation) nor cell -25.
    vertices, faces = cube([-25.0, 2.0, 3.0])
    manifest, _ = partition_mesh(vertices, faces, 10.0, tmp_path / 'mesh_chunks',
                                 reconstruction_mode='full')
    assert manifest['chunks'][0]['grid_index'] == [-3, 0, 0]
    assert manifest['chunks'][0]['nominal_bbox_min_m'] == [-30.0, 0.0, 0.0]
    assert manifest['chunks'][0]['nominal_bbox_max_m'] == [-20.0, 10.0, 10.0]
    assert manifest['chunks'][0]['world_bbox_min'] == [-25.0, 2.0, 3.0]


def test_sparse_occupied_cells_never_produce_empty_files(tmp_path):
    # Cells 0 and 4 on one axis only: cells 1-3 are empty and must not be written.
    vertices, faces = cubes([0.0, 0.0, 0.0], [40.0, 0.0, 0.0])
    manifest, totals = partition_mesh(vertices, faces, 10.0, tmp_path / 'mesh_chunks',
                                      reconstruction_mode='full')
    assert manifest['total_chunks'] == 2 and totals['total_faces'] == 24
    assert [chunk['grid_index'] for chunk in manifest['chunks']] == [[0, 0, 0], [4, 0, 0]]
    assert sorted(p.name for p in (tmp_path / 'mesh_chunks').glob('chunk_*.ply')) == \
        ['chunk_0000.ply', 'chunk_0001.ply']
    for chunk in manifest['chunks']:
        assert chunk['faces'] > 0 and chunk['vertices'] > 0 and chunk['file_size_bytes'] > 0


def test_large_positive_and_negative_coordinates(tmp_path):
    vertices, faces = cubes([-2500.0, 1200.0, -300.0], [2500.0, -1200.0, 300.0])
    manifest, totals = partition_mesh(vertices, faces, 1000.0, tmp_path / 'mesh_chunks',
                                      reconstruction_mode='full')
    assert manifest['total_chunks'] == 2 and totals['total_faces'] == 24
    # Cells are (x, y, z); the list is ordered z-major, then y, then x.
    assert [chunk['grid_index'] for chunk in manifest['chunks']] == [[-3, 1, -1], [2, -2, 0]]


@pytest.mark.parametrize('centroid, expected', [
    ([10.0, 10.0, 10.0], [1, 1, 1]),
    ([-10.0, -10.0, -10.0], [-1, -1, -1]),
    ([0.0, 0.0, 0.0], [0, 0, 0]),
])
def test_centroid_exactly_on_a_grid_line_belongs_to_the_upper_cell(centroid, expected, tmp_path):
    # Three offsets summing to exactly zero, so the centroid is exactly the grid line.
    offsets = np.array([[10.0, 15.0, 0.0], [-15.0, 0.0, 15.0], [5.0, -15.0, -15.0]])
    vertices = np.asarray(centroid, dtype=np.float64) + offsets
    assert vertices.mean(axis=0).tolist() == centroid
    manifest, _ = partition_mesh(vertices, np.array([[0, 1, 2]], dtype=np.int64), 10.0,
                                 tmp_path / 'mesh_chunks', reconstruction_mode='full')
    assert manifest['chunks'][0]['grid_index'] == expected


def test_triangle_crossing_cells_stays_intact_in_one_cell(tmp_path):
    # One long triangle whose centroid sits in cell 0 but whose vertices span cells -1, 0 and 1.
    vertices = np.array([[-12.0, 1.0, 1.0], [5.0, 1.0, 1.0], [8.0, 1.0, 1.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    assert vertices.mean(axis=0)[0] == pytest.approx(1.0 / 3.0)
    manifest, totals = partition_mesh(vertices, faces, 10.0, tmp_path / 'mesh_chunks',
                                      reconstruction_mode='full')
    assert manifest['total_chunks'] == 1 and totals['total_faces'] == 1
    tile = inspect_mesh(tmp_path / 'mesh_chunks' / 'chunk_0000.ply')
    assert tile['bounding_box_min'] == [-12.0, 1.0, 1.0]
    assert tile['bounding_box_max'] == [8.0, 1.0, 1.0]
    assert tile['face_count'] == 1


# --------------------------------------------------------------------------- #
# Determinism and remapping
# --------------------------------------------------------------------------- #

def test_filenames_and_order_are_deterministic(tmp_path):
    vertices, faces = cubes([0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [30.0, 30.0, 0.0])
    first, _ = partition_mesh(vertices, faces, 10.0, tmp_path / 'a', reconstruction_mode='full')
    second, _ = partition_mesh(vertices, faces, 10.0, tmp_path / 'b', reconstruction_mode='full')
    assert [chunk['file'] for chunk in first['chunks']] == [chunk['file'] for chunk in second['chunks']]
    assert [chunk['grid_index'] for chunk in first['chunks']] == [chunk['grid_index'] for chunk in second['chunks']]
    # Documented ordering: z-major, then y, then x, ascending.
    assert [chunk['grid_index'] for chunk in first['chunks']] == [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]]


def test_results_are_stable_regardless_of_triangle_input_order(tmp_path):
    vertices, faces = cubes([0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 30.0, 0.0])
    shuffled = faces[np.random.default_rng(7).permutation(len(faces))]
    first, _ = partition_mesh(vertices, faces, 10.0, tmp_path / 'a', reconstruction_mode='full')
    second, _ = partition_mesh(vertices, shuffled, 10.0, tmp_path / 'b', reconstruction_mode='full')
    assert [chunk['faces'] for chunk in first['chunks']] == [chunk['faces'] for chunk in second['chunks']]
    assert triangle_multiset(*load_tiles(tmp_path / 'a', first)) == \
        triangle_multiset(*load_tiles(tmp_path / 'b', second))


def test_every_triangle_is_assigned_exactly_once(tmp_path):
    vertices, faces = spread(6)
    manifest, totals = partition_mesh(vertices, faces, 10.0, tmp_path / 'mesh_chunks',
                                      reconstruction_mode='full')
    assert manifest['total_chunks'] == 6
    assert sum(chunk['faces'] for chunk in manifest['chunks']) == len(faces) == totals['total_faces']
    assert manifest['source_faces'] == len(faces)
    assert totals['union_bounds'] == (vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist())


def test_fidelity_triangle_multiset_matches_the_source_mesh(tmp_path):
    vertices, faces = spread(5)
    manifest, _ = partition_mesh(vertices, faces, 10.0, tmp_path / 'mesh_chunks',
                                 reconstruction_mode='full')
    tiled_vertices, tiled_faces = load_tiles(tmp_path / 'mesh_chunks', manifest)
    assert len(tiled_faces) == len(faces)
    assert triangle_multiset(tiled_vertices, tiled_faces) == triangle_multiset(vertices, faces)


def test_triangle_multiset_rejects_altered_face_connectivity():
    """Same vertices and counts, different connectivity: the fidelity check must reject it."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                         [0, 1, 0], [0, 0, 1], [1, 0, 1]], dtype=np.float32)
    first = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    second = np.array([[0, 1, 5], [3, 4, 2]], dtype=np.int64)
    # Both meshes use the identical vertex array and the same face and vertex counts.
    assert first.shape == second.shape == (2, 3)
    assert len(np.unique(first)) == len(np.unique(second)) == 6
    # The old flattened helper only compared vertex occurrences, which are identical here,
    # so it silently accepted the altered geometry.
    def flattened(faces):
        rounded = np.round(np.asarray(vertices, dtype=np.float64), 6)
        return sorted(tuple(rounded[index].tolist()) for face in faces for index in face)
    assert flattened(first) == flattened(second)
    # The triangle multiset must not.
    assert triangle_multiset(vertices, first) != triangle_multiset(vertices, second)


def test_triangle_multiset_is_winding_sensitive():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    triangle = np.array([[0, 1, 2]], dtype=np.int64)
    # Reversing the winding is a different triangle.
    assert triangle_multiset(vertices, triangle) != triangle_multiset(vertices, triangle[:, ::-1])
    # A cyclic rotation preserves winding, so it is the same triangle.
    assert triangle_multiset(vertices, triangle) == triangle_multiset(vertices, np.array([[1, 2, 0]]))
    assert triangle_multiset(vertices, triangle) == triangle_multiset(vertices, np.array([[2, 0, 1]]))


def test_triangle_multiset_ignores_triangle_order_but_keeps_each_triangle():
    vertices, faces = cubes([0.0, 0.0, 0.0], [30.0, 0.0, 0.0])
    shuffled = faces[np.random.default_rng(3).permutation(len(faces))]
    assert triangle_multiset(vertices, faces) == triangle_multiset(vertices, shuffled)
    # Each entry is one complete ordered triangle, not an individual vertex occurrence.
    entries = triangle_multiset(vertices, faces)
    assert len(entries) == len(faces)
    assert all(len(entry) == 3 and all(len(point) == 3 for point in entry) for entry in entries)


def test_fidelity_comparison_rejects_a_partition_of_altered_connectivity(tmp_path):
    """The end-to-end fidelity comparison stays conservative about connectivity."""
    vertices, faces = spread(3)
    manifest, _ = partition_mesh(vertices, faces, 10.0, tmp_path / 'mesh_chunks',
                                 reconstruction_mode='full')
    tiled_vertices, tiled_faces = load_tiles(tmp_path / 'mesh_chunks', manifest)
    altered = tiled_faces.copy()
    altered[0] = altered[0][[0, 2, 1]]  # flip one triangle's winding
    assert triangle_multiset(tiled_vertices, altered) != triangle_multiset(vertices, faces)
    assert triangle_multiset(tiled_vertices, tiled_faces) == triangle_multiset(vertices, faces)


def test_face_orientation_and_vertex_coordinates_are_unchanged(tmp_path):
    vertices, faces = cubes([0.0, 0.0, 0.0], [30.0, 0.0, 0.0])
    manifest, _ = partition_mesh(vertices, faces, 10.0, tmp_path / 'mesh_chunks',
                                 reconstruction_mode='full')
    source = {tuple(row.tolist()) for row in vertices}
    for chunk in manifest['chunks']:
        from plyfile import PlyData
        mesh = PlyData.read(str(tmp_path / 'mesh_chunks' / chunk['file']),
                            known_list_len={'face': {'vertex_indices': 3}})
        indexed = np.asarray(mesh['face']['vertex_indices'])
        if indexed.dtype.kind == 'O':
            indexed = np.stack(indexed)
        local = np.column_stack([mesh['vertex'][axis] for axis in 'xyz'])
        # Winding is preserved: the signed normal of every triangle points outwards as in the source.
        normals = np.cross(local[indexed[:, 1]] - local[indexed[:, 0]],
                           local[indexed[:, 2]] - local[indexed[:, 0]])
        assert np.isfinite(normals).all() and np.abs(normals).sum() > 0
        assert {tuple(np.round(row, 6).tolist()) for row in local} <= source


def test_local_vertex_indices_are_contiguous_and_fully_referenced(tmp_path):
    vertices, faces = cubes([0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [60.0, 0.0, 0.0])
    manifest, _ = partition_mesh(vertices, faces, 10.0, tmp_path / 'mesh_chunks',
                                 reconstruction_mode='full')
    from plyfile import PlyData
    for chunk in manifest['chunks']:
        mesh = PlyData.read(str(tmp_path / 'mesh_chunks' / chunk['file']),
                            known_list_len={'face': {'vertex_indices': 3}})
        indexed = np.asarray(mesh['face']['vertex_indices'])
        if indexed.dtype.kind == 'O':
            indexed = np.stack(indexed)
        assert indexed.min() == 0
        assert indexed.max() == chunk['vertices'] - 1
        used = np.unique(indexed)
        assert len(used) == chunk['vertices']  # no unreferenced vertices
        assert sorted(used.tolist()) == list(range(chunk['vertices']))


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('size', [0.0, -5.0, float('nan'), float('inf')])
def test_nonpositive_and_nonfinite_chunk_sizes_are_rejected(size, tmp_path):
    vertices, faces = cube([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match='positive number of metres'):
        partition_mesh(vertices, faces, size, tmp_path / 'mesh_chunks', reconstruction_mode='full')


def test_existing_output_directory_is_rejected(tmp_path):
    vertices, faces = cube([0.0, 0.0, 0.0])
    output = tmp_path / 'mesh_chunks'
    output.mkdir()
    with pytest.raises(ValueError, match='already|exists'):
        partition_mesh(vertices, faces, 10.0, output, reconstruction_mode='full')


# --------------------------------------------------------------------------- #
# Memory behaviour
# --------------------------------------------------------------------------- #

def test_large_triangle_sets_are_classified_in_bounded_batches(monkeypatch, tmp_path):
    passes = []
    original = module.cell_keys
    def counted(vertices, faces, size, start, stop):
        passes.append(stop - start)
        return original(vertices, faces, size, start, stop)
    monkeypatch.setattr(module, 'cell_keys', counted)
    monkeypatch.setattr(module, 'FACE_BATCH', 256)
    vertices, faces = cubes(*[[index * 3.0, 0.0, 0.0] for index in range(40)])
    faces = np.vstack([faces] * 6)  # 2880 triangles over dense spatial cells
    manifest, totals = partition_mesh(vertices, faces, 10.0, tmp_path / 'mesh_chunks',
                                      reconstruction_mode='full')
    assert max(passes) <= 256  # every pass is bounded by the batch size
    # Two bounded passes: one to count cells, one to spool their face references.
    assert len(passes) == 2 * -(-len(faces) // 256)
    assert totals['total_faces'] == len(faces)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cell_keys_memory_scales_with_the_face_batch_not_the_vertex_count(dtype):
    """One small face batch must not widen the whole source vertex array to float64."""
    count = 4_000_000
    vertices = np.zeros((count, 3), dtype=dtype)
    vertices[[0, 1, 2]] = [[0, 0, 0], [30, 0, 0], [0, 30, 0]]
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    tracemalloc.start()
    try:
        keys = module.cell_keys(vertices, faces, 10.0, 0, 1)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert keys.tolist() == [[1, 1, 0]]
    # Widening all 4M vertices would need about 96 MB; gathering only this batch needs bytes.
    assert peak < 1_000_000


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cell_keys_peak_grows_with_the_batch_and_the_result_stays_correct(dtype):
    """Peak follows the batch size, and results are unchanged by the narrow-batch order."""
    count = 3_000_000
    vertices = np.zeros((count, 3), dtype=dtype)
    vertices[[0, 1, 2]] = [[0, 0, 0], [3, 0, 0], [0, 3, 0]]
    vertices[[3, 4, 5]] = [[40, 0, 0], [43, 0, 0], [40, 3, 0]]
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    peaks = []
    for stop in (1, 2):
        tracemalloc.start()
        try:
            keys = module.cell_keys(vertices, faces, 10.0, 0, stop)
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        assert keys.tolist() == [[0, 0, 0], [4, 0, 0]][:stop]
    # Memory tracks the two-vertex batch, not the three million source vertices.
    assert max(peaks) < 1_000_000


def test_partition_mesh_does_not_pay_for_a_full_float64_vertex_copy(tmp_path):
    """A whole partition over a huge float32 mesh stays far below a full float64 widening.

    The remaining peak comes from ``validate_mesh``'s own single boolean ``isfinite``
    pass over the source mesh, which is unchanged and out of this fix's scope.
    """
    count = 4_000_000
    vertices = np.zeros((count, 3), dtype=np.float32)
    vertices[[0, 1, 2]] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    full_float64_copy = count * 3 * 8  # about 96 MB for this source
    tracemalloc.start()
    try:
        manifest, totals = partition_mesh(vertices, faces, 10.0, tmp_path / 'mesh_chunks',
                                          reconstruction_mode='full')
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert manifest['total_chunks'] == 1 and totals['total_faces'] == 1
    assert peak < full_float64_copy / 4


def test_only_one_tile_geometry_is_materialized_at_a_time(monkeypatch, tmp_path):
    vertices, faces = spread(5)
    alive = []
    original = module.compact_tile
    def tracked(source_vertices, source_faces, references):
        assert not any(ref() is not None for ref in alive), 'the previous tile was still alive'
        result = original(source_vertices, source_faces, references)
        alive.extend([weakref.ref(result[0]), weakref.ref(result[1])])
        return result
    monkeypatch.setattr(module, 'compact_tile', tracked)
    manifest, _ = partition_mesh(vertices, faces, 10.0, tmp_path / 'mesh_chunks',
                                 reconstruction_mode='full')
    assert len(alive) == 2 * manifest['total_chunks']


def test_compact_tile_allocates_only_within_the_requested_triangles():
    vertices = np.zeros((1_000_000, 3), dtype=np.float32)
    vertices[[10, 11, 12]] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[10, 11, 12]], dtype=np.int64)
    tracemalloc.start()
    try:
        local_vertices, local_faces = compact_tile(vertices, faces, np.array([0], dtype=np.int64))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(local_vertices) == 3 and local_faces.tolist() == [[0, 1, 2]]
    # A per-vertex mask (1 MB bool) or an int64 remap (8 MB) would blow this budget.
    assert peak < 200_000


def test_open_file_handles_stay_bounded_while_spooling(monkeypatch, tmp_path):
    observed = []
    original = module._FaceSpool.append
    def counted(self, index, references):
        observed.append(len(self._handles))
        return original(self, index, references)
    monkeypatch.setattr(module, '_FaceSpool', type('Spool', (module._FaceSpool,), {'append': counted}))
    monkeypatch.setattr(module, 'SPOOL_HANDLE_LIMIT', 2)
    monkeypatch.setattr(module, 'FACE_BATCH', 4)
    vertices, faces = spread(6)
    manifest, totals = partition_mesh(vertices, faces, 10.0, tmp_path / 'mesh_chunks',
                                      reconstruction_mode='full')
    assert observed and max(observed) <= 2
    assert manifest['total_chunks'] == 6 and totals['total_faces'] == len(faces)


def test_scratch_files_are_removed_after_success(tmp_path):
    vertices, faces = spread(3)
    partition_mesh(vertices, faces, 10.0, tmp_path / 'mesh_chunks', reconstruction_mode='full')
    assert not list(tmp_path.glob('nksr-partition-*'))
    assert sorted(p.name for p in (tmp_path / 'mesh_chunks').iterdir()) == [
        'chunk_0000.ply', 'chunk_0001.ply', 'chunk_0002.ply', 'chunks.json']
