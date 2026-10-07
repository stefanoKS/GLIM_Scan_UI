"""Color transfer preserves geometry exactly and only adds RGB."""
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from factory_mapping import color_transfer as ct


def _write_vertex_ply(path, points, intensity):
    vertex = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                          ('intensity', '<f4')])
    vertex['x'] = points[:, 0].astype(np.float32)
    vertex['y'] = points[:, 1].astype(np.float32)
    vertex['z'] = points[:, 2].astype(np.float32)
    vertex['intensity'] = intensity.astype(np.float32)
    PlyData([PlyElement.describe(vertex, 'vertex')], text=False, byte_order='<').write(str(path))


def _write_mesh_ply(path, points, faces):
    vertex = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    vertex['x'] = points[:, 0].astype(np.float32)
    vertex['y'] = points[:, 1].astype(np.float32)
    vertex['z'] = points[:, 2].astype(np.float32)
    face = np.empty(len(faces), dtype=[('vertex_indices', '<i4', (3,))])
    face['vertex_indices'] = faces
    PlyData([PlyElement.describe(vertex, 'vertex'), PlyElement.describe(face, 'face')],
            text=False, byte_order='<').write(str(path))


def test_glim_transfer_preserves_geometry_exactly(tmp_path):
    source_points = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    intensity = np.array([0.1, 0.2, 0.3], np.float32)
    source = tmp_path/'glim.ply'
    _write_vertex_ply(source, source_points, intensity)

    colored_points = np.array([[0.0, 0, 0], [1.0, 0, 0]], np.float64)
    cd = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.float32)
    confidence = np.array([1.0, 1.0], np.float32)
    output = tmp_path/'glim_colored.ply'
    stats = ct.transfer_colors_to_points(source, colored_points, cd, confidence, output, radius=0.025, k=1)

    mesh = PlyData.read(str(output))
    vertex = mesh['vertex'].data
    # Geometry and intensity preserved exactly.
    np.testing.assert_allclose(np.column_stack((vertex['x'], vertex['y'], vertex['z'])),
                               source_points.astype(np.float32))
    np.testing.assert_allclose(vertex['intensity'], intensity)
    # Colors: vertex 0 red, vertex 1 blue, vertex 2 fallback gray.
    np.testing.assert_array_equal(vertex['red'], [255, 0, 128])
    np.testing.assert_array_equal(vertex['blue'], [0, 255, 128])
    assert stats['vertices_total'] == 3
    assert stats['vertices_colored'] == 2
    assert stats['vertices_uncolored'] == 1


def test_mesh_transfer_preserves_positions_and_topology(tmp_path):
    points = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]])
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    source = tmp_path/'mesh.ply'
    _write_mesh_ply(source, points, faces)

    colored_points = np.array([[0.0, 0, 0], [1.0, 0, 0]], np.float64)
    cd = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], np.float32)
    confidence = np.array([1.0, 0.5], np.float32)
    output = tmp_path/'nksr_colored.ply'
    stats = ct.transfer_colors_to_mesh(source, colored_points, cd, confidence, output, radius=0.025, k=1)

    mesh = PlyData.read(str(output))
    vertex = mesh['vertex'].data
    np.testing.assert_allclose(np.column_stack((vertex['x'], vertex['y'], vertex['z'])),
                               points.astype(np.float32))
    faces_out = np.asarray(mesh['face']['vertex_indices'])
    if faces_out.dtype.kind == 'O':
        faces_out = np.stack(faces_out)
    np.testing.assert_array_equal(faces_out, faces)
    np.testing.assert_array_equal(vertex['red'][:2], [0, 255])
    assert stats['vertices_total'] == 4
    assert stats['vertices_colored'] == 2


def test_transfer_rejects_uncolored_source():
    with pytest.raises(ValueError, match='No colored master points'):
        ct._transfer(np.array([[0.0, 0, 0]]), np.array([[0.0, 0, 0]]),
                     np.full((1, 3), np.nan, np.float32), np.array([0.0], np.float32),
                     0.025, 5, print, (128, 128, 128))


def test_transfer_validates_inputs():
    with pytest.raises(ValueError, match='radius'):
        ct._transfer(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3), np.float32),
                     np.ones(1, np.float32), 0.0, 5, print, (128, 128, 128))
    with pytest.raises(ValueError, match='positive integer'):
        ct._transfer(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3), np.float32),
                     np.ones(1, np.float32), 0.025, 0, print, (128, 128, 128))


def test_transfer_requires_scipy_with_a_clear_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'scipy', None)
    monkeypatch.setitem(sys.modules, 'scipy.spatial', None)
    with pytest.raises(ImportError, match='SciPy is required for color transfer'):
        ct._transfer(np.array([[0.0, 0, 0]]), np.array([[0.0, 0, 0]]),
                     np.array([[1.0, 0, 0]], np.float32), np.array([1.0], np.float32),
                     0.025, 1, print, (128, 128, 128))


def test_transfer_rejects_geometry_from_another_run(tmp_path):
    source_points = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]], np.float32)
    source = tmp_path/'glim.ply'
    _write_vertex_ply(source, source_points, np.zeros(3, np.float32))
    output = tmp_path/'colored.ply'
    cd = np.array([[1.0, 0.0, 0.0]], np.float32)
    confidence = np.array([1.0], np.float32)
    unrelated = np.array([[500.0, 500.0, 500.0]], np.float64)
    with pytest.raises(ValueError, match='does not overlap the colored master cloud'):
        ct.transfer_colors_to_points(source, unrelated, cd, confidence, output)
    assert not output.exists()


def test_transfer_rejects_a_target_far_from_any_colored_point(tmp_path):
    middle = np.full((4, 3), 5.0, np.float32)
    source = tmp_path/'glim.ply'
    _write_vertex_ply(source, middle, np.zeros(4, np.float32))
    output = tmp_path/'colored.ply'
    master = np.array([[0.0, 0, 0], [10.0, 10, 10]], np.float64)
    cd = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    confidence = np.ones(2, np.float32)
    with pytest.raises(ValueError, match='transfer radius'):
        ct.transfer_colors_to_points(source, master, cd, confidence, output)
    assert not output.exists()
    # The guard is advisory only when explicitly disabled.
    stats = ct.transfer_colors_to_points(source, master, cd, confidence, output, sanity=False)
    assert stats['sanity_checked'] is False and stats['vertices_colored'] == 0


def test_transfer_sanity_statistics_are_reported(tmp_path):
    source_points = np.array([[0.0, 0, 0], [1.0, 0, 0]], np.float32)
    source = tmp_path/'glim.ply'
    _write_vertex_ply(source, source_points, np.zeros(2, np.float32))
    cd = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    confidence = np.ones(2, np.float32)
    stats = ct.transfer_colors_to_points(source, source_points.astype(np.float64), cd, confidence,
                                         tmp_path/'colored.ply', radius=0.025, k=2)
    for key in ('vertices_total', 'vertices_colored', 'vertices_uncolored', 'coverage_percent',
                'mean_nearest_color_distance', 'max_transfer_radius',
                'sanity_bounding_box_overlap', 'sanity_median_nearest_distance',
                'sanity_sample_coverage'):
        assert key in stats, key
    assert stats['coverage_percent'] == stats['percentage_colored'] == 100.0
    assert stats['max_transfer_radius'] == 0.025
    assert stats['mean_nearest_color_distance'] == 0.0
    assert stats['sanity_sample_coverage'] == 1.0


def test_transfer_preserves_xyz_bit_for_bit(tmp_path):
    rng = np.random.default_rng(5)
    source_points = rng.random((200, 3)).astype(np.float32)*10
    intensity = rng.random(200).astype(np.float32)
    source = tmp_path/'glim.ply'
    _write_vertex_ply(source, source_points, intensity)
    cd = rng.random((50, 3)).astype(np.float32)
    confidence = np.ones(50, np.float32)
    stats = ct.transfer_colors_to_points(source, source_points[:50].astype(np.float64), cd,
                                         confidence, tmp_path/'colored.ply', radius=5.0, k=3)
    vertex = PlyData.read(str(tmp_path/'colored.ply'))['vertex'].data
    np.testing.assert_array_equal(np.column_stack((vertex['x'], vertex['y'], vertex['z'])),
                                  source_points)
    np.testing.assert_array_equal(vertex['intensity'], intensity)
    assert stats['source_vertices_preserved'] is True


def test_mesh_transfer_preserves_vertices_and_faces_bit_for_bit(tmp_path):
    rng = np.random.default_rng(9)
    points = rng.random((60, 3)).astype(np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.int32)
    source = tmp_path/'mesh.ply'
    _write_mesh_ply(source, points, faces)
    cd = rng.random((20, 3)).astype(np.float32)
    ct.transfer_colors_to_mesh(source, points[:20].astype(np.float64), cd, np.ones(20, np.float32),
                               tmp_path/'nksr_colored.ply', radius=5.0, k=2)
    mesh = PlyData.read(str(tmp_path/'nksr_colored.ply'))
    vertex = mesh['vertex'].data
    np.testing.assert_array_equal(np.column_stack((vertex['x'], vertex['y'], vertex['z'])), points)
    faces_out = np.asarray(mesh['face']['vertex_indices'])
    if faces_out.dtype.kind == 'O':
        faces_out = np.stack(faces_out)
    np.testing.assert_array_equal(faces_out, faces)


def test_transfer_never_overwrites_the_source_ply(tmp_path):
    points = np.array([[0.0, 0, 0]], np.float32)
    source = tmp_path/'glim.ply'
    _write_vertex_ply(source, points, np.zeros(1, np.float32))
    before = source.read_bytes()
    cd = np.array([[1.0, 0.0, 0.0]], np.float32)
    with pytest.raises(ValueError, match='Refusing to overwrite the source PLY'):
        ct.transfer_colors_to_points(source, points.astype(np.float64), cd, np.ones(1, np.float32),
                                     source)
    with pytest.raises(ValueError, match='Refusing to overwrite the source PLY'):
        ct.transfer_colors_to_mesh(source, points.astype(np.float64), cd, np.ones(1, np.float32),
                                   source)
    assert source.read_bytes() == before
