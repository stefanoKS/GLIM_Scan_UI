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
