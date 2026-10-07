"""Focused tests for per-chunk NKSR mesh export (no GPU or torch required)."""
import contextlib
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from factory_mapping.nksr_worker import (parser, execute, extract_and_save_chunks,
    write_full_single_chunk, chunk_core_bounds, crop_mesh_to_core, chunk_grid_from_centers,
    WorkerError, MESH_OUTPUT_MODES)
from factory_mapping.nksr_mesh import write_mesh, inspect_mesh
from factory_mapping.api import MeshRequest
from factory_mapping.storage import read_json


def test_default_output_mode_is_merged():
    assert MeshRequest().mesh_output_mode == 'merged'
    settings = parser().parse_args([])
    assert settings.mesh_output_mode == 'merged'


def test_parser_rejects_invalid_output_mode():
    with pytest.raises(SystemExit):
        parser().parse_args(['--mesh-output-mode', 'poisson'])
    with pytest.raises(SystemExit):
        parser().parse_args(['--mesh-output-mode', 'CHUNKS'])
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        MeshRequest(mesh_output_mode='poisson')


def test_chunk_grid_from_centers_stride_and_index():
    stride = 15.0
    centers = np.array([[0, 0, 0], [stride, 0, 0], [0, stride, 0], [stride, stride, 0]], dtype=np.float64)
    strides, grid_index = chunk_grid_from_centers(centers)
    assert strides[0] == pytest.approx(stride)
    assert strides[1] == pytest.approx(stride)
    assert not np.isfinite(strides[2])
    assert grid_index.tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]


def test_chunk_core_bounds_half_open_midpoints():
    stride = 15.0
    centers = np.array([[0.0, 0, 0], [stride, 0, 0]], dtype=np.float64)
    bounds = chunk_core_bounds(centers)
    lo0, hi0 = bounds[0]
    lo1, hi1 = bounds[1]
    # Left chunk owns (-inf, 7.5), right chunk owns [7.5, +inf).
    assert lo0[0] == -np.inf and hi0[0] == pytest.approx(stride / 2)
    assert lo1[0] == pytest.approx(stride / 2) and hi1[0] == np.inf
    for axis in (1, 2):
        assert lo0[axis] == -np.inf and hi0[axis] == np.inf
        assert lo1[axis] == -np.inf and hi1[axis] == np.inf


def test_crop_mesh_to_core_centroid_ownership_and_compaction():
    vertices = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [8, 0, 0], [9, 0, 0], [8, 2, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    kept_v, kept_f = crop_mesh_to_core(vertices, faces, np.array([-np.inf, -np.inf, -np.inf]),
                                       np.array([5.0, np.inf, np.inf]))
    assert kept_f.tolist() == [[0, 1, 2]]
    np.testing.assert_array_equal(kept_v, vertices[:3])


def test_crop_mesh_to_core_empty_result():
    vertices = np.array([[8, 0, 0], [9, 0, 0], [8, 2, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    kept_v, kept_f = crop_mesh_to_core(vertices, faces, np.array([-np.inf, -np.inf, -np.inf]),
                                       np.array([5.0, np.inf, np.inf]))
    assert kept_v.shape == (0, 3) and kept_f.shape == (0, 3)


class FakeField:
    def __init__(self, v, f):
        self.v = np.asarray(v, dtype=np.float32)
        self.f = np.asarray(f, dtype=np.int64)
        self.devices = []
        self.extracted = 0

    def to_(self, device):
        self.devices.append(device)
        return self

    def extract_dual_mesh(self, **kwargs):
        self.extracted += 1
        return SimpleNamespace(v=self.v, f=self.f)


class FakeTransform:
    def __init__(self, center):
        self.t = np.asarray(center, dtype=np.float64)
        self.q = SimpleNamespace(rotation_matrix=np.eye(3))


def test_three_chunk_world_coordinates_scale_and_manifest(tmp_path):
    scale = 5.0
    stride = 15.0  # scaled chunk_sub_size for chunk_size=20, overlap_ratio=0.25
    settings = SimpleNamespace(mise_iter=1, chunk_size=20.0, overlap_ratio=0.25,
                               mesh_output_mode='chunks')
    # Chunk 0 centered at origin: triangles in local = global scaled coordinates.
    field0 = FakeField(
        v=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],   # A: centroid x ~ .33 -> chunk 0
                    [7, 0, 0], [8, 0, 0], [7, 1, 0],   # B: centroid x ~ 7.33 -> chunk 0
                    [8, 0, 0], [9, 0, 0], [8, 1, 0]],  # C: centroid x ~ 8.33 -> chunk 1
                   dtype=np.float32),
        f=np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64))
    # Chunk 1 centered at x=stride: same world triangles expressed in local coords.
    field1 = FakeField(
        v=np.array([[-8, 0, 0], [-7, 0, 0], [-8, 1, 0],  # B world -> centroid 7.33 -> chunk 0
                    [-7, 0, 0], [-6, 0, 0], [-7, 1, 0]],  # C world -> centroid 8.33 -> chunk 1
                   dtype=np.float32),
        f=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64))
    centers = np.array([[0.0, 0, 0], [stride, 0, 0]], dtype=np.float64)
    rotations = [np.eye(3), np.eye(3)]
    device = SimpleNamespace(type='cpu')
    manifest, totals = extract_and_save_chunks(
        [field0, field1], centers, rotations, settings, scale, tmp_path / 'mesh_chunks', lambda *a, **k: None, None, device)

    # Deterministic filenames in field order.
    assert [c['file'] for c in manifest['chunks']] == ['chunk_0000.ply', 'chunk_0001.ply']
    # Triangle C belongs only to chunk 1; B belongs only to chunk 0; no duplicates.
    assert manifest['chunks'][0]['faces'] == 2
    assert manifest['chunks'][1]['faces'] == 1
    assert totals['total_faces'] == 3
    assert manifest['total_vertices'] == manifest['chunks'][0]['vertices'] + manifest['chunks'][1]['vertices']

    chunk0 = inspect_mesh(tmp_path / 'mesh_chunks' / 'chunk_0000.ply')
    chunk1 = inspect_mesh(tmp_path / 'mesh_chunks' / 'chunk_0001.ply')
    # World coordinates: chunk 0 local == global scaled, then / scale.
    np.testing.assert_allclose(chunk0['bounding_box_min'], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(chunk0['bounding_box_max'], [8.0 / scale, 1.0 / scale, 0.0], atol=1e-5)
    # Chunk 1 local -> global (add stride) -> / scale: kept triangle C has world x in [8/5, 9/5].
    np.testing.assert_allclose(chunk1['bounding_box_min'], [8.0 / scale, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(chunk1['bounding_box_max'], [9.0 / scale, 1.0 / scale, 0.0], atol=1e-5)

    # Concatenated chunk bounds cover the same world extent as the whole scene.
    union_min = np.minimum(np.array(chunk0['bounding_box_min']), np.array(chunk1['bounding_box_min']))
    union_max = np.maximum(np.array(chunk0['bounding_box_max']), np.array(chunk1['bounding_box_max']))
    np.testing.assert_allclose(union_min, [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(union_max, [9.0 / scale, 1.0 / scale, 0.0], atol=1e-5)

    saved = read_json(tmp_path / 'mesh_chunks' / 'chunks.json')
    assert saved['coordinate_system'] == 'GLIM_world' and saved['units'] == 'meters'
    assert saved['coordinate_scale'] == scale and saved['output_mode'] == 'chunks'
    assert saved['total_chunks'] == 2 and saved['total_faces'] == 3
    assert saved['chunks'][0]['grid_index'] == [0, 0, 0]
    assert saved['chunks'][1]['grid_index'] == [1, 0, 0]
    assert saved['chunks'][0]['core_bbox_max'][0] == pytest.approx(stride / 2)


def test_empty_chunk_after_cropping_is_safe(tmp_path):
    settings = SimpleNamespace(mise_iter=1, chunk_size=20.0, overlap_ratio=0.25, mesh_output_mode='chunks')
    field0 = FakeField(v=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                       f=np.array([[0, 1, 2]], dtype=np.int64))
    field1 = FakeField(v=np.array([[-14, 0, 0], [-13, 0, 0], [-14, 1, 0]], dtype=np.float32),
                       f=np.array([[0, 1, 2]], dtype=np.int64))
    # Chunk 1's triangle lies entirely in chunk 0's side of the midpoint (global x < 7.5).
    centers = np.array([[0.0, 0, 0], [15.0, 0, 0]], dtype=np.float64)
    rotations = [np.eye(3), np.eye(3)]
    device = SimpleNamespace(type='cpu')
    manifest, totals = extract_and_save_chunks(
        [field0, field1], centers, rotations, settings, 5.0, tmp_path / 'mesh_chunks', lambda *a, **k: None, None, device)
    assert manifest['chunks'][0]['file'] == 'chunk_0000.ply'
    assert manifest['chunks'][1]['file'] is None and manifest['chunks'][1]['vertices'] == 0
    assert totals['total_faces'] == 1
    assert not (tmp_path / 'mesh_chunks' / 'chunk_0001.ply').exists()


def test_single_chunk_has_full_extent_core(tmp_path):
    settings = SimpleNamespace(mise_iter=1, chunk_size=20.0, overlap_ratio=0.25, mesh_output_mode='chunks')
    field = FakeField(v=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                      f=np.array([[0, 1, 2]], dtype=np.int64))
    centers = np.array([[3.0, 4.0, 5.0]], dtype=np.float64)
    rotations = [np.eye(3)]
    device = SimpleNamespace(type='cpu')
    manifest, totals = extract_and_save_chunks(
        [field], centers, rotations, settings, 5.0, tmp_path / 'mesh_chunks', lambda *a, **k: None, None, device)
    lo, hi = chunk_core_bounds(centers)[0]
    assert np.all(lo == -np.inf) and np.all(hi == np.inf)
    assert manifest['total_chunks'] == 1 and manifest['chunks'][0]['faces'] == 1
    # Single chunk world coordinates = (local + center) / scale.
    assert inspect_mesh(tmp_path / 'mesh_chunks' / 'chunk_0000.ply')['bounding_box_min'] == pytest.approx([0.6, 0.8, 1.0])


def test_write_full_single_chunk(tmp_path):
    settings = SimpleNamespace(mesh_output_mode='both')
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    manifest, totals = write_full_single_chunk(settings, vertices, faces, tmp_path / 'mesh_chunks')
    assert manifest['total_chunks'] == 1 and totals['total_faces'] == 1
    assert inspect_mesh(tmp_path / 'mesh_chunks/chunk_0000.ply')['face_count'] == 1
    assert manifest['chunks'][0]['note'].startswith('Full')
    assert manifest['coordinate_scale'] == 1.0 and manifest['overlap_ratio'] is None


def test_api_rejects_invalid_mesh_output_mode(root, monkeypatch):
    from fastapi.testclient import TestClient
    from factory_mapping.api import make_app
    from factory_mapping import nksr_jobs as jobs
    calls = []
    async def run(service, sid, rid, settings): calls.append(settings); return {'state': 'RUNNING'}
    monkeypatch.setattr(jobs, 'reconstruct', run)
    with TestClient(make_app(root, True)) as client:
        url = '/api/sessions/session/reconstruction/run_0123456789ab/mesh'
        assert client.post(url, json={'mesh_output_mode': 'chunks'}).status_code == 202
        assert calls[-1]['mesh_output_mode'] == 'chunks'
        assert client.post(url, json={'mesh_output_mode': 'poisson'}).status_code == 422


def test_validate_completed_accepts_chunk_outputs(tmp_path):
    from factory_mapping import nksr_jobs as jobs
    from factory_mapping.storage import atomic_json
    settings = SimpleNamespace(mesh_output_mode='chunks')
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    manifest, totals = write_full_single_chunk(settings, vertices, faces, tmp_path / 'mesh_chunks')
    atomic_json(tmp_path / 'nksr_metadata.json', dict(mesh_output_mode='chunks', actual_mode='full',
                                                      vertex_count=totals['total_vertices'],
                                                      face_count=totals['total_faces'],
                                                      chunk_count=manifest['total_chunks']))
    result = jobs.validate_completed(tmp_path, 0)
    assert result['chunk_count'] == 1 and result['vertex_count'] == totals['total_vertices']


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def float(self):
        return self

    def to(self, device):
        return self

    def __mul__(self, scale):
        return FakeTensor(self.value * scale)


def execute_runtime(field, fused_mesh=None):
    if fused_mesh is None:
        fused_mesh = SimpleNamespace(v=np.zeros((3, 3)), f=np.array([[0, 1, 2]]))
    cuda = SimpleNamespace(mem_get_info=lambda device: (8 * 1024**3, 8 * 1024**3),
                           empty_cache=lambda: None, max_memory_allocated=lambda device: 0,
                           reset_peak_memory_stats=lambda device: None)
    torch = SimpleNamespace(device=lambda name: SimpleNamespace(type=name.split(':')[0]),
                            from_numpy=FakeTensor, cuda=cuda,
                            inference_mode=contextlib.nullcontext)
    nksr = SimpleNamespace(get_estimate_normal_preprocess_fn=lambda *a: (lambda xyz, n, s: (xyz, n, s)))
    reconstructor = SimpleNamespace(reconstruct=lambda xyz, **kwargs: field,
                                    network=SimpleNamespace(to=lambda device: None))
    device = SimpleNamespace(type='cpu')
    return torch, nksr, reconstructor, device


def execute_settings(tmp_path, mode='chunked', output_mode='merged'):
    return parser().parse_args(['--mode', mode, '--chunk-size', '4', '--mesh-output-mode', output_mode,
                                '--output', str(tmp_path / 'output' / 'mesh.ply')])


def test_chunks_mode_does_not_call_fused_extraction(tmp_path):
    chunk0 = FakeField(v=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                       f=np.array([[0, 1, 2]], dtype=np.int64))
    fused_calls = []

    class Fused:
        fields = [chunk0]
        transforms = [FakeTransform([0.0, 0.0, 0.0])]

        def to_(self, device):
            pass

        def extract_dual_mesh(self, **kwargs):
            fused_calls.append(kwargs)
            return SimpleNamespace(v=np.zeros((3, 3)), f=np.array([[0, 1, 2]]))

    torch, nksr, reconstructor, device = execute_runtime(Fused())
    settings = execute_settings(tmp_path, output_mode='chunks')
    vertices, faces, result = execute(np.ones((70, 3)), np.ones((70, 3)) * 5, settings,
                                      lambda *a, **k: None, torch, nksr, device, reconstructor)
    assert vertices is None and faces is None
    assert fused_calls == []
    assert result['mesh_output_mode'] == 'chunks' and result['chunk_count'] == 1
    assert (tmp_path / 'output' / 'mesh_chunks' / 'chunk_0000.ply').is_file()
    assert not (tmp_path / 'output' / 'mesh.ply').exists()


def test_both_mode_produces_chunks_and_fused(tmp_path):
    chunk0 = FakeField(v=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                       f=np.array([[0, 1, 2]], dtype=np.int64))
    fused_calls = []

    class Fused:
        fields = [chunk0]
        transforms = [FakeTransform([0.0, 0.0, 0.0])]

        def to_(self, device):
            pass

        def extract_dual_mesh(self, **kwargs):
            fused_calls.append(kwargs)
            return SimpleNamespace(v=np.zeros((3, 3)), f=np.array([[0, 1, 2]]))

    torch, nksr, reconstructor, device = execute_runtime(Fused())
    settings = execute_settings(tmp_path, output_mode='both')
    vertices, faces, result = execute(np.ones((70, 3)), np.ones((70, 3)) * 5, settings,
                                      lambda *a, **k: None, torch, nksr, device, reconstructor)
    assert len(fused_calls) == 1
    assert vertices is not None and faces is not None
    assert result['chunk_count'] == 1
    assert (tmp_path / 'output' / 'mesh_chunks' / 'chunk_0000.ply').is_file()


def test_merged_mode_preserves_existing_single_extraction(tmp_path):
    fused_calls = []

    class Fused:
        def to_(self, device):
            pass

        def extract_dual_mesh(self, **kwargs):
            fused_calls.append(kwargs)
            return SimpleNamespace(v=np.zeros((3, 3)), f=np.array([[0, 1, 2]]))

    torch, nksr, reconstructor, device = execute_runtime(Fused())
    settings = execute_settings(tmp_path, output_mode='merged')
    vertices, faces, result = execute(np.ones((70, 3)), np.ones((70, 3)) * 5, settings,
                                      lambda *a, **k: None, torch, nksr, device, reconstructor)
    assert len(fused_calls) == 1
    assert vertices is not None and result.get('chunk_count') is None
    assert not (tmp_path / 'output' / 'mesh_chunks').exists()


def test_chunks_mode_requires_chunk_fields(tmp_path):
    class Fused:
        def to_(self, device):
            pass

        def extract_dual_mesh(self, **kwargs):
            return SimpleNamespace(v=np.zeros((3, 3)), f=np.array([[0, 1, 2]]))

    torch, nksr, reconstructor, device = execute_runtime(Fused())
    settings = execute_settings(tmp_path, output_mode='chunks')
    with pytest.raises(WorkerError, match='chunk fields'):
        execute(np.ones((70, 3)), np.ones((70, 3)) * 5, settings,
                lambda *a, **k: None, torch, nksr, device, reconstructor)
