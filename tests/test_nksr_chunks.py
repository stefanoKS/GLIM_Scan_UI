"""Focused tests for per-chunk NKSR mesh export (no GPU or torch required)."""
import contextlib
from pathlib import Path
from types import SimpleNamespace
import asyncio
import tempfile
import numpy as np
import pytest
from factory_mapping import nksr_jobs as jobs
from factory_mapping.nksr_worker import (parser, execute, extract_and_save_chunks,
    write_full_single_chunk, chunk_core_bounds, chunk_grid_layout, crop_mesh_to_owned,
    owned_triangle_mask, WorkerError)
from factory_mapping.nksr_mesh import write_mesh, inspect_mesh
from factory_mapping.api import MeshRequest
from factory_mapping.storage import read_json, atomic_json


SCALE = 5.0  # NKSR native voxel 0.1 m over target voxel 0.02 m


@pytest.fixture
def prepared(root, monkeypatch):
    """Minimal PREPARED reconstruction run, mirroring tests/test_nksr.py."""
    from factory_mapping.service import Service
    service = Service(root, True)
    item = service.sessions.create('NKSR', '', service.config)
    session = service.sessions.get(item['id'])
    run = session / 'reconstruction/run_0123456789ab'
    (run / 'input').mkdir(parents=True)
    np.savez(run / 'input/nksr_input.npz', points=np.zeros((3, 3)), sensor_origins=np.ones((3, 3)))
    atomic_json(run / 'job.json', {'state': 'completed', 'voxel_size_m': .01})
    python = root / 'NKSR env with spaces/bin/python'
    python.parent.mkdir(parents=True); python.touch()
    monkeypatch.setenv('NKSR_PYTHON', str(python)); service.mock = False
    return service, item['id'], run, python


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


# --------------------------------------------------------------------------- #
# Grid layout: nominal stride and preserved gaps
# --------------------------------------------------------------------------- #

def test_grid_layout_uses_nominal_stride_and_preserves_gaps():
    # chunk_size 4 m * scale 5 * (1 - .05) = 19 m scaled stride.
    centers = np.array([[0.0, 0, 0], [38.0, 0, 0]])
    stride, relative, source = chunk_grid_layout(centers, 4.0 * SCALE, .05)
    assert stride == pytest.approx(19.0) and source == 'nksr_chunk_size'
    assert relative.tolist() == [[0, 0, 0], [2, 0, 0]]  # gap preserved, not rank-compressed


def test_grid_layout_preserves_gaps_in_3d():
    centers = np.array([[0.0, 0, 0], [19.0, 0, 0], [76.0, 19.0, 19.0]])
    stride, relative, source = chunk_grid_layout(centers, 20.0, .05)
    assert stride == pytest.approx(19.0) and source == 'nksr_chunk_size'
    assert relative.tolist() == [[0, 0, 0], [1, 0, 0], [4, 1, 1]]


def test_grid_layout_single_center_and_mismatched_stride():
    stride, relative, source = chunk_grid_layout(np.array([[3.0, 4.0, 5.0]]), 20.0, .05)
    assert stride == pytest.approx(19.0) and relative.tolist() == [[0, 0, 0]]
    # A nominal stride that disagrees with the surviving centers falls back explicitly.
    stride, relative, source = chunk_grid_layout(np.array([[0.0, 0, 0], [15.0, 7.0, 0]]), 20.0, .05)
    assert source == 'derived_from_centers' and stride == pytest.approx(7.0)


def test_chunk_grid_layout_rejects_empty_fields():
    with pytest.raises(WorkerError):
        chunk_grid_layout(np.zeros((0, 3)), 20.0, .05)


# --------------------------------------------------------------------------- #
# Ownership: sparse grids, holes, diagonal overlap
# --------------------------------------------------------------------------- #

def test_sparse_diagonal_grid_does_not_crop_unrelated_geometry():
    """Active centers [0,0,0] and [15,15,0]: a per-axis rank boundary must not crop."""
    centers = np.array([[0.0, 0, 0], [15.0, 15.0, 0]])
    centroid = np.array([[9.0, 0.0, 0.0]])
    # 9 > midpoint(0, 15) = 7.5, so the previous rank-based rule cropped this triangle.
    assert owned_triangle_mask(centroid, centers, 15.0, 0)[0]
    # Owned geometry moves to the other chunk only when that chunk's cube really contains it.
    claimed = np.array([[9.0, 9.0, 4.0]])  # inside [15,15,0]'s cube, outside [0,0,0]'s cube
    assert not owned_triangle_mask(claimed, centers, 15.0, 0)[0]
    assert owned_triangle_mask(claimed, centers, 15.0, 1)[0]


def test_sparse_grid_hole_geometry_is_never_silently_cropped():
    """A 2D grid with a missing cell: unclaimed geometry stays with its emitter."""
    stride = 19.0
    relative = [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)]  # hole at (1,1)
    centers = np.array([[rx * stride, ry * stride, 0.0] for rx, ry in relative])
    hole = np.array([[stride, stride, 0.0]])  # inside the missing cell, inside no active cube
    assert (1, 1) not in relative
    # No cube claims the hole, so whichever chunk emits geometry there keeps it.
    assert all(owned_triangle_mask(hole, centers, 20.0, index)[0] for index in range(len(centers)))
    stride, relative_index, source = chunk_grid_layout(centers, 19.0, .0)
    assert source == 'nksr_chunk_size' and stride == pytest.approx(19.0)
    assert relative_index.tolist() == [[rx, ry, 0] for rx, ry in relative]
    # Cell (0,0) is bounded only by its genuine neighbours at 19, i.e. at the midpoint 9.5.
    bounds = chunk_core_bounds(centers, relative_index, stride)
    assert bounds[0][1][0] == pytest.approx(9.5) and bounds[0][1][1] == pytest.approx(9.5)
    assert bounds[0][0][0] == -np.inf and bounds[0][0][1] == -np.inf


def test_sparse_grid_diagonal_overlap_owner_is_deterministic():
    """Two active cells that only share a corner still split their overlap exactly once."""
    centers = np.array([[0.0, 0, 0], [19.0, 19.0, 19.0]])
    near_first = np.array([[9.2, 9.2, 9.2]])
    near_second = np.array([[9.8, 9.8, 9.8]])
    assert owned_triangle_mask(near_first, centers, 20.0, 0)[0]
    assert not owned_triangle_mask(near_first, centers, 20.0, 1)[0]
    assert owned_triangle_mask(near_second, centers, 20.0, 1)[0]
    assert not owned_triangle_mask(near_second, centers, 20.0, 0)[0]


def test_sparse_3d_grid_with_holes_and_unrelated_chunk():
    centers = np.array([[0.0, 0, 0], [19.0, 0, 0], [0.0, 19.0, 0], [0.0, 0, 19], [19.0, 19.0, 19.0]])
    inside_first = np.array([[4.0, 4.0, 4.0]])
    assert owned_triangle_mask(inside_first, centers, 20.0, 0)[0]
    assert not owned_triangle_mask(inside_first, centers, 20.0, 1)[0]
    # The (1,1,0) cell is missing: no cube claims its centre, so the emitter keeps it.
    hole = np.array([[19.0, 19.0, 0.0]])
    assert owned_triangle_mask(hole, centers, 20.0, 0)[0]
    # Exact midpoint between two adjacent centres is a distance tie -> lowest field index.
    tie = np.array([[9.5, 0.0, 0.0]])
    assert owned_triangle_mask(tie, centers, 20.0, 0)[0]
    assert not owned_triangle_mask(tie, centers, 20.0, 1)[0]
    # Only the diagonal chunk's cube contains this point.
    diagonal_only = np.array([[19.0, 19.0, 24.0]])
    assert owned_triangle_mask(diagonal_only, centers, 20.0, 4)[0]
    assert not owned_triangle_mask(diagonal_only, centers, 20.0, 0)[0]


def test_adjacent_overlap_exports_no_duplicate_triangles():
    """Both chunks emit the same overlap triangles; only the owner keeps them."""
    centers = np.array([[0.0, 0, 0], [19.0, 0, 0]])
    for centroid, owner in ((np.array([[9.2, 0.0, 0.0]]), 0), (np.array([[9.8, 0.0, 0.0]]), 1)):
        assert owned_triangle_mask(centroid, centers, 20.0, owner)[0]
        assert not owned_triangle_mask(centroid, centers, 20.0, 1 - owner)[0]


def test_core_bounds_only_split_against_genuinely_adjacent_cells():
    """A distant diagonal chunk must not bound the first chunk's whole core cell."""
    centers = np.array([[0.0, 0, 0], [15.0, 15.0, 0]])
    stride, relative, _ = chunk_grid_layout(centers, 15.0, .0)
    lo, hi = chunk_core_bounds(centers, relative, stride)[0]
    # [15,15,0] differs by one on x and y, so it bounds those axes...
    assert hi[0] == pytest.approx(7.5) and hi[1] == pytest.approx(7.5)
    # ...but never the z axis, where the two chunks are unrelated.
    assert lo[2] == -np.inf and hi[2] == np.inf
    assert lo[0] == -np.inf and lo[1] == -np.inf


def test_core_bounds_ignore_cells_with_index_gaps():
    centers = np.array([[0.0, 0, 0], [76.0, 0, 0]])
    stride, relative, _ = chunk_grid_layout(centers, 20.0, .05)
    assert relative.tolist() == [[0, 0, 0], [4, 0, 0]]
    bounds = chunk_core_bounds(centers, relative, stride)
    assert bounds[0][1][0] == np.inf and bounds[1][0][0] == -np.inf


# --------------------------------------------------------------------------- #
# Extraction: coordinates, scale, manifest
# --------------------------------------------------------------------------- #

class FakeField:
    def __init__(self, v, f):
        self.v = np.asarray(v, dtype=np.float32).reshape(-1, 3)
        self.f = np.asarray(f, dtype=np.int64).reshape(-1, 3)
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


def local_triangle(centroid_scaled, center, size=0.2):
    """Local coordinates of a triangle whose global scaled centroid is exactly as given."""
    offsets = np.array([[0, 0, 0], [size, 0, 0], [-size, 0, 0]], dtype=np.float64)
    return ((np.asarray(centroid_scaled, dtype=np.float64) + offsets)
            - np.asarray(center, dtype=np.float64)).astype(np.float32)


def run_extract(tmp_path, centers, centroids_per_field, chunk_size_m, scale=SCALE,
                overlap_ratio=.0, output_mode='chunks'):
    """Extract chunks for one fake field per center, each emitting its given centroids."""
    fields = []
    for center, centroids in zip(centers, centroids_per_field):
        vertices, faces = [], []
        for centroid in centroids:
            base = len(vertices)
            vertices.extend(local_triangle(centroid, center).tolist())
            faces.append([base, base + 1, base + 2])
        fields.append(FakeField(np.asarray(vertices, dtype=np.float32).reshape(-1, 3),
                                np.asarray(faces, dtype=np.int64).reshape(-1, 3)))
    settings = SimpleNamespace(mise_iter=1, chunk_size=chunk_size_m, overlap_ratio=overlap_ratio,
                               mesh_output_mode=output_mode)
    return extract_and_save_chunks(fields, np.asarray(centers, dtype=np.float64),
                                   [np.eye(3)] * len(centers), settings, scale, chunk_size_m,
                                   Path(tmp_path) / 'mesh_chunks',
                                   lambda *args, **kwargs: None, None, SimpleNamespace(type='cpu'))


def test_chunk_world_coordinates_scale_and_manifest(tmp_path):
    # chunk_size 4 m, scale 5, overlap .25 -> 15 scaled stride between the two centres.
    centers = [[0.0, 0, 0], [15.0, 0, 0]]
    centroids = [
        [[1.0, 0, 0], [7.0, 0, 0], [9.0, 0, 0]],   # overlap triangles are emitted by both
        [[7.0, 0, 0], [9.0, 0, 0], [20.0, 0, 0]],
    ]
    manifest, totals = run_extract(tmp_path, centers, centroids, chunk_size_m=4.0, overlap_ratio=.25)
    assert [chunk['file'] for chunk in manifest['chunks']] == ['chunk_0000.ply', 'chunk_0001.ply']
    # 1 and 7 stay with the first chunk; 9 and 20 go to the second: four unique triangles.
    assert manifest['chunks'][0]['faces'] == 2
    assert manifest['chunks'][1]['faces'] == 2
    assert totals['total_faces'] == 4
    assert manifest['total_vertices'] == sum(chunk['vertices'] for chunk in manifest['chunks'])

    chunk0 = inspect_mesh(tmp_path / 'mesh_chunks' / 'chunk_0000.ply')
    chunk1 = inspect_mesh(tmp_path / 'mesh_chunks' / 'chunk_0001.ply')
    # Scaled centroids -> world meters: 1/5, 7/5, 9/5 and 20/5, grown by the triangle offset.
    assert chunk0['bounding_box_min'][0] == pytest.approx((1.0 - .2) / SCALE, abs=1e-6)
    assert chunk0['bounding_box_max'][0] == pytest.approx((7.0 + .2) / SCALE, abs=1e-6)
    assert chunk1['bounding_box_min'][0] == pytest.approx((9.0 - .2) / SCALE, abs=1e-6)
    assert chunk1['bounding_box_max'][0] == pytest.approx((20.0 + .2) / SCALE, abs=1e-6)

    union_min = np.minimum(chunk0['bounding_box_min'], chunk1['bounding_box_min'])
    union_max = np.maximum(chunk0['bounding_box_max'], chunk1['bounding_box_max'])
    assert union_min[0] == pytest.approx((1.0 - .2) / SCALE, abs=1e-6)
    assert union_max[0] == pytest.approx((20.0 + .2) / SCALE, abs=1e-6)

    saved = read_json(tmp_path / 'mesh_chunks' / 'chunks.json')
    assert saved['coordinate_system'] == 'GLIM_world' and saved['units'] == 'meters'
    assert saved['coordinate_scale'] == SCALE and saved['output_mode'] == 'chunks'
    assert saved['total_chunks'] == 2 and saved['total_faces'] == 4
    assert saved['chunks'][0]['grid_index'] == [0, 0, 0] and saved['chunks'][1]['grid_index'] == [1, 0, 0]
    assert saved['chunks'][0]['field_origin_scaled'] == [0.0, 0.0, 0.0]
    assert saved['chunks'][1]['field_origin_scaled'] == [15.0, 0.0, 0.0]


def test_chunk_emitting_only_unowned_triangles_is_empty_but_safe(tmp_path):
    centers = [[0.0, 0, 0], [15.0, 0, 0]]
    # (14,0,0) lies inside only the second cube, even though the first field emitted it.
    manifest, totals = run_extract(tmp_path, centers, [[[14.0, 0, 0]], [[14.0, 0, 0]]],
                                   chunk_size_m=4.0, overlap_ratio=.25)
    assert manifest['chunks'][0]['file'] is None and manifest['chunks'][0]['vertices'] == 0
    assert manifest['chunks'][0]['world_bbox_min'] is None
    assert manifest['chunks'][1]['file'] == 'chunk_0001.ply' and manifest['chunks'][1]['faces'] == 1
    assert totals['total_faces'] == 1
    assert not (tmp_path / 'mesh_chunks' / 'chunk_0000.ply').exists()


def test_no_owned_triangles_leaves_no_chunk_files(tmp_path):
    centers = [[0.0, 0, 0], [15.0, 0, 0]]
    manifest, totals = run_extract(tmp_path, centers, [[[14.0, 0, 0]], []],
                                   chunk_size_m=4.0, overlap_ratio=.25)
    assert manifest['chunks'][0]['file'] is None and manifest['chunks'][1]['file'] is None
    assert totals['total_faces'] == 0 and totals['union_bounds'] is None


def test_single_chunk_has_unbounded_core_and_world_coordinates(tmp_path):
    manifest, _ = run_extract(tmp_path, [[3.0, 4.0, 5.0]], [[[3.0, 4.0, 5.0]]], chunk_size_m=4.0)
    assert manifest['total_chunks'] == 1 and manifest['chunks'][0]['faces'] == 1
    assert manifest['chunks'][0]['core_bbox_min'] == [None, None, None]
    assert manifest['chunks'][0]['core_bbox_max'] == [None, None, None]
    assert inspect_mesh(tmp_path / 'mesh_chunks' / 'chunk_0000.ply')['bounding_box_min'] == \
        pytest.approx([(3.0 - .2) / SCALE, 4.0 / SCALE, 5.0 / SCALE])


def test_manifest_core_bbox_is_meters_and_scaled_values_are_kept(tmp_path):
    centers = [[0.0, 0, 0], [15.0, 0, 0]]
    manifest, _ = run_extract(tmp_path, centers, [[[1.0, 0, 0]], []],
                              chunk_size_m=4.0, overlap_ratio=.25)
    chunk = manifest['chunks'][0]
    # Scaled midpoint to the neighbour is 7.5; the same bound in world metres is 1.5.
    assert chunk['core_bbox_max_scaled'][0] == pytest.approx(7.5)
    assert chunk['core_bbox_max'][0] == pytest.approx(1.5)
    assert chunk['core_bbox_max'][0] != chunk['core_bbox_max_scaled'][0]
    assert chunk['core_bbox_min'] == [None, None, None]
    assert chunk['core_bbox_min_scaled'] == [None, None, None]
    assert manifest['units'] == 'meters' and manifest['coordinate_scale'] == SCALE
    # The field origin stays explicitly scaled.
    assert manifest['chunks'][1]['field_origin_scaled'] == [15.0, 0.0, 0.0]


def test_manifest_reports_nominal_chunk_size_and_stride(tmp_path):
    manifest, _ = run_extract(tmp_path, [[0.0, 0, 0]], [[[0.0, 0, 0]]],
                              chunk_size_m=4.0, overlap_ratio=.05)
    assert manifest['chunk_size_m'] == 4.0
    assert manifest['nksr_chunk_size_scaled'] == pytest.approx(20.0)
    assert manifest['chunk_stride_scaled'] == pytest.approx(19.0)
    assert manifest['chunk_stride_source'] == 'nksr_chunk_size'
    assert manifest['overlap_ratio'] == .05


def test_write_full_single_chunk(tmp_path):
    settings = SimpleNamespace(mesh_output_mode='both')
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    manifest, totals = write_full_single_chunk(settings, vertices, faces, tmp_path / 'mesh_chunks')
    assert manifest['total_chunks'] == 1 and totals['total_faces'] == 1
    assert inspect_mesh(tmp_path / 'mesh_chunks/chunk_0000.ply')['face_count'] == 1
    assert manifest['chunks'][0]['note'].startswith('Full')
    assert manifest['coordinate_scale'] == 1.0 and manifest['overlap_ratio'] is None
    assert manifest['chunk_size_m'] is None and manifest['chunk_stride_scaled'] is None
    # A full reconstruction owns the whole field, so its core box is the mesh box.
    assert manifest['chunks'][0]['core_bbox_min'] == manifest['chunks'][0]['world_bbox_min']


# --------------------------------------------------------------------------- #
# execute() modes
# --------------------------------------------------------------------------- #

class FakeTensor:
    def __init__(self, value):
        self.value = value

    def float(self):
        return self

    def to(self, device):
        return self

    def __mul__(self, scale):
        return FakeTensor(self.value * scale)


def execute_runtime(field):
    cuda = SimpleNamespace(mem_get_info=lambda device: (8 * 1024**3, 8 * 1024**3),
                           empty_cache=lambda: None, max_memory_allocated=lambda device: 0,
                           reset_peak_memory_stats=lambda device: None)
    torch = SimpleNamespace(device=lambda name: SimpleNamespace(type=name.split(':')[0]),
                            from_numpy=FakeTensor, cuda=cuda, inference_mode=contextlib.nullcontext)
    nksr = SimpleNamespace(get_estimate_normal_preprocess_fn=lambda *a: (lambda xyz, n, s: (xyz, n, s)))
    reconstructor = SimpleNamespace(reconstruct=lambda xyz, **kwargs: field,
                                    network=SimpleNamespace(to=lambda device: None))
    return torch, nksr, reconstructor, SimpleNamespace(type='cpu')


def execute_settings(tmp_path, mode='chunked', output_mode='merged', chunk_size='4'):
    args = ['--mode', mode, '--mesh-output-mode', output_mode,
            '--output', str(tmp_path / 'output' / 'mesh.ply')]
    if chunk_size is not None:
        args += ['--chunk-size', chunk_size]
    return parser().parse_args(args)


class FusedField:
    """Stands in for nksr.FusedField: keeps the individual chunk fields and transforms."""

    def __init__(self, chunk_fields, centers):
        self.fields = chunk_fields
        self.transforms = [FakeTransform(center) for center in centers]
        self.fused_calls = []

    def to_(self, device):
        pass

    def extract_dual_mesh(self, **kwargs):
        self.fused_calls.append(kwargs)
        return SimpleNamespace(v=np.zeros((3, 3)), f=np.array([[0, 1, 2]]))


def chunk_field(centroid, center):
    return FakeField(local_triangle(centroid, center), [[0, 1, 2]])


def test_chunks_mode_does_not_call_fused_extraction(tmp_path):
    field = FusedField([chunk_field([0.0, 0, 0], [0.0, 0, 0])], [[0.0, 0, 0]])
    torch, nksr, reconstructor, device = execute_runtime(field)
    settings = execute_settings(tmp_path, output_mode='chunks')
    vertices, faces, result = execute(np.ones((70, 3)), np.ones((70, 3)) * 5, settings,
                                      lambda *a, **k: None, torch, nksr, device, reconstructor)
    assert vertices is None and faces is None
    assert field.fused_calls == []
    assert result['mesh_output_mode'] == 'chunks' and result['chunk_count'] == 1
    assert (tmp_path / 'output' / 'mesh_chunks' / 'chunk_0000.ply').is_file()
    assert not (tmp_path / 'output' / 'mesh.ply').exists()


def test_both_mode_produces_chunks_and_fused(tmp_path):
    field = FusedField([chunk_field([0.0, 0, 0], [0.0, 0, 0])], [[0.0, 0, 0]])
    torch, nksr, reconstructor, device = execute_runtime(field)
    settings = execute_settings(tmp_path, output_mode='both')
    vertices, faces, result = execute(np.ones((70, 3)), np.ones((70, 3)) * 5, settings,
                                      lambda *a, **k: None, torch, nksr, device, reconstructor)
    assert len(field.fused_calls) == 1
    assert vertices is not None and faces is not None
    assert result['chunk_count'] == 1
    assert (tmp_path / 'output' / 'mesh_chunks' / 'chunk_0000.ply').is_file()


def test_merged_mode_preserves_existing_single_extraction(tmp_path):
    field = FusedField([chunk_field([0.0, 0, 0], [0.0, 0, 0])], [[0.0, 0, 0]])
    torch, nksr, reconstructor, device = execute_runtime(field)
    settings = execute_settings(tmp_path, output_mode='merged')
    vertices, faces, result = execute(np.ones((70, 3)), np.ones((70, 3)) * 5, settings,
                                      lambda *a, **k: None, torch, nksr, device, reconstructor)
    assert len(field.fused_calls) == 1
    assert vertices is not None and result.get('chunk_count') is None
    assert not (tmp_path / 'output' / 'mesh_chunks').exists()


def test_auto_chunk_size_manifest_records_resolved_values(tmp_path):
    """Auto resolves the chunk size; the manifest still reports physical and scaled values."""
    field = FusedField([chunk_field([0.0, 0, 0], [0.0, 0, 0])], [[0.0, 0, 0]])
    torch, nksr, reconstructor, device = execute_runtime(field)
    settings = execute_settings(tmp_path, output_mode='chunks', chunk_size=None)
    assert settings.chunk_size is None
    _, _, result = execute(np.ones((70, 3)), np.ones((70, 3)) * 5, settings,
                           lambda *a, **k: None, torch, nksr, device, reconstructor)
    manifest = read_json(tmp_path / 'output' / 'mesh_chunks' / 'chunks.json')
    assert manifest['chunk_size_m'] == 20.0  # select_chunk picked 20 m for this small input
    assert manifest['nksr_chunk_size_scaled'] == pytest.approx(100.0)
    assert manifest['chunk_stride_scaled'] == pytest.approx(95.0)  # 100 * (1 - .05)
    assert manifest['overlap_ratio'] == .05
    assert result['chunk_size_m'] == 20.0
    assert result['nksr_chunk_size_scaled'] == pytest.approx(100.0)
    assert result['chunk_stride_scaled'] == pytest.approx(95.0)
    # The resolved size is what the manifest's grid indices are derived from.
    assert manifest['chunk_stride_source'] == 'nksr_chunk_size'
    assert manifest['chunks'][0]['grid_index'] == [0, 0, 0]


def test_chunks_mode_requires_chunk_fields(tmp_path):
    class Plain:
        def __init__(self):
            self.fused_calls = []

        def to_(self, device):
            pass

        def extract_dual_mesh(self, **kwargs):
            return SimpleNamespace(v=np.zeros((3, 3)), f=np.array([[0, 1, 2]]))

    torch, nksr, reconstructor, device = execute_runtime(Plain())
    settings = execute_settings(tmp_path, output_mode='chunks')
    with pytest.raises(WorkerError, match='chunk fields'):
        execute(np.ones((70, 3)), np.ones((70, 3)) * 5, settings,
                lambda *a, **k: None, torch, nksr, device, reconstructor)


def test_low_ram_rejects_non_merged_mesh_output(monkeypatch):
    from factory_mapping import nksr_worker as worker
    monkeypatch.setattr(worker.signal, 'signal', lambda *args: None)
    for output_mode in ('chunks', 'both'):
        monkeypatch.setattr(worker.sys, 'argv',
                            ['worker', '--mode', 'low_ram', '--mesh-output-mode', output_mode])
        with pytest.raises(SystemExit):
            worker.main()
    assert parser().parse_args(['--mode', 'low_ram']).mesh_output_mode == 'merged'


def test_low_ram_orchestration_rejects_non_merged_output(prepared):
    service, sid, run, _ = prepared
    settings = MeshRequest(mode='low_ram', mesh_output_mode='chunks').model_dump()
    with pytest.raises(ValueError, match='Low RAM'):
        asyncio.run(jobs.reconstruct(service, sid, run.name, settings))


# --------------------------------------------------------------------------- #
# Completion validation
# --------------------------------------------------------------------------- #

def build_chunk_output(tmp_path, mode='chunks', chunks=2):
    """Create a plausible worker output directory for one output mode."""
    output = tmp_path / 'output'
    (output / 'mesh_chunks').mkdir(parents=True)
    vertex = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    face = np.array([[0, 1, 2]], dtype=np.int64)
    entries, total_vertices, total_faces = [], 0, 0
    for index in range(chunks):
        name = f'chunk_{index:04d}.ply'
        stats = write_mesh(output / 'mesh_chunks' / name, vertex + index * 10, face)
        entries.append(dict(index=index, file=name, vertices=stats['vertex_count'],
                            faces=stats['face_count']))
        total_vertices += stats['vertex_count']
        total_faces += stats['face_count']
    atomic_json(output / 'mesh_chunks' / 'chunks.json',
                dict(version=1, units='meters', coordinate_scale=SCALE, output_mode=mode,
                     total_chunks=chunks, total_vertices=total_vertices, total_faces=total_faces,
                     chunks=entries))
    metadata = dict(mesh_output_mode=mode, actual_mode='chunked', chunk_count=chunks,
                    chunk_vertices_total=total_vertices, chunk_faces_total=total_faces)
    if mode != 'chunks':
        stats = write_mesh(output / 'mesh.ply', vertex, face)
        metadata.update(vertex_count=stats['vertex_count'], face_count=stats['face_count'])
    else:
        metadata.update(vertex_count=total_vertices, face_count=total_faces)
    atomic_json(output / 'nksr_metadata.json', metadata)
    return output


@pytest.mark.parametrize('mode', ['chunks', 'both'])
def test_validate_completed_accepts_chunk_outputs(mode, tmp_path):
    assert jobs.validate_completed(build_chunk_output(tmp_path, mode=mode), 0)['mesh_output_mode'] == mode


def test_validate_completed_rejects_missing_chunk_file(tmp_path):
    output = build_chunk_output(tmp_path, mode='both')
    (output / 'mesh_chunks' / 'chunk_0001.ply').unlink()
    with pytest.raises(ValueError, match='missing'):
        jobs.validate_completed(output, 0)


def test_validate_completed_rejects_chunk_with_wrong_counts(tmp_path):
    output = build_chunk_output(tmp_path, mode='both')
    write_mesh(output / 'mesh_chunks' / 'chunk_0001.ply', np.zeros((6, 3)),
               np.array([[0, 1, 2], [3, 4, 5]]))
    with pytest.raises(ValueError, match='counts do not match'):
        jobs.validate_completed(output, 0)


def test_validate_completed_rejects_manifest_total_mismatch(tmp_path):
    output = build_chunk_output(tmp_path, mode='both')
    manifest = read_json(output / 'mesh_chunks/chunks.json')
    manifest['total_vertices'] += 1
    atomic_json(output / 'mesh_chunks/chunks.json', manifest)
    with pytest.raises(ValueError, match='manifest totals'):
        jobs.validate_completed(output, 0)


def test_validate_completed_rejects_metadata_chunk_count_mismatch(tmp_path):
    output = build_chunk_output(tmp_path, mode='both')
    metadata = read_json(output / 'nksr_metadata.json')
    metadata['chunk_count'] = 99
    atomic_json(output / 'nksr_metadata.json', metadata)
    with pytest.raises(ValueError, match='Chunk count'):
        jobs.validate_completed(output, 0)


def test_validate_completed_rejects_metadata_chunk_total_mismatch(tmp_path):
    output = build_chunk_output(tmp_path, mode='chunks')
    metadata = read_json(output / 'nksr_metadata.json')
    metadata['chunk_faces_total'] += 2
    atomic_json(output / 'nksr_metadata.json', metadata)
    with pytest.raises(ValueError, match='totals do not match'):
        jobs.validate_completed(output, 0)


def test_validate_completed_rejects_missing_merged_mesh(tmp_path):
    output = build_chunk_output(tmp_path, mode='both')
    (output / 'mesh.ply').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        jobs.validate_completed(output, 0)


def test_validate_completed_both_mode_corrupted_merged_mesh(tmp_path):
    output = build_chunk_output(tmp_path, mode='both')
    metadata = read_json(output / 'nksr_metadata.json')
    metadata['face_count'] += 3
    atomic_json(output / 'nksr_metadata.json', metadata)
    with pytest.raises(ValueError, match='Mesh counts'):
        jobs.validate_completed(output, 0)


@pytest.mark.parametrize('name', ['../mesh.ply', '../../etc/passwd', '/etc/passwd',
                                  'sub/chunk_0000.ply', '', '.', '..'])
def test_validate_completed_rejects_chunk_path_traversal(name, tmp_path):
    output = build_chunk_output(tmp_path, mode='chunks')
    manifest = read_json(output / 'mesh_chunks/chunks.json')
    manifest['chunks'][0]['file'] = name
    atomic_json(output / 'mesh_chunks/chunks.json', manifest)
    with pytest.raises(ValueError):
        jobs.validate_completed(output, 0)


def test_validate_completed_rejects_symlinked_chunk_file(tmp_path):
    output = build_chunk_output(tmp_path, mode='chunks')
    outside = tmp_path / 'outside.ply'
    write_mesh(outside, np.zeros((3, 3)), np.array([[0, 1, 2]]))
    link = output / 'mesh_chunks/chunk_0000.ply'
    link.unlink()
    link.symlink_to(outside)
    with pytest.raises(ValueError):
        jobs.validate_completed(output, 0)


def test_validate_completed_rejects_symlinked_chunk_directory(tmp_path):
    output = build_chunk_output(tmp_path, mode='chunks')
    moved = tmp_path / 'moved_chunks'
    (output / 'mesh_chunks').rename(moved)
    (output / 'mesh_chunks').symlink_to(moved)
    with pytest.raises(ValueError, match='symlink'):
        jobs.validate_completed(output, 0)


def test_validate_completed_merged_mode_requires_valid_mesh(tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    stats = write_mesh(output / 'mesh.ply',
                       np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                       np.array([[0, 1, 2]], dtype=np.int64))
    atomic_json(output / 'nksr_metadata.json', dict(mesh_output_mode='merged', **stats))
    assert jobs.validate_completed(output, 0)['face_count'] == 1
    atomic_json(output / 'nksr_metadata.json',
                dict(mesh_output_mode='merged', vertex_count=3, face_count=7))
    with pytest.raises(ValueError, match='Mesh counts'):
        jobs.validate_completed(output, 0)
    with pytest.raises(ValueError, match='unsuccessfully'):
        jobs.validate_completed(output, 1)


def test_api_rejects_invalid_mesh_output_mode(root, monkeypatch):
    from fastapi.testclient import TestClient
    from factory_mapping.api import make_app
    calls = []
    async def run(service, sid, rid, settings): calls.append(settings); return {'state': 'RUNNING'}
    monkeypatch.setattr(jobs, 'reconstruct', run)
    with TestClient(make_app(root, True)) as client:
        url = '/api/sessions/session/reconstruction/run_0123456789ab/mesh'
        assert client.post(url, json={'mesh_output_mode': 'chunks'}).status_code == 202
        assert calls[-1]['mesh_output_mode'] == 'chunks'
        assert client.post(url, json={'mesh_output_mode': 'poisson'}).status_code == 422
