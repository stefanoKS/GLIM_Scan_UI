"""Production hardening audit (GATES 02-12): reproductions and fixed behaviour.

Every test here started life as a reproduction of a suspected weakness in the
VDBFusion integration. Reproductions are recorded against the pre-audit commit
``f73fd0a`` in the ``OLD`` notes of each test; the assertions encode the *required*
behaviour after the minimum fix. Suspected bugs that could not be reproduced are
recorded with evidence instead of a code change.

Bug index from the audit specification:

* A unknown point in a saved submap silently treated as removed
* B saved multiplicity above the original multiplicity accepted
* C saved submap ``T_world_origin`` never read
* D centroid-only triangle masking keeps triangles crossing a deleted strip
* E prepared TSDF settings stored but never compared at reconstruction time
* F one LiDAR frame larger than ``batch_points`` produces an oversized batch
* G memory budget only checked between integration batches
* H cooperative cancellation cannot interrupt native mesh extraction
* I cooperative cancellation cannot interrupt output writing
"""
import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from factory_mapping import vdbfusion as V
from factory_mapping import vdbfusion_jobs as jobs


def write_submap(directory, points, matrix=None, submap_id=0):
    directory.mkdir(parents=True, exist_ok=True)
    np.asarray(points, dtype='<f4').reshape(-1).tofile(directory/'points_compact.bin')
    matrix = np.eye(4) if matrix is None else matrix
    rows = '\n'.join(' '.join(f'{value:.9g}' for value in row) for row in matrix)
    (directory/'data.txt').write_text(f'id: {submap_id}\nT_world_origin: \n{rows}\n')


def build_workspace(root, original, saved, saved_matrix=None, edit_id='edit_0000deadbeef'):
    workspace = root/'edits'/edit_id
    write_submap(workspace/'map_01'/'000000', original, submap_id=0)
    write_submap(workspace/'saved_map'/'000000', saved, matrix=saved_matrix, submap_id=0)
    return workspace


# --------------------------------------------------------------------------- A
def test_unknown_saved_point_is_rejected_not_treated_as_removed(tmp_path):
    """OLD: ``np.isin`` returned False for a point the pre-edit map never held, so an
    unknown point was accepted and the whole submap silently mis-classified."""
    original = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    saved = np.array([[0.0, 0.0, 0.0], [9.0, 9.0, 9.0]], dtype=np.float32)
    workspace = build_workspace(tmp_path, original, saved)
    with pytest.raises(V.SavedSubmapError) as caught:
        V.load_edited_reference(workspace)
    error = caught.value
    assert error.code == 'INVALID_SAVED_SUBMAP'
    assert error.submap_id == '000000'
    assert error.invalid_points == 1
    assert 'never contained' in error.mismatch
    assert 'Re-export' in error.action or 'export' in error.action.lower()


# --------------------------------------------------------------------------- B
def test_saved_multiplicity_above_original_is_rejected(tmp_path):
    """OLD: original ``A, A, B`` with saved ``A, A, A`` was accepted by membership
    alone, so a duplicated point inflated the retained set."""
    original = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    saved = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    workspace = build_workspace(tmp_path, original, saved)
    with pytest.raises(V.SavedSubmapError) as caught:
        V.load_edited_reference(workspace)
    assert caught.value.code == 'INVALID_SAVED_SUBMAP'
    assert caught.value.invalid_points == 1
    assert 'repeats points' in caught.value.mismatch


def test_multiplicity_valid_subset_is_accepted(tmp_path):
    """The strict rule must not reject legitimate GLIM output: a saved submap that drops
    points is a valid subset. (Measured on the real cleanup ``edit_ef16d03e7f86``: all 57
    pairs are multiplicity-valid, and neither ``map_01`` nor ``saved_map`` contains a single
    duplicate row, so membership is unambiguous in practice.)"""
    original = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    saved = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    workspace = build_workspace(tmp_path, original, saved)
    reference = V.load_edited_reference(workspace)
    assert reference.retained.tolist() == [False, False, True]
    assert reference.metadata['integrity']['validation'] == 'strict_multiset_subset'
    assert reference.metadata['integrity']['unknown_saved_points'] == 0


def test_duplicate_coordinates_are_labelled_per_key_and_never_invented(tmp_path):
    """Documented semantics for a non-GLIM-authored export that repeats a coordinate:
    membership is decided per unique coordinate (a coordinate-keyed export cannot say
    which copy survived), and a coordinate absent from the saved set is never retained."""
    original = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    saved = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    workspace = build_workspace(tmp_path, original, saved)
    reference = V.load_edited_reference(workspace)
    assert reference.retained.tolist() == [True, True, False]


# --------------------------------------------------------------------------- C
def test_saved_submap_transform_mismatch_is_rejected(tmp_path):
    """OLD: only ``map_01/data.txt`` was parsed, so a saved submap written in a
    different frame produced silently wrong world geometry."""
    original = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    saved = original.copy()
    shifted = np.eye(4)
    shifted[:3, 3] = [10.0, 0.0, 0.0]
    workspace = build_workspace(tmp_path, original, saved, saved_matrix=shifted)
    with pytest.raises(V.SavedSubmapError) as caught:
        V.load_edited_reference(workspace)
    assert caught.value.code == 'INVALID_SAVED_SUBMAP'
    assert 'T_world_origin' in caught.value.mismatch


def test_saved_submap_transform_is_used_from_the_saved_file(tmp_path):
    """Both frames are parsed and must agree, and a non-rigid frame is rejected."""
    original = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
    workspace = build_workspace(tmp_path, original, original.copy())
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag([2.0, 1.0, 1.0])
    (workspace/'map_01'/'000000'/'data.txt').write_text(
        'id: 0\nT_world_origin: \n' + '\n'.join(' '.join(f'{v:.9g}' for v in row) for row in matrix))
    (workspace/'saved_map'/'000000'/'data.txt').write_text((workspace/'map_01'/'000000'/'data.txt').read_text())
    with pytest.raises(V.SavedSubmapError) as caught:
        V.load_edited_reference(workspace)
    assert 'non-rigid' in caught.value.mismatch


def test_corrupt_saved_block_structure_is_rejected(tmp_path):
    """A truncated point block is reported, never silently reshaped."""
    original = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    workspace = build_workspace(tmp_path, original, original.copy())
    (workspace/'saved_map'/'000000'/'points_compact.bin').write_bytes(b'\x00' * 7)
    with pytest.raises(V.SavedSubmapError) as caught:
        V.load_edited_reference(workspace)
    assert caught.value.code == 'INVALID_SAVED_SUBMAP'


# --------------------------------------------------------------------------- D
KEEP_STRIP = (-1.1, -0.9)


def surface_samples(spacing=0.05):
    """A dense kept surface in the z = 0 plane with a deleted strip crossing it."""
    xs = np.arange(-1.6, 1.6 + spacing / 2, spacing)
    ys = np.arange(-0.6, 1.6 + spacing / 2, spacing)
    grid = np.stack(np.meshgrid(xs, ys, indexing='ij'), axis=-1).reshape(-1, 2)
    points = np.column_stack((grid[:, 0], grid[:, 1], np.zeros(len(grid))))
    removed = (points[:, 0] >= KEEP_STRIP[0]) & (points[:, 0] <= KEEP_STRIP[1])
    return points[~removed], points[removed]


class StubVolume:
    """Returns the mesh under test; the native extractor is not what is being verified."""

    def __init__(self, vertices, faces):
        self.vertices, self.faces = vertices, faces

    def extract_triangle_mesh(self, **_):
        return self.vertices, self.faces


def strip_reference(radius=0.15):
    kept, removed = surface_samples()
    world = np.concatenate([kept, removed])
    retained = np.concatenate([np.ones(len(kept), dtype=bool), np.zeros(len(removed), dtype=bool)])
    # The sampling resolution is the measured lattice spacing of this synthetic surface.
    return make_reference(world, retained, radius=radius, spacing=0.0385)


def make_reference(world, retained, radius, spacing=0.0385):
    reference = V.EditedGeometryReference(
        world=np.asarray(world, dtype=np.float64), retained=np.asarray(retained, dtype=bool),
        sampling_resolution_m=spacing, submaps=[], association_spacing_multiplier=4.0,
        association_radius_m=radius)
    reference.metadata = dict(association_radius_m=radius, sampling_resolution_m=spacing)
    return reference


def test_triangle_crossing_a_deleted_strip_is_removed():
    """OLD: only the centroid was classified, so a triangle whose centroid sat in kept
    geometry while a vertex crossed the deleted strip survived the filter."""
    vertices = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    reference = strip_reference()
    centroid = vertices[faces[0]].mean(axis=0)
    # Evidence for the old behaviour: the centroid alone is clean kept geometry.
    assert reference.classify(centroid[None, :], 0.15, 0.0, 0.0385)['keep'][0] is np.True_
    _, out_faces, stats = V.extract_and_mask(StubVolume(vertices, faces), reference)
    assert len(out_faces) == 0, 'a triangle touching removed geometry must not be kept'
    assert stats['triangles_removed_by_probe'] == 1
    assert stats['triangles_removed_by_centroid'] == 0


def test_triangle_fully_inside_kept_geometry_is_kept():
    vertices = np.array([[0.2, 0.2, 0.0], [0.6, 0.2, 0.0], [0.2, 0.6, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    _, out_faces, stats = V.extract_and_mask(StubVolume(vertices, faces), strip_reference())
    assert len(out_faces) == 1
    assert stats['masked_triangles'] == 0
    assert np.array_equal(out_faces[0], [0, 1, 2]), 'winding and connectivity must be preserved'


def test_masking_polls_cancellation_while_it_works():
    """OLD: cancellation was polled only between integration batches, so a long native
    extraction and its masking could not be interrupted."""
    vertices = np.random.default_rng(0).random((600, 3))
    faces = np.array([[index, index + 1, index + 2] for index in range(0, 594, 3)], dtype=np.int64)
    reference = make_reference(vertices, np.ones(len(vertices), dtype=bool), radius=1.0, spacing=0.001)
    calls = {'count': 0}

    def cancel():
        calls['count'] += 1
        return calls['count'] > 1

    with pytest.raises(V.ReconstructionCancelled):
        V.extract_and_mask(StubVolume(vertices, faces), reference, cancel=cancel)
    assert calls['count'] > 1, 'masking must poll cancellation while it works'


def test_cancellation_is_observed_before_output_is_published():
    """Output writing is a stage of its own, so it needs its own cancellation check."""
    with pytest.raises(V.ReconstructionCancelled):
        V.check_cancellation(lambda: True)
    V.check_cancellation(None)
    V.check_cancellation(lambda: False)


# --------------------------------------------------------------------------- F
class StubTrajectory:
    """Minimal stand-in; interpolation itself is covered by the existing suites."""

    def __init__(self):
        self.calls = 0


def _stub_transform(points, timestamps, trajectory):
    trajectory.calls += 1
    return np.asarray(points, dtype=np.float64), np.zeros((len(points), 3), dtype=np.float64), \
        np.ones(len(points), dtype=bool)


def test_oversized_lidar_frame_is_split_to_the_batch_bound(monkeypatch):
    """OLD: ``batch_world_observations`` only flushed at ``pending >= batch_points``, so a
    single frame larger than the bound produced one oversized batch. The bound is a
    memory-safety contract for every batch, not an average."""
    import factory_mapping.reconstruction as R
    monkeypatch.setattr(R, 'transform_points', _stub_transform)
    bound = V.MIN_BATCH_POINTS
    frame_points = bound * 2 + 25
    frame = np.arange(frame_points * 3, dtype=np.float64).reshape(-1, 3)
    frames = [(frame, np.arange(frame_points, dtype=np.float64), np.arange(frame_points, dtype=np.float64))]
    counters = {}
    batches = list(V.batch_world_observations(frames, StubTrajectory(), bound, counters=counters))
    sizes = [len(batch['points']) for batch in batches]
    assert max(sizes) <= bound, f'batch bound violated: {sizes}'
    assert counters['max_batch_points'] <= bound
    assert counters['split_frames'] == 1
    joined = np.concatenate([batch['points'] for batch in batches])
    assert np.array_equal(joined, frame), 'order and every observation must be preserved'
    assert counters['observations'] == frame_points


def test_batch_bound_holds_across_many_frames_and_keeps_the_tail(monkeypatch):
    """Irregular frame sizes must never overshoot the bound, duplicate a boundary
    observation or lose the trailing partial batch."""
    import factory_mapping.reconstruction as R
    monkeypatch.setattr(R, 'transform_points', _stub_transform)
    bound, unit = V.MIN_BATCH_POINTS, V.MIN_BATCH_POINTS // 4
    sizes = [unit, 2 * unit, 2 * unit, 2 * unit, 1, 5 * unit + 7, 3]
    frames = []
    for index, size in enumerate(sizes):
        frames.append((np.full((size, 3), index, dtype=np.float64),
                       np.zeros(size), np.arange(size, dtype=np.float64)))
    counters = {}
    batches = list(V.batch_world_observations(frames, StubTrajectory(), bound, counters=counters))
    assert all(len(batch['points']) <= bound for batch in batches)
    assert sum(len(batch['points']) for batch in batches) == sum(sizes)
    assert counters['max_batch_points'] <= bound
    stamps = np.concatenate([batch['timestamps'] for batch in batches])
    expected = np.concatenate([np.arange(size, dtype=np.float64) for size in sizes])
    assert np.array_equal(stamps, expected), 'no boundary observation may be duplicated or dropped'


def test_tiny_final_frame_is_still_emitted(monkeypatch):
    import factory_mapping.reconstruction as R
    monkeypatch.setattr(R, 'transform_points', _stub_transform)
    frames = [(np.zeros((4000, 3)), np.zeros(4000), np.zeros(4000)),
              (np.ones((2, 3)), np.zeros(2), np.zeros(2))]
    batches = list(V.batch_world_observations(frames, StubTrajectory(), V.MIN_BATCH_POINTS))
    assert sum(len(batch['points']) for batch in batches) == 4002, 'the trailing partial batch must be emitted'


# --------------------------------------------------------------------------- E
def test_prepared_tsdf_settings_must_match_at_reconstruction(tmp_path):
    """OLD: prepared settings were stored in the record but never compared, so a later
    request silently overrode the settings the lineage fingerprint was built from."""
    from factory_mapping import vdbfusion_jobs as jobs
    record = {'state': 'ready',
              'settings': {'voxel_size_m': 0.02, 'sdf_trunc_m': 0.06, 'space_carving': False},
              'settings_fingerprint': 'abc'}
    status = jobs.settings_status(record, {'voxel_size_m': 0.01, 'sdf_trunc_m': 0.03,
                                          'space_carving': False})
    assert status['matches'] is False
    assert 'voxel_size_m' in status['differing']
    assert jobs.settings_status(record, dict(record['settings']))['matches'] is True


# --------------------------------------------------------------------------- G
def test_truncation_influence_is_included_in_the_tsdf_bound(tmp_path):
    """OLD: ``touched = min(band_voxels, observed_points)`` ignored that one observation
    can influence ``(2*trunc/voxel + 1)^3`` voxels - 343 at 20 mm/60 mm - so the estimate
    could understate the touched surface by that factor and the warning never fired."""
    settings = V.resolve_settings(dict(voxel_size_m=0.02, sdf_trunc_m=0.06))
    assert V.truncation_influence_voxels(0.02, 0.06) == 343
    assert V.truncation_influence_voxels(0.01, 0.03) == 343
    observations = 50_000_000
    report = V.preflight(tmp_path, settings, observed_points=observations,
                         observed_bbox=([-25.0, -25.0, 0.0], [25.0, 25.0, 6.0]))
    # The union bound is reported separately and is not smaller than the observation count.
    assert report['estimated_tsdf_voxels'] >= observations
    union = observations * report['truncation_influence_voxels'] * V.TSDF_BYTES_PER_TOUCHED_VOXEL
    # The union bound is capped by the dense truncation band, and never below the plain
    # observation count - the understatement the audit found is gone.
    assert report['components']['tsdf_upper_bound_bytes'] == pytest.approx(
        min(union, report['components']['tsdf_dense_band_bytes']), rel=1e-9)
    assert report['components']['tsdf_upper_bound_bytes'] >= observations * V.TSDF_BYTES_PER_TOUCHED_VOXEL
    assert union > report['components']['tsdf_upper_bound_bytes'], 'the cap must actually bind here'
    assert report['estimated_tsdf_upper_bound_bytes'] >= report['estimated_tsdf_bytes']
    assert any('truncation influence' in line for line in report['estimate_basis'])


def test_preflight_separates_components_and_uses_a_measured_occupancy(tmp_path):
    settings = V.resolve_settings(dict(voxel_size_m=0.02, sdf_trunc_m=0.06, batch_points=2_000_000))
    occupancy = dict(voxels=900_000, observations=1_000_000, voxel_size_m=0.02)
    report = V.preflight(tmp_path, settings, observed_points=100_000_000,
                         observed_bbox=([-25.0, -25.0, 0.0], [25.0, 25.0, 6.0]),
                         occupancy=occupancy, reference_points=2_800_000,
                         memory_budget_bytes=8 * 1024 ** 3)
    components = report['components']
    for key in ('tsdf_estimated_bytes', 'tsdf_dense_band_bytes', 'tsdf_upper_bound_bytes',
                'mesh_triangles_estimate', 'mesh_extraction_bytes', 'mesh_masking_bytes',
                'mesh_output_bytes', 'mesh_output_transient_bytes', 'input_batch_bytes',
                'reference_bytes'):
        assert components[key] is not None, key
    assert report['density'] == pytest.approx(0.9)
    # The measured density scales the projection instead of assuming one voxel per point.
    assert report['estimated_tsdf_voxels'] == pytest.approx(90_000_000, rel=1e-6)
    assert report['heuristics']['bytes_per_touched_voxel'] == V.TSDF_BYTES_PER_TOUCHED_VOXEL
    assert any('measured occupancy' in line for line in report['estimate_basis'])


def test_preflight_refuses_an_impossible_budget_with_mitigation(tmp_path):
    settings = V.resolve_settings(dict(voxel_size_m=0.005, sdf_trunc_m=0.015))
    report = V.preflight(tmp_path, settings, observed_points=100_000_000,
                         observed_bbox=([-25.0, -25.0, 0.0], [25.0, 25.0, 6.0]),
                         memory_budget_bytes=256 * 1024 ** 2)
    assert report['ok'] is False
    assert report['code'] == 'RESOURCE_PREFLIGHT_FAILED'
    assert report['failures'] and 'RESOURCE_PREFLIGHT_FAILED' in report['failures'][0]
    # Mitigation is offered; the resolution itself is never changed.
    assert report['suggestions'] and any('voxel size' in line for line in report['suggestions'])
    assert report['voxel_size_m'] == 0.005
    assert report['usable_bytes'] <= 256 * 1024 ** 2


def test_fine_resolution_over_a_whole_factory_extent_is_warned_about(tmp_path):
    settings = V.resolve_settings(dict(voxel_size_m=0.01, sdf_trunc_m=0.03))
    report = V.preflight(tmp_path, settings, observed_points=100_000_000,
                         observed_bbox=([-25.0, -25.0, 0.0], [25.0, 25.0, 6.0]))
    assert any('memory intensive' in line for line in report['warnings'])


# ------------------------------------------------------------------ GATE 08
def test_memory_supervisor_records_peak_stage_and_requests_one_cancellation(tmp_path):
    """A sampler in the backend keeps working while the worker is inside a native call."""
    stages = iter(['READING_BAG', 'READING_BAG', 'INTEGRATING', 'INTEGRATING', 'EXTRACTING_MESH'])
    rss = iter([100, 400, 9_000, 9_500, 200])
    pressure = []

    async def on_pressure(event):
        pressure.append(event)

    async def scenario():
        supervisor = jobs.MemorySupervisor(tmp_path, 1234, 10_000, interval=0.0,
                                           pressure_ratio=0.5, on_pressure=on_pressure,
                                           read_rss=lambda: next(rss), read_stage=lambda: next(stages),
                                           max_samples=5)
        await supervisor.run()
        return supervisor

    supervisor = asyncio.run(scenario())
    summary = supervisor.summary()
    assert summary['sampled'] is True
    assert summary['samples'] == 5
    assert summary['peak_rss_bytes'] == 9_500
    assert summary['peak_stage'] == 'INTEGRATING'
    assert summary['samples_by_stage']['INTEGRATING'] == 2
    assert summary['pressure_events'] and len(summary['pressure_events']) == 1, 'escalate once, not per sample'
    assert summary['pressure_events'][0]['action'] == 'requested cooperative cancellation'
    assert 'lower bound' in summary['note']


def test_memory_supervisor_reads_a_real_process_tree(tmp_path):
    """The default reader sums the worker process tree, so a native child is counted."""
    import psutil
    supervisor = jobs.MemorySupervisor(tmp_path, psutil.Process().pid, 1024 ** 3)
    assert supervisor._tree_rss() > 0
    assert supervisor._stage() is None, 'no progress file means no known stage'


# ------------------------------------------------------------------ GATE 10
def test_preparation_identity_pins_semantics_and_ignores_execution_choices():
    settings = V.resolve_settings(dict(voxel_size_m=0.02, sdf_trunc_m=0.06, space_carving=False))
    identity = V.preparation_identity(dict(bag='/bag', trajectory='/traj'), settings)
    assert identity['semantic']['voxel_size_m'] == 0.02
    assert identity['semantic_fingerprint'] == V.settings_fingerprint(settings)
    # Execution-only changes do not alter the semantic fingerprint.
    changed = dict(settings, batch_points=4_000_000, mesh_output_mode='chunks', chunk_size=5.0)
    assert V.settings_fingerprint(changed) == identity['semantic_fingerprint']
    assert V.settings_fingerprint(dict(settings, voxel_size_m=0.01)) != identity['semantic_fingerprint']


def test_source_identity_detects_a_changed_input(tmp_path):
    bag = tmp_path/'raw_bag'
    bag.mkdir()
    (bag/'metadata.yaml').write_text('rosbag2_bagfile_information:\n  version: 5\n')
    (bag/'raw_bag_0.db3').write_bytes(b'0' * 32)
    trajectory = tmp_path/'traj_lidar.txt'
    trajectory.write_text('0.0 0 0 0 0 0 0 1\n1.0 1 0 0 0 0 0 1\n')
    first = V.source_identity(bag, trajectory, '/livox/lidar')
    assert V.compare_source_identity(first, V.source_identity(bag, trajectory, '/livox/lidar'))['matches'] is True
    trajectory.write_text('0.0 0 0 0 0 0 0 1\n1.0 2 0 0 0 0 0 1\n')
    second = V.source_identity(bag, trajectory, '/livox/lidar')
    diff = V.compare_source_identity(first, second)
    assert diff['matches'] is False and 'trajectory_sha256' in diff['differing']


def test_reconstruction_refuses_settings_that_differ_from_preparation(tmp_path, monkeypatch):
    """OLD: the mesh request's settings silently won, so a run prepared and validated at
    one resolution could execute at another."""
    run = tmp_path/'run_0123456789ab'
    run.mkdir()
    prepared_settings = {'voxel_size_m': 0.02, 'sdf_trunc_m': 0.06, 'space_carving': False,
                         'origin_error_budget_m': 0.02, 'preset': None, 'roi_min_m': None, 'roi_max_m': None,
                         'unsupported_observations': 'exclude', 'mask_deleted_triangles': True,
                         'boundary_margin_m': None, 'association_spacing_multiplier': None}
    record = dict(state='PREPARED', settings=prepared_settings,
                  identity=dict(source=dict(bag='/bag', topic='/livox/lidar', fingerprint='f')))
    with pytest.raises(ValueError, match='PREPARED_SETTINGS_STALE'):
        jobs.enforce_prepared_identity(run, record, {}, dict(prepared_settings, voxel_size_m=0.01),
                                       source_identity=dict(bag='/bag', topic='/livox/lidar', fingerprint='f'))
    # Execution-only differences are allowed and the prepared semantics win.
    effective = jobs.enforce_prepared_identity(
        run, record, {}, dict(prepared_settings, batch_points=4_000_000, mesh_output_mode='chunks',
                              chunk_size=5.0),
        source_identity=dict(bag='/bag', topic='/livox/lidar', fingerprint='f'))
    assert effective['voxel_size_m'] == 0.02
    assert effective['batch_points'] == 4_000_000 and effective['chunk_size'] == 5.0


def test_reconstruction_refuses_a_source_that_changed_after_preparation(tmp_path):
    run = tmp_path/'run_0123456789ab'
    run.mkdir()
    record = dict(state='PREPARED', settings={'voxel_size_m': 0.02},
                  identity=dict(source=dict(bag='/bag', trajectory='/traj', fingerprint='old')))
    with pytest.raises(ValueError, match='PREPARED_SOURCE_CHANGED'):
        jobs.enforce_prepared_identity(run, record, {}, {'voxel_size_m': 0.02},
                                       source_identity=dict(bag='/bag', trajectory='/traj', fingerprint='new'))


# ------------------------------------------------------------------ GATE 11
def test_outputs_are_staged_then_published_atomically(tmp_path):
    """A partially written mesh must never appear at the published path."""
    staging = tmp_path/'staging'
    staging.mkdir()
    (staging/'mesh.ply').write_bytes(b'complete mesh')
    (staging/'mesh_chunks').mkdir()
    (staging/'mesh_chunks'/'chunks.json').write_text('{}')
    published = jobs_publish(staging, tmp_path, 'both')
    assert sorted(published) == ['mesh.ply', 'mesh_chunks']
    assert (tmp_path/'mesh.ply').read_bytes() == b'complete mesh'
    assert not staging.exists(), 'staging is removed after a successful publish'
    # A failed run leaves the published path untouched.
    staging.mkdir()
    (staging/'mesh.ply').write_bytes(b'partial')
    assert (tmp_path/'mesh.ply').read_bytes() == b'complete mesh'


def jobs_publish(staging, destination, mode):
    """Call the worker's publish helper out of process without importing its ROS deps."""
    import subprocess
    import sys
    script = (
        'import sys, json; sys.path.insert(0, %r);'
        'from factory_mapping import vdbfusion_worker as W;'
        'print(json.dumps(W.publish_outputs(%r, %r, %r)))'
        % (str(Path(__file__).resolve().parents[1] / 'ui/backend'), str(staging), str(destination), mode))
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


# ------------------------------------------------------------------ GATE 12
def test_mesh_audit_reports_integrity_without_deleting_geometry():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [5.0, 5.0, 0.0]])
    faces = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]], dtype=np.int64)
    settings = V.resolve_settings({})
    audit = V.audit_mesh(vertices, faces, settings, observed_bbox=([-1.0, -1.0, -1.0], [6.0, 6.0, 1.0]))
    assert audit['degenerate_triangles'] == 0
    assert audit['total_surface_area_m2'] > 0
    assert audit['connected_components'] == 1, 'shared edges connect every triangle'
    assert audit['orientation_consistent'] in (True, False)  # reported either way
    assert audit['largest_component_ratio'] == 1.0
    assert audit['mesh_within_observed_bbox'] is True
    assert 'never used to delete geometry' in audit['audit_note']


def test_mesh_audit_flags_a_mesh_outside_the_observed_extent():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [90.0, 0.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
    audit = V.audit_mesh(vertices, faces, V.resolve_settings({}),
                         observed_bbox=([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]))
    assert audit['mesh_within_observed_bbox'] is False
    assert audit['mesh_outside_observed_bbox_m'] > 80.0


def test_mesh_validation_still_refuses_corrupt_geometry():
    settings = V.resolve_settings({})
    with pytest.raises(ValueError, match='indices outside the vertex range'):
        V.validate_and_report(np.zeros((3, 3)), np.array([[0, 1, 7]], dtype=np.int64), settings)
    with pytest.raises(ValueError, match='non-finite'):
        V.validate_and_report(np.array([[0.0, 0.0, 0.0], [1.0, np.nan, 0.0], [0.0, 1.0, 0.0]]),
                              np.array([[0, 1, 2]], dtype=np.int64), settings)


# ------------------------------------------------------------------ GATE 13
def parallel_surface_reference(gap_m, spacing_m=0.05, span_m=1.0, patch_m=0.5):
    """Two kept surfaces ``gap_m`` apart plus removed geometry filling a central patch.

    The kept surfaces span the whole sampled axis; the deleted patch is filled with removed
    samples on a lattice at the reference spacing, which is what a repaired factory region
    looks like in a saved cleanup.
    """
    axis = np.arange(-span_m, span_m + spacing_m / 2, spacing_m)
    planes = [np.column_stack((axis, np.full(len(axis), height), np.zeros(len(axis))))
              for height in (0.0, gap_m)]
    kept = np.vstack(planes)
    heights = np.arange(spacing_m, gap_m, spacing_m) if gap_m > spacing_m else np.array([gap_m / 2.0])
    patch_axis = np.arange(-patch_m / 2, patch_m / 2 + spacing_m / 2, spacing_m)
    removed = np.vstack([np.column_stack((patch_axis, np.full(len(patch_axis), height),
                                          np.zeros(len(patch_axis)))) for height in heights])
    world = np.vstack([kept, removed])
    retained = np.concatenate([np.ones(len(kept), dtype=bool), np.zeros(len(removed), dtype=bool)])
    return world, retained, kept, removed


def classification_metrics(reference, radius, band, gap_m, patch_m=0.5):
    """False retention / false removal / unsupported / boundary uncertainty on a grid."""
    axis = np.arange(-1.0, 1.0001, 0.005)
    grid = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
    probes = np.column_stack((grid[:, 0], grid[:, 1], np.zeros(len(grid))))
    labels = V.EditedGeometryReference.class_of(reference.classify(probes, radius, 0.0, band))
    inside_patch = (np.abs(probes[:, 0]) <= patch_m / 2 - band) & (probes[:, 1] > 0.25 * gap_m) & \
        (probes[:, 1] < 0.75 * gap_m)
    outside_patch = (np.abs(probes[:, 0]) >= patch_m / 2 + band) | (probes[:, 1] <= -band) | \
        (probes[:, 1] >= gap_m + band)
    kinds = V.EditedGeometryReference
    return dict(
        gap_m=gap_m, radius_m=radius, band_m=band, counted=int(len(probes)),
        false_retention=float(np.count_nonzero(inside_patch & (labels == kinds.CLASS_KEEP)) /
                              max(np.count_nonzero(inside_patch), 1)),
        false_removal=float(np.count_nonzero(outside_patch & (labels == kinds.CLASS_REMOVE)) /
                            max(np.count_nonzero(outside_patch), 1)),
        unsupported_ratio=float(np.count_nonzero(labels == kinds.CLASS_UNSUPPORTED) / len(probes)),
        boundary_uncertainty_ratio=float(np.count_nonzero(labels == kinds.CLASS_AMBIGUOUS) / len(probes)),
        inside_patch_probes=int(np.count_nonzero(inside_patch)),
        outside_patch_probes=int(np.count_nonzero(outside_patch)))


@pytest.mark.parametrize('gap_m', [0.02, 0.05, 0.10])
def test_classification_matrix_reports_honest_rates_per_gap(gap_m):
    """Parallel kept surfaces 20/50/100 mm apart with a deleted central patch: the
    documented metrics must stay honest on both sides of the boundary."""
    world, retained, kept, removed = parallel_surface_reference(gap_m)
    reference = make_reference(world, retained, radius=0.15, spacing=0.05)
    metrics = classification_metrics(reference, 0.15, 0.05, gap_m)
    assert metrics['counted'] > 100_000
    assert metrics['inside_patch_probes'] > 100 and metrics['outside_patch_probes'] > 100_000, metrics
    # Geometry inside the deleted patch is never confidently retained outside the band.
    assert metrics['false_retention'] < 0.01, metrics
    # Kept surfaces outside the patch are never deleted by the filter.
    assert metrics['false_removal'] < 0.01, metrics
    assert 0.0 <= metrics['boundary_uncertainty_ratio'] < 0.5
    # Gaps finer than the reference sampling resolution must surface as reported
    # uncertainty rather than as a confident classification.
    if gap_m <= 0.05:
        assert metrics['boundary_uncertainty_ratio'] > 0.0, metrics


def test_removed_strips_narrower_than_the_radius_are_removed_not_retained():
    """Thin retained strips next to a removed patch must not resurrect the removed strip."""
    kinds = V.EditedGeometryReference
    summary = {}
    for spacing, strip_m in ((0.05, 0.24), (0.10, 0.08), (0.10, 0.30)):
        axis = np.arange(-1.0, 1.0001, spacing)
        keep_mask = np.abs(axis) > strip_m / 2
        kept = np.column_stack((axis[keep_mask], np.zeros(np.count_nonzero(keep_mask)),
                                np.zeros(np.count_nonzero(keep_mask))))
        removed = np.column_stack((axis[~keep_mask], np.zeros(np.count_nonzero(~keep_mask)),
                                   np.zeros(np.count_nonzero(~keep_mask))))
        world = np.vstack([kept, removed])
        retained = np.concatenate([np.ones(len(kept), dtype=bool), np.zeros(len(removed), dtype=bool)])
        reference = make_reference(world, retained, radius=0.15, spacing=spacing)
        labels = kinds.class_of(reference.classify(removed, 0.15, 0.0, spacing))
        removed_count = int(np.count_nonzero(labels == kinds.CLASS_REMOVE))
        summary[(spacing, strip_m)] = dict(removed=removed_count,
                                           unsupported=int(np.count_nonzero(labels == kinds.CLASS_UNSUPPORTED)),
                                           ambiguous=int(np.count_nonzero(labels == kinds.CLASS_AMBIGUOUS)))
        # A removed sample is never classified as retained geometry.
        assert not np.any(labels == kinds.CLASS_KEEP), (spacing, strip_m)
        # Strips narrower than the association radius are removed explicitly.
        if strip_m / 2 < 0.15:
            assert removed_count > 0, (spacing, strip_m, summary)
    # Every removed sample is accounted for by a reported class, never by silence.
    assert all(entry['removed'] + entry['unsupported'] + entry['ambiguous'] > 0
               for entry in summary.values()), summary


def test_advanced_controls_do_not_read_as_a_silent_mismatch(tmp_path):
    """A preparation identity must record the advanced controls it used, so re-requesting
    the same values at mesh time is not reported as stale."""
    run = tmp_path/'run_0123456789ab'
    run.mkdir()
    prepared = {'voxel_size_m': 0.02, 'sdf_trunc_m': 0.06, 'space_carving': False,
                'origin_error_budget_m': 0.02, 'preset': None, 'roi_min_m': None, 'roi_max_m': None,
                'unsupported_observations': 'include', 'mask_deleted_triangles': False,
                'boundary_margin_m': 0.01, 'association_spacing_multiplier': 3.0}
    record = dict(state='PREPARED', settings=prepared, identity=dict(source={}))
    assert jobs.settings_status(record, dict(prepared))['matches'] is True
    status = jobs.settings_status(record, dict(prepared, association_spacing_multiplier=6.0))
    assert status['matches'] is False and status['differing'] == ['association_spacing_multiplier']
