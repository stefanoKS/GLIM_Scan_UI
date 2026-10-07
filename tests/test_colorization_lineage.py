"""Colorization targets must come from the same run as the colorized trajectory.

Automatic resolution never falls back to "newest file": an unknown or ambiguous
lineage raises so a run cannot silently receive another run's colors.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from factory_mapping import colorization as c

ROOT = Path(__file__).resolve().parents[1]


def _trajectory(session, run):
    path = session/f'processing/{run}/glim_dump/traj_lidar.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, [[1, 0, 0, 0, 0, 0, 0, 1], [3, 2, 0, 0, 0, 0, 0, 1]])
    return path


def _export(session, run, suffix):
    path = session/f'exports/{run}_{suffix}.ply'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'ply\n')
    return path


def _mesh_run(session, run, trajectory, state='COMPLETED'):
    folder = session/f'reconstruction/{run}'
    (folder/'output').mkdir(parents=True, exist_ok=True)
    relative = str(trajectory.relative_to(session))
    (folder/'job.json').write_text(json.dumps(dict(state='PREPARED', trajectory=relative,
                                                   created_at='2026-10-06')))
    (folder/'mesh_job.json').write_text(json.dumps(dict(state=state)))
    (folder/'output/mesh.ply').write_bytes(b'ply\n')
    return folder


@pytest.fixture
def runs(tmp_path):
    """Two processing runs, two exports and two reconstruction runs."""
    session = tmp_path/'session'
    session.mkdir()
    first = _trajectory(session, 'run_001')
    second = _trajectory(session, 'run_002')
    _export(session, 'run_001', 'aaaa1111')
    _export(session, 'run_002', 'bbbb2222')  # sorts last: "newest" is the wrong run
    _mesh_run(session, 'run_aaaaaaaaaaaa', first)
    _mesh_run(session, 'run_bbbbbbbbbbbb', second)
    edits = session/'edits/edit_1/saved_map'
    edits.mkdir(parents=True)
    np.savetxt(edits/'traj_lidar.txt', [[1, 0, 0, 0, 0, 0, 0, 1], [3, 2, 0, 0, 0, 0, 0, 1]])
    return session, first, second


def test_processing_run_id_is_derived_from_the_trajectory_path(runs):
    session, first, second = runs
    assert c.processing_run_id(session, first) == 'run_001'
    assert c.processing_run_id(session, second) == 'run_002'
    assert c.processing_run_id(session, session/'edits/edit_1/saved_map/traj_lidar.txt') is None
    assert c.processing_run_id(session, None) is None
    assert c.processing_run_id(ROOT, ROOT/'config/system.yaml') is None


def test_glim_export_comes_from_the_colorized_run_not_the_newest(runs):
    session, first, _ = runs
    path, provenance = c.resolve_glim_ply(session, trajectory_path=first)
    assert path.name == 'run_001_aaaa1111.ply'
    assert provenance['glim_ply_lineage'] == 'run_001'


def test_ambiguous_glim_export_fails_instead_of_guessing(runs):
    session, first, _ = runs
    _export(session, 'run_001', 'cccc3333')
    with pytest.raises(ValueError, match='Multiple GLIM exports exist for run_001'):
        c.resolve_glim_ply(session, trajectory_path=first)


def test_missing_glim_export_fails_safely(tmp_path):
    session = tmp_path/'session'
    session.mkdir()
    trajectory = _trajectory(session, 'run_003')
    _export(session, 'run_001', 'aaaa1111')
    with pytest.raises(ValueError, match='No GLIM PLY export found for run_003'):
        c.resolve_glim_ply(session, trajectory_path=trajectory)


def test_edited_trajectory_requires_an_explicit_export(runs):
    session, _, _ = runs
    edited = session/'edits/edit_1/saved_map/traj_lidar.txt'
    with pytest.raises(ValueError, match='no GLIM processing run'):
        c.resolve_glim_ply(session, trajectory_path=edited)
    path, provenance = c.resolve_glim_ply(session, session/'exports/run_002_bbbb2222.ply')
    assert path.name == 'run_002_bbbb2222.ply' and provenance['glim_ply_lineage'] == 'explicit'


def test_explicit_export_must_live_inside_the_session(runs, tmp_path):
    session, _, _ = runs
    outside = tmp_path/'elsewhere.ply'
    outside.write_bytes(b'ply\n')
    with pytest.raises(ValueError, match='inside this session'):
        c.resolve_glim_ply(session, outside)
    with pytest.raises(ValueError, match='GLIM PLY not found'):
        c.resolve_glim_ply(session, 'exports/missing.ply')


def test_nksr_mesh_is_resolved_by_trajectory(runs):
    session, first, second = runs
    path, provenance = c.resolve_nksr_mesh(session, trajectory_path=first)
    assert provenance['source_reconstruction_run'] == 'run_aaaaaaaaaaaa'
    assert provenance['nksr_mesh_lineage'] == 'run_aaaaaaaaaaaa'
    assert path == session/'reconstruction/run_aaaaaaaaaaaa/output/mesh.ply'
    other, other_provenance = c.resolve_nksr_mesh(session, trajectory_path=second)
    assert other_provenance['source_reconstruction_run'] == 'run_bbbbbbbbbbbb'


def test_nksr_mesh_for_another_run_is_never_selected(runs):
    session, first, _ = runs
    # The other run's mesh geometrically exists and is newer, but it belongs to
    # run_002, so an auto-resolution for run_001 must not choose it.
    path, _ = c.resolve_nksr_mesh(session, trajectory_path=first)
    assert 'run_bbbbbbbbbbbb' not in str(path)
    # A mesh whose worker never completed is not a candidate either.
    mesh_job = session/'reconstruction/run_aaaaaaaaaaaa/mesh_job.json'
    mesh_job.write_text(json.dumps(dict(state='RUNNING')))
    with pytest.raises(ValueError, match='No NKSR mesh is associated'):
        c.resolve_nksr_mesh(session, trajectory_path=first)


def test_ambiguous_nksr_mesh_fails_instead_of_guessing(runs):
    session, first, _ = runs
    _mesh_run(session, 'run_cccccccccccc', first)
    with pytest.raises(ValueError, match='Multiple NKSR meshes exist'):
        c.resolve_nksr_mesh(session, trajectory_path=first)


def test_missing_nksr_mesh_fails_safely(runs):
    session, _, second = runs
    import shutil
    shutil.rmtree(session/'reconstruction/run_bbbbbbbbbbbb')
    with pytest.raises(ValueError, match='No NKSR mesh is associated'):
        c.resolve_nksr_mesh(session, trajectory_path=second)
    with pytest.raises(ValueError, match='--nksr-mesh explicitly'):
        c.resolve_nksr_mesh(session)


def _tool_module():
    spec = importlib.util.spec_from_file_location('glim_colorize_tool', ROOT/'tools/glim_colorize.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_records_target_lineage_in_colorization_metadata(tmp_path, runs):
    session, first, _ = runs
    output = tmp_path/'run_out'
    output.mkdir()
    (output/'metadata.json').write_text(json.dumps(dict(source_processing_run='run_001',
                                                        source_trajectory=str(first))))
    tool = _tool_module()
    trajectory, metadata = tool._lineage(output)
    assert trajectory == first and metadata['source_processing_run'] == 'run_001'
    target, provenance = c.resolve_glim_ply(session, trajectory_path=trajectory)
    provenance['source_processing_run'] = metadata.get('source_processing_run')
    tool._record_transfer(output, 'glim_transfer', dict(vertices_total=3), provenance)
    written = json.loads((output/'metadata.json').read_text())
    assert written['source_glim_ply'].endswith('run_001_aaaa1111.ply')
    assert written['source_processing_run'] == 'run_001'
    assert written['glim_transfer']['vertices_total'] == 3
    assert str(target).endswith('run_001_aaaa1111.ply')


def test_tool_rejects_metadata_without_a_source_trajectory(tmp_path):
    output = tmp_path/'run_out'
    output.mkdir()
    (output/'metadata.json').write_text(json.dumps(dict(source_trajectory='/nonexistent/traj.txt')))
    tool = _tool_module()
    with pytest.raises(ValueError, match='no usable source trajectory'):
        tool._lineage(output)
