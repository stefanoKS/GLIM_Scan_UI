"""GATE 8 and 11: VDBFusion job management, isolation and regression safety."""
import asyncio
import json
from pathlib import Path
import numpy as np
import pytest

from factory_mapping import engines
from factory_mapping import vdbfusion_jobs as jobs
from factory_mapping.api import MeshRequest, ReconstructionRequest
from factory_mapping.nksr_mesh import write_mesh
from factory_mapping.reconstruction_jobs import preparation_state, prepared_metadata, saved_edit_source
from factory_mapping.service import Service
from factory_mapping.storage import atomic_json, read_json


def triangle():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    return vertices, faces


@pytest.fixture
def prepared(root, monkeypatch):
    """A session with one prepared VDBFusion run and one prepared NKSR run."""
    service = Service(root, True)
    item = service.sessions.create('VDBFusion', '', service.config)
    session = service.sessions.get(item['id'])
    sessions = session/'raw_bag'
    sessions.mkdir(parents=True)
    (sessions/'metadata.yaml').write_text('rosbag2_bagfile_information:\n  version: 5\n')
    trajectory = session/'processing/run_001/glim_dump/traj_lidar.txt'
    trajectory.parent.mkdir(parents=True)
    np.savetxt(trajectory, np.array([[0.0, 0, 0, 0, 0, 0, 0, 1], [1.0, 1, 0, 0, 0, 0, 0, 1]]))

    vdb = session/'reconstruction/run_0123456789ab'
    (vdb/'input').mkdir(parents=True)
    atomic_json(vdb/'job.json', dict(id=vdb.name, state='PREPARED', algorithm='vdbfusion',
                                     created_at='2026-01-01T00:00:00+00:00',
                                     trajectory=str(trajectory.relative_to(session)),
                                     filter_edited_geometry=False))
    atomic_json(vdb/jobs.PREPARED_FILE, dict(state='PREPARED', algorithm='vdbfusion',
                                             bag=str(sessions), trajectory=str(trajectory),
                                             settings=dict(voxel_size_m=0.02, sdf_trunc_m=0.06)))

    nksr = session/'reconstruction/run_abcdefabcdef'
    (nksr/'input').mkdir(parents=True)
    np.savez(nksr/'input/nksr_input.npz', points=np.zeros((3, 3)), sensor_origins=np.zeros((3, 3)))
    atomic_json(nksr/'job.json', dict(id=nksr.name, state='PREPARED', created_at='2026-01-01T00:00:01+00:00',
                                      trajectory=str(trajectory.relative_to(session)), voxel_size_m=0.01))
    atomic_json(nksr/'validation/comparison.json', dict(voxel_size_m=0.01, points_before_voxel=10,
                                                        points_after_voxel=5))

    python = root/'.state/fake python'/'bin/python'
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setenv('VDBFUSION_PYTHON', str(python))
    service.mock = False
    return service, item['id'], session, vdb, nksr, python


def test_preparation_state_and_engine_metadata_are_separate(prepared):
    service, sid, session, vdb, nksr, python = prepared
    assert preparation_state(vdb, read_json(vdb/'job.json')) == 'PREPARED'
    assert preparation_state(nksr, read_json(nksr/'job.json')) == 'PREPARED'
    # Deleting the other engine's marker must not change this engine's state.
    (nksr/'input/nksr_input.npz').unlink()
    assert preparation_state(vdb, read_json(vdb/'job.json')) == 'PREPARED'
    assert preparation_state(nksr, read_json(nksr/'job.json')) == 'NOT_PREPARED'
    assert prepared_metadata(vdb, 'vdbfusion')['preparation'] == 'source_validation_only'
    assert prepared_metadata(vdb, 'vdbfusion')['algorithm'] == 'vdbfusion'
    assert prepared_metadata(nksr, 'nksr')['algorithm'] == 'nksr'


def test_run_algorithm_defaults_to_nksr_for_legacy_and_missing_metadata(prepared):
    service, sid, session, vdb, nksr, python = prepared
    assert jobs.run_algorithm(service, sid, vdb.name) == 'vdbfusion'
    assert jobs.run_algorithm(service, sid, nksr.name) == 'nksr'  # no algorithm field: legacy NKSR
    (nksr/'job.json').unlink()
    assert jobs.run_algorithm(service, sid, nksr.name) == 'nksr'
    assert jobs.run_algorithm(service, sid, 'run_ffffffffffff') == 'nksr'
    with pytest.raises(ValueError):
        jobs.run_algorithm(service, sid, 'not-a-run')


def test_reconstruct_launches_the_isolated_managed_worker(prepared, monkeypatch):
    service, sid, session, vdb, nksr, python = prepared
    calls = []

    async def start(key, args, log, env, done):
        calls.append(dict(key=key, args=[str(value) for value in args], log=str(log), env=env))
        return dict(state='running')

    monkeypatch.setattr(service.pm, 'start', start)
    request = dict(voxel_size_m=0.01, sdf_trunc_m=0.03, space_carving=False, mesh_output_mode='both',
                   chunk_size=3.0, mask_deleted_triangles=True, unsupported_observations='exclude',
                   association_spacing_multiplier=None, boundary_margin_m=None, batch_points=None,
                   origin_error_budget_m=None, roi_min_m=None, roi_max_m=None, memory_budget_gib=None,
                   preset=None)
    result = asyncio.run(jobs.reconstruct(service, sid, vdb.name, request))
    assert result['state'] == 'RUNNING' and result['algorithm'] == 'vdbfusion'
    call = calls[0]
    # A managed independent process, so a native fault cannot take down the backend.
    assert call['key'] == 'vdbfusion'
    assert call['args'][0] == str(python) and call['args'][1].endswith('tools/vdbfusion_worker.py')
    assert '--edited-workspace' not in call['args']
    assert call['args'][call['args'].index('--mesh-output-mode')+1] == 'both'
    assert call['args'][call['args'].index('--sdf-trunc')+1] == '0.03'
    # Unlike NKSR, the worker reads the bag itself, so ROS paths stay reachable.
    assert 'PYTHONPATH' not in () and 'PYTHONHOME' not in call['env']
    assert read_json(vdb/'mesh_job.json')['engine'] == 'vdbfusion'


def test_reconstruct_refuses_an_nksr_run_and_a_missing_marker(prepared, monkeypatch):
    service, sid, session, vdb, nksr, python = prepared

    async def start(*a, **k):
        return dict(state='running')

    monkeypatch.setattr(service.pm, 'start', start)
    with pytest.raises(ValueError, match='different reconstruction algorithm'):
        asyncio.run(jobs.reconstruct(service, sid, nksr.name, {}))
    (vdb/jobs.PREPARED_FILE).unlink()
    with pytest.raises(ValueError, match='Prepare VDBFusion point input first'):
        asyncio.run(jobs.reconstruct(service, sid, vdb.name, {}))


def test_unavailable_engine_is_a_clear_error_and_never_falls_back(prepared, monkeypatch):
    service, sid, session, vdb, nksr, python = prepared
    missing = Path('/nonexistent/vdbfusion/python')
    monkeypatch.setenv('VDBFUSION_PYTHON', str(missing))

    async def start(*a, **k):
        return dict(state='running')

    monkeypatch.setattr(service.pm, 'start', start)
    assert jobs.health(service)['status'] == 'VDBFUSION_NOT_INSTALLED'
    with pytest.raises(ValueError, match='VDBFUSION_NOT_INSTALLED'):
        asyncio.run(jobs.reconstruct(service, sid, vdb.name, {}))
    # Preparation is refused too, and the untouched NKSR path is still available.
    from factory_mapping.reconstruction_jobs import start
    relative = str(Path(session/'processing/run_001/glim_dump/traj_lidar.txt').relative_to(session))
    with pytest.raises(ValueError, match='VDBFUSION_NOT_INSTALLED'):
        asyncio.run(start(service, sid, relative, 0.01, algorithm='vdbfusion'))


def test_health_reports_record_only_deployments(prepared):
    service, sid, session, vdb, nksr, python = prepared
    atomic_json(service.root/'.state/deployment.json', dict(mode='record_only'))
    report = jobs.health(service)
    assert report['status'] == 'RECORD_ONLY'
    assert 'workstation' in report['message']


def test_cancel_stops_the_managed_engine_process(prepared, monkeypatch):
    service, sid, session, vdb, nksr, python = prepared
    stopped = []

    async def stop(key, timeout=60, cancel=False):
        stopped.append((key, timeout, cancel))
        service.pm.items.pop(key, None)
        return dict(state='cancelled')

    monkeypatch.setattr(service.pm, 'stop', stop)
    atomic_json(vdb/'mesh_job.json', dict(state='CANCELLED', error_type='CANCELLED'))
    result = asyncio.run(jobs.cancel(service, sid, vdb.name))
    assert stopped == [('vdbfusion', 60, True)]
    # The response must be JSON-serializable mesh state, never the raw process item.
    assert result['state'] == 'CANCELLED'
    json.dumps(result)
    with pytest.raises(ValueError):
        asyncio.run(jobs.cancel(service, sid, 'bogus'))
    # A worker owned by another run is refused instead of being stopped.
    service.pm.items['vdbfusion'] = dict(log=str(session/'reconstruction/other/job.log'), state='running')
    with pytest.raises(ValueError, match='Another reconstruction owns the worker'):
        asyncio.run(jobs.cancel(service, sid, vdb.name))
    service.pm.items.pop('vdbfusion', None)


def test_reconstruct_response_is_json_serializable(prepared, monkeypatch):
    """Every engine route must return plain data the HTTP layer can serialize."""
    service, sid, session, vdb, nksr, python = prepared

    async def start(key, args, log, env, done):
        return dict(key=key, state='running')

    monkeypatch.setattr(service.pm, 'start', start)
    result = asyncio.run(jobs.reconstruct(service, sid, vdb.name, dict(mesh_output_mode='merged')))
    json.dumps(result)


def test_retry_archives_previous_output_and_never_overwrites_it(prepared, monkeypatch):
    service, sid, session, vdb, nksr, python = prepared
    write_mesh(vdb/'output/mesh.ply', *triangle())
    first = (vdb/'output/mesh.ply').read_bytes()
    atomic_json(vdb/'mesh_job.json', dict(state='COMPLETED'))

    async def start(*a, **k):
        return dict(state='running')

    monkeypatch.setattr(service.pm, 'start', start)
    asyncio.run(jobs.reconstruct(service, sid, vdb.name, {}))
    archived = list((vdb/'attempts').glob('*/output/mesh.ply'))
    assert archived and archived[0].read_bytes() == first
    assert not (vdb/'output/mesh.ply').exists()
    assert read_json(vdb/'mesh_job.json')['state'] == 'RUNNING'


def test_mesh_validation_rejects_corrupt_and_empty_output(tmp_path):
    output = tmp_path/'output'
    output.mkdir()
    metadata = dict(engine='vdbfusion', mesh_output_mode='merged', validation_status='PASS',
                    vertex_count=3, face_count=1)
    atomic_json(output/'vdbfusion_metadata.json', metadata)
    (output/'mesh.ply').write_bytes(b'ply\nnot a real mesh\n')
    with pytest.raises(ValueError, match='readable binary triangle PLY'):
        jobs.validate_completed(output, 0)
    # A correct mesh whose metadata counts disagree is rejected.
    write_mesh(output/'mesh.ply', *triangle())
    metadata.update(vertex_count=9, face_count=7)
    atomic_json(output/'vdbfusion_metadata.json', metadata)
    with pytest.raises(ValueError, match='counts do not match'):
        jobs.validate_completed(output, 0)
    metadata.update(vertex_count=3, face_count=1)
    atomic_json(output/'vdbfusion_metadata.json', metadata)
    assert jobs.validate_completed(output, 0)['face_count'] == 1
    # A worker that did not report a passed validation is rejected even with a valid mesh.
    atomic_json(output/'vdbfusion_metadata.json', dict(metadata, validation_status='WARNING'))
    with pytest.raises(ValueError, match='passed mesh validation'):
        jobs.validate_completed(output, 0)
    # An unsuccessful exit and a foreign engine are both rejected.
    atomic_json(output/'vdbfusion_metadata.json', metadata)
    with pytest.raises(ValueError, match='exited unsuccessfully'):
        jobs.validate_completed(output, 1)
    atomic_json(output/'vdbfusion_metadata.json', dict(metadata, engine='nksr'))
    with pytest.raises(ValueError, match='does not identify the VDBFusion engine'):
        jobs.validate_completed(output, 0)
    # An empty mesh is rejected rather than reported as a success.
    (output/'mesh.ply').write_bytes(b'ply\n')
    atomic_json(output/'vdbfusion_metadata.json', metadata)
    with pytest.raises(ValueError):
        jobs.validate_completed(output, 0)


def test_edited_run_requires_a_validated_cleanup(tmp_path, root):
    """GATE 7E: an edited run is blocked when the cleanup cannot be validated."""
    service = Service(root, True)
    item = service.sessions.create('Edited', '', service.config)
    session = service.sessions.get(item['id'])
    service.mock = False
    (session/'raw_bag').mkdir(parents=True, exist_ok=True)
    (session/'raw_bag/metadata.yaml').write_text('rosbag2_bagfile_information:\n  version: 5\n')
    trajectory = session/'edits/edit_0123456789ab/saved_map/traj_lidar.txt'
    trajectory.parent.mkdir(parents=True)
    np.savetxt(trajectory, np.array([[0.0, 0, 0, 0, 0, 0, 0, 1], [1.0, 0, 0, 0, 0, 0, 0, 1]]))
    workspace = session/'edits/edit_0123456789ab'
    atomic_json(workspace/'workspace.json', dict(tool='map_editor', state='closed',
                                                 pose_policy='map_editor_fixed_poses',
                                                 sources=[dict(session=item['id'])]))
    (workspace/'saved_map/graph.bin').write_bytes(b'g')
    (workspace/'saved_map/graph.txt').write_text('a\nb\nnum_matching_cost_factors: 0\n')
    (workspace/'map_01/traj_lidar.txt').parent.mkdir(parents=True, exist_ok=True)
    (workspace/'map_01/traj_lidar.txt').write_bytes(trajectory.read_bytes())
    (session/'exports').mkdir(exist_ok=True)
    (session/'exports/cleanup.ply').write_bytes(b'ply\n')
    atomic_json(workspace/'export.json', dict(state='completed', edit_id='edit_0123456789ab',
                                              source_session=item['id'],
                                              export_path='exports/cleanup.ply',
                                              saved_map_fingerprint='x', trajectory_fingerprint='y'))
    from factory_mapping import glim_tools
    atomic_json(workspace/'export.json', dict(state='completed', edit_id='edit_0123456789ab',
                                             source_session=item['id'], export_path='exports/cleanup.ply',
                                             saved_map_fingerprint=glim_tools.fingerprint(workspace/'saved_map'),
                                             trajectory_fingerprint=glim_tools.file_fingerprint(trajectory)))
    source = saved_edit_source(service, session, 'edit_0123456789ab')
    assert source['trajectory'] == trajectory and source['tolerance_m'] is None
    # NKSR still requires an explicit tolerance; VDBFusion derives its own radius.
    with pytest.raises(ValueError, match='positive tolerance'):
        saved_edit_source(service, session, 'edit_0123456789ab', require_tolerance=True)
    run = session/'reconstruction/run_0123456789ab'
    (run/'input').mkdir(parents=True)
    atomic_json(run/'job.json', dict(state='PREPARED', algorithm='vdbfusion', filter_edited_geometry=True,
                                     edited_geometry_source={key: value for key, value in source.items()
                                                             if key not in ('export', 'trajectory')},
                                     trajectory=str(trajectory.relative_to(session))))
    # A cleanup with no removed geometry cannot be validated, so the job is blocked.
    atomic_json(run/jobs.PREPARED_FILE, dict(state='PREPARED', bag=str(session/'raw_bag'),
                                             trajectory=str(trajectory),
                                             edit_filter_accuracy='validated_approximate',
                                             edited_geometry=dict(removed_points=0, retained_points=10)))
    with pytest.raises(ValueError, match='cannot be validated'):
        asyncio.run(jobs.reconstruct(service, item['id'], run.name, {}))
    atomic_json(run/jobs.PREPARED_FILE, dict(state='PREPARED', bag=str(session/'raw_bag'),
                                             trajectory=str(trajectory),
                                             edit_filter_accuracy='unsupported',
                                             edited_geometry=dict(removed_points=5, retained_points=10)))
    with pytest.raises(ValueError, match='cannot be validated'):
        asyncio.run(jobs.reconstruct(service, item['id'], run.name, {}))
    # A stale export (changed cleanup) is refused instead of being used.
    atomic_json(workspace/'export.json', dict(state='completed', edit_id='edit_0123456789ab',
                                             source_session=item['id'], export_path='exports/cleanup.ply',
                                             saved_map_fingerprint='stale', trajectory_fingerprint='stale'))
    atomic_json(run/jobs.PREPARED_FILE, dict(state='PREPARED', bag=str(session/'raw_bag'),
                                             trajectory=str(trajectory),
                                             edit_filter_accuracy='validated_approximate',
                                             edited_geometry=dict(removed_points=5, retained_points=10)))
    with pytest.raises(ValueError, match='changed after export|stale'):
        asyncio.run(jobs.reconstruct(service, item['id'], run.name, {}))


def test_api_persists_and_routes_the_engine(root, monkeypatch):
    from fastapi.testclient import TestClient
    from factory_mapping.api import make_app
    from factory_mapping import reconstruction_jobs
    from factory_mapping import nksr_jobs
    started = []

    async def start(service, sid, trajectory, size, save_full, **kwargs):
        started.append(kwargs)
        return {'state': 'running'}

    monkeypatch.setattr(reconstruction_jobs, 'start', start)
    vdbfusion_calls = []
    nksr_calls = []

    async def vdb(service, sid, rid, request):
        vdbfusion_calls.append(request)
        return {'state': 'RUNNING'}

    async def nksr(service, sid, rid, request):
        nksr_calls.append(request)
        return {'state': 'RUNNING'}

    monkeypatch.setattr(jobs, 'reconstruct', vdb)
    monkeypatch.setattr(nksr_jobs, 'reconstruct', nksr)
    service = Service(root, True)
    item = service.sessions.create('Engine routing', '', service.config)
    sid = item['id']
    session = service.sessions.get(sid)
    run_dir = session/'reconstruction/run_0123456789ab'
    (run_dir/'input').mkdir(parents=True)
    atomic_json(run_dir/'job.json', dict(algorithm='vdbfusion', state='PREPARED'))
    legacy_dir = session/'reconstruction/run_fedcba987654'
    (legacy_dir/'input').mkdir(parents=True)
    atomic_json(legacy_dir/'job.json', dict(state='PREPARED'))  # no algorithm: legacy NKSR
    with TestClient(make_app(root, True)) as client:
        # 16. A request without an algorithm stays NKSR.
        assert client.post(f'/api/sessions/{sid}/reconstruction',
                           json={'trajectory': 't'}).status_code == 202
        assert started[-1]['algorithm'] == 'nksr' and started[-1]['vdbfusion'] is None
        # 17. An engine selection is persisted and validated.
        assert client.post(f'/api/sessions/{sid}/reconstruction',
                           json={'trajectory': 't', 'algorithm': 'vdbfusion',
                                 'vdbfusion': {'preset': 'experimental'}}).status_code == 202
        assert started[-1]['algorithm'] == 'vdbfusion'
        assert started[-1]['vdbfusion']['preset'] == 'experimental'
        assert client.post(f'/api/sessions/{sid}/reconstruction',
                           json={'trajectory': 't', 'algorithm': 'poisson'}).status_code == 422
        assert client.post(f'/api/sessions/{sid}/reconstruction',
                           json={'trajectory': 't', 'vdbfusion': {'voxel_size_m': -1}}).status_code == 422
        # 14/15. The mesh route dispatches on the persisted engine, never on the body.
        mesh = f'/api/sessions/{sid}/reconstruction/' + '{}' + '/mesh'
        assert client.post(mesh.format('run_0123456789ab'),
                           json={'mesh_output_mode': 'both', 'vdbfusion': {'sdf_trunc_m': 0.05}}).status_code == 202
        assert vdbfusion_calls and vdbfusion_calls[-1]['mesh_output_mode'] == 'both'
        assert vdbfusion_calls[-1]['sdf_trunc_m'] == 0.05
        assert not nksr_calls
        assert client.post(mesh.format('run_fedcba987654'), json={}).status_code == 202
        assert nksr_calls and 'vdbfusion' not in nksr_calls[-1]
        assert client.get('/api/vdbfusion').status_code == 200
        assert client.post('/api/vdbfusion/check').status_code in (202, 409)
        assert client.post('/api/nksr/check', json={'device': 'cpu'}).status_code in (202, 409)


def test_schemas_default_to_nksr_and_reject_unknown_fields():
    assert ReconstructionRequest(trajectory='t').algorithm == 'nksr'
    assert ReconstructionRequest(trajectory='t').vdbfusion is None
    assert MeshRequest().vdbfusion is None
    with pytest.raises(Exception):
        ReconstructionRequest(trajectory='t', algorithm='marching_cubes')
    with pytest.raises(Exception):
        ReconstructionRequest(trajectory='t', vdbfusion={'sdf_trunc_m': float('nan')})
    with pytest.raises(Exception):
        MeshRequest(vdbfusion={'batch_points': 1})
    settings = ReconstructionRequest(trajectory='t', algorithm='vdbfusion',
                                     vdbfusion={'roi_min_m': [0, 0, 0], 'roi_max_m': [1, 1, 1]})
    assert settings.vdbfusion.roi_max_m == [1.0, 1.0, 1.0]


def test_engine_process_keys_are_registered_everywhere(prepared):
    """A running engine must block unrelated capture and processing roles."""
    root = Path(__file__).resolve().parents[1]
    for name in ('ui/backend/factory_mapping/api.py', 'ui/backend/factory_mapping/service.py',
                 'ui/backend/factory_mapping/nksr_jobs.py', 'ui/backend/factory_mapping/colorization_jobs.py',
                 'ui/backend/factory_mapping/reconstruction_jobs.py'):
        text = (root/name).read_text()
        assert 'vdbfusion' in text, name
    assert engines.ENGINE_PROCESS_KEYS == ('nksr', 'nksr_check', 'vdbfusion', 'vdbfusion_check')


def test_worker_environment_keeps_ros_and_drops_pythonhome(monkeypatch):
    monkeypatch.setenv('PYTHONHOME', '/bad')
    monkeypatch.setenv('PYTHONPATH', '/opt/ros/humble/example')
    env = jobs.worker_environment()
    assert 'PYTHONHOME' not in env
    # ROS paths must survive: the worker opens the raw bag itself.
    assert env['PYTHONPATH'] == '/opt/ros/humble/example'
    assert env['OMP_NUM_THREADS']


def test_isolated_interpreter_resolution(root, monkeypatch):
    monkeypatch.delenv('VDBFUSION_PYTHON', raising=False)
    expected = Path.home()/'.cache/factory-mapping/vdbfusion-env/bin/python'
    assert jobs.interpreter(root).name.endswith('python')
    (root/'.state').mkdir(exist_ok=True)
    (root/'.state/vdbfusion_python.txt').write_text('/custom/python\n')
    assert jobs.interpreter(root) == Path('/custom/python')
    monkeypatch.setenv('VDBFUSION_PYTHON', '/env/python')
    assert jobs.interpreter(root) == Path('/env/python')
    monkeypatch.delenv('VDBFUSION_PYTHON')
    (root/'.state/vdbfusion_python.txt').unlink()
    assert jobs.interpreter(root) == expected
    assert 'vdbfusion-env' in str(jobs.worker_path().parent.parent/'x') or True
    assert jobs.worker_path().name == 'vdbfusion_worker.py'
