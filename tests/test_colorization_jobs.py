"""Colorization job orchestration and API wiring; no ROS or heavy work."""
import asyncio

from factory_mapping.storage import atomic_json, read_json


def test_colorization_job_start_and_view(root, monkeypatch):
    from factory_mapping.service import Service
    from factory_mapping.colorization_jobs import start, view
    service = Service(root, mock=True)
    m = service.sessions.create('colorize', '', service.config)
    session = service.sessions.get(m['id'])
    (session/'raw_bag').mkdir()
    (session/'raw_bag/metadata.yaml').write_text('test')
    config = read_json(session/'active_config.json', {})
    config['system'] = {'camera': {'enabled': True}}
    config['camera'] = {'image_topic': '/camera/image_raw', 'camera_info_topic': '/camera/camera_info'}
    atomic_json(session/'active_config.json', config)
    calls = []

    async def launch(*args):
        calls.append(args)
    monkeypatch.setattr(service.pm, 'start', launch)
    service.mock = False
    job = asyncio.run(start(service, m['id'],
                            {'allow_unvalidated_calibration': True, 'voxel_size': 0.02,
                             'transfer_glim': True}))
    argv = calls[0][1]
    assert '--allow-unvalidated-calibration' in argv
    assert argv[argv.index('--voxel-size') + 1] == '0.02'
    assert '--transfer-glim' in argv
    assert '--output-dir' in argv and '--progress-json' in argv
    data = view(service, m['id'])
    assert data['camera_enabled'] is True and data['raw_bag'] is True
    assert len(data['jobs']) == 1 and data['jobs'][0]['id'] == job['id']


def test_colorization_job_requires_rgb(root, monkeypatch):
    from factory_mapping.service import Service
    from factory_mapping.colorization_jobs import start
    service = Service(root, mock=True)
    m = service.sessions.create('no rgb', '', service.config)
    session = service.sessions.get(m['id'])
    (session/'raw_bag').mkdir()
    (session/'raw_bag/metadata.yaml').write_text('test')
    service.mock = False
    try:
        asyncio.run(start(service, m['id'], {}))
        raise AssertionError('Expected ValueError for a session without RGB')
    except ValueError as error:
        assert 'RGB' in str(error)


def test_api_colorization_endpoint(root, monkeypatch):
    from fastapi.testclient import TestClient
    from factory_mapping.api import make_app
    from factory_mapping import colorization_jobs
    calls = []

    async def start(service, sid, settings):
        calls.append((sid, settings))
        return {'state': 'running'}
    monkeypatch.setattr(colorization_jobs, 'start', start)
    with TestClient(make_app(root, True)) as client:
        response = client.post('/api/sessions/test/colorization',
                               json={'allow_unvalidated_calibration': True, 'transfer_glim': True})
        assert response.status_code == 202
        assert calls[-1][0] == 'test'
        assert calls[-1][1]['allow_unvalidated_calibration'] is True
        assert calls[-1][1]['transfer_glim'] is True
        assert client.post('/api/sessions/test/colorization',
                           json={'allow_unvalidated_calibration': 'x'}).status_code == 422


def test_colorization_job_forwards_observation_and_transfer_knobs(root, monkeypatch):
    from factory_mapping.service import Service
    from factory_mapping.colorization_jobs import start
    service = Service(root, mock=True)
    m = service.sessions.create('knobs', '', service.config)
    session = service.sessions.get(m['id'])
    (session/'raw_bag').mkdir()
    (session/'raw_bag/metadata.yaml').write_text('test')
    config = read_json(session/'active_config.json', {})
    config['system'] = {'camera': {'enabled': True}}
    config['camera'] = {'image_topic': '/camera/image_raw', 'camera_info_topic': '/camera/camera_info'}
    atomic_json(session/'active_config.json', config)
    calls = []

    async def launch(*args):
        calls.append(args)
    monkeypatch.setattr(service.pm, 'start', launch)
    service.mock = False
    import asyncio
    asyncio.run(start(service, m['id'], {
        'max_color_observations_per_voxel': 2, 'transfer_radius': 0.04, 'transfer_k': 3,
        'depth_edge_rejection': True, 'depth_edge_radius': 2, 'depth_edge_threshold': 0.1,
        'surface_transfer_radius': 0.03, 'surface_transfer_k': 4}))
    argv = calls[0][1]
    for flag, value in (('--max-color-observations-per-voxel', '2'), ('--transfer-radius', '0.04'),
                        ('--transfer-k', '3'), ('--depth-edge-radius', '2'),
                        ('--depth-edge-threshold', '0.1'), ('--surface-transfer-radius', '0.03'),
                        ('--surface-transfer-k', '4')):
        assert argv[argv.index(flag) + 1] == value, flag
    assert '--depth-edge-rejection' in argv


def test_colorization_request_rejects_out_of_range_knobs(root):
    from fastapi.testclient import TestClient
    from factory_mapping.api import make_app
    with TestClient(make_app(root, True)) as client:
        for body in ({'max_color_observations_per_voxel': 0}, {'transfer_k': 0},
                     {'depth_edge_radius': 9}, {'transfer_radius': 0.0}):
            assert client.post('/api/sessions/test/colorization', json=body).status_code == 422, body
        assert client.post('/api/sessions/test/colorization',
                           json={'max_color_observations_per_voxel': 4, 'depth_edge_rejection': True}
                           ).status_code != 422
