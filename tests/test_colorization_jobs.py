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
