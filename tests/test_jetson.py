"""Recording-host policy, including hosts with existing processing installs."""
import asyncio
import pytest
from fastapi.testclient import TestClient
from factory_mapping.config import deployment_mode
from factory_mapping.api import make_app
from factory_mapping.service import Service
from factory_mapping import nksr_jobs, reconstruction_jobs
from factory_mapping.storage import atomic_json
from test_capture import capture, wait_state


def test_arm_defaults_to_recording(root, monkeypatch):
    (root/'.state/deployment.json').unlink()
    monkeypatch.setattr('factory_mapping.config.platform.machine', lambda: 'aarch64')
    assert deployment_mode(root) == 'record_only'
    atomic_json(root/'.state/deployment.json', {'mode': 'workstation'})
    assert deployment_mode(root) == 'workstation'
    atomic_json(root/'.state/deployment.json', {'mode': 'typo'})
    with pytest.raises(ValueError): deployment_mode(root)


def test_recording_policy_overrides_installed_tools_and_saved_settings(root):
    atomic_json(root/'.state/deployment.json', {'mode': 'record_only'})
    atomic_json(root/'.state/capture_settings.json', {'live_glim': True, 'auto_process': True})
    with TestClient(make_app(root, True)) as client:
        service = client.app.state.service
        caps = client.get('/api/capture').json()['capabilities']
        assert caps['record_only'] and not caps['processing']
        assert not caps['settings']['live_glim'] and not caps['settings']['auto_process']
        assert client.get('/api/nksr').json()['status'] == 'RECORD_ONLY'
        capture(client, 'start_scan')
        wait_state(client, 'SCANNING')
        capture(client, 'stop_scan')
        assert wait_state(client, 'COMPLETE')['outcome'] == 'recorded'
        assert not {'offline', 'glim', 'nksr', 'reconstruction'} & service.pm.items.keys()


def test_processing_entry_points_fail_before_side_effects(root):
    atomic_json(root/'.state/deployment.json', {'mode': 'record_only'})
    service = Service(root, mock=True)
    async def check():
        operations = [
            lambda: service.start_glim('missing', 'jetson_cpu'),
            lambda: service.offline('missing', 'jetson_cpu'),
            lambda: service.export('missing', 'missing'),
            lambda: service.export_edit('missing', 'missing'),
            lambda: service.open_tool('missing', 'missing', 'map_editor', []),
            lambda: service.start_validator(),
            lambda: nksr_jobs.check(service),
            lambda: nksr_jobs.reconstruct(service, 'missing', 'missing', {}),
            lambda: reconstruction_jobs.start(service, 'missing', 'missing', .02),
        ]
        for operation in operations:
            with pytest.raises(ValueError, match='Recording-only host'):
                await operation()
        assert not service.pm.items
    asyncio.run(check())
