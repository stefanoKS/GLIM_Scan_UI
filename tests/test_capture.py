import asyncio
import json
import time
import pytest
from fastapi.testclient import TestClient
from factory_mapping.api import make_app
from factory_mapping.storage import read_json, atomic_json


def wait_state(client, state, timeout=8):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        current=client.get('/api/capture').json()
        if current['state']==state:return current
        time.sleep(.05)
    raise AssertionError(current)


def capture(client, action):
    response=client.post('/api/capture/action',json={'action':action})
    assert response.status_code==202,response.text
    return response.json()


def test_one_click_scan_auto_session_and_processing(root):
    with TestClient(make_app(root,True)) as client:
        started=capture(client,'start_scan');sid=started['session']
        assert started['state']=='PREFLIGHT'
        wait_state(client,'SCANNING')
        assert client.post('/api/capture/action',json={'action':'start_scan'}).status_code==409
        capture(client,'stop_scan')
        wait_state(client,'PROCESSING')
        complete=wait_state(client,'COMPLETE')
        assert complete['outcome']=='processed'
        p=root/'data/sessions'/sid
        assert (p/'raw_bag/MOCK_ONLY.txt').is_file()
        assert read_json(p/'metadata.json')['bag_finalized']
        assert read_json(p/'processing/run_001/job.json')['state']=='completed'
        assert client.patch('/api/sessions/'+sid,json={'name':'Zone A','notes':'Completed route'}).status_code==200
        assert read_json(p/'metadata.json')['name']=='Zone A'
        assert client.post('/api/capture/action',json={'action':'shell'}).status_code==422


def test_record_only_host_skips_glim_and_processing(root,monkeypatch):
    with TestClient(make_app(root,True)) as client:
        s=client.app.state.service;monkeypatch.setattr(s,'glim_available',lambda:False)
        async def forbidden(*args,**kwargs):raise AssertionError('GLIM must not run')
        monkeypatch.setattr(s,'start_glim',forbidden);monkeypatch.setattr(s,'offline',forbidden)
        assert client.get('/api/capture').json()['capabilities']['record_only']
        capture(client,'start_scan');wait_state(client,'SCANNING');time.sleep(.2)
        capture(client,'stop_scan');result=wait_state(client,'COMPLETE')
        assert result['outcome']=='recorded' and result['error'] is None
        assert 'offline' not in s.pm.items


@pytest.mark.parametrize('failure',['startup','runtime'])
def test_live_glim_failure_never_stops_recording(root,monkeypatch,failure):
    with TestClient(make_app(root,True)) as client:
        s=client.app.state.service
        client.put('/api/capture/settings',json={'live_glim':True,'auto_process':False})
        if failure=='startup':
            async def fail(*args,**kwargs):raise ValueError('Estimator unavailable')
            monkeypatch.setattr(s,'start_glim',fail)
        capture(client,'start_scan');wait_state(client,'SCANNING');time.sleep(.2)
        if failure=='runtime':
            client.portal.call(s.pm.stop,'glim',2)
            s.pm.items['glim']['state']='failed'
            client.portal.call(s.capture.reconcile)
        assert s.pm.active('recording')
        assert client.get('/api/capture').json()['warnings']
        capture(client,'stop_scan');assert wait_state(client,'COMPLETE')['outcome']=='recorded'


def test_camera_only_has_no_lidar_dependency_or_glim(root,monkeypatch):
    with TestClient(make_app(root,True)) as client:
        s=client.app.state.service
        async def forbidden(*args,**kwargs):raise AssertionError('Camera recording must not start LiDAR/GLIM')
        monkeypatch.setattr(s,'start_driver',forbidden);monkeypatch.setattr(s,'start_glim',forbidden)
        started=capture(client,'start_camera_recording');wait_state(client,'SCANNING')
        assert client.post('/api/capture/action',json={'action':'stop_scan'}).status_code==409
        time.sleep(.2);capture(client,'stop_camera_recording');assert wait_state(client,'COMPLETE')['outcome']=='recorded'
        p=root/'data/sessions'/started['session'];m=read_json(p/'metadata.json')
        assert m['kind']=='camera' and set(m['topics'].values())=={'/camera/image_raw','/camera/camera_info'}
        assert not s.config['system']['camera']['enabled'] # independent of scan's RGB preference
        assert client.post('/api/action',json={'action':'process','session':m['id']}).status_code==409
        assert client.get('/api/camera/preview').status_code==200


def test_preflight_failure_does_not_claim_complete_scan(root,monkeypatch):
    with TestClient(make_app(root,True)) as client:
        s=client.app.state.service
        async def fail():raise ValueError('LiDAR disconnected')
        monkeypatch.setattr(s,'prepare_recording',fail)
        started=capture(client,'start_scan');result=wait_state(client,'COMPLETE')
        assert result['outcome']=='failed' and 'disconnected' in result['error']
        p=root/'data/sessions'/started['session']
        assert not (p/'raw_bag').exists() and read_json(p/'metadata.json')['state']=='failed'


def test_stop_during_preflight_avoids_live_glim(root,monkeypatch):
    with TestClient(make_app(root,True)) as client:
        s=client.app.state.service
        client.put('/api/capture/settings',json={'live_glim':True,'auto_process':True})
        original=s.prepare_recording
        async def slow():await asyncio.sleep(.3);await original()
        monkeypatch.setattr(s,'prepare_recording',slow)
        capture(client,'start_scan');capture(client,'stop_scan');wait_state(client,'COMPLETE')
        assert not s.pm.active('recording') and 'glim' not in s.pm.items and 'offline' not in s.pm.items


def test_recorder_failure_finalizes_without_automatic_processing(root):
    with TestClient(make_app(root,True)) as client:
        s=client.app.state.service
        started=capture(client,'start_scan');wait_state(client,'SCANNING');time.sleep(.2)
        client.portal.call(s.pm.stop,'recording',2)
        s.pm.items['recording']['state']='failed';s.pm.items['recording']['returncode']=7
        client.portal.call(s.capture.reconcile)
        result=client.get('/api/capture').json()
        assert result['state']=='COMPLETE' and result['outcome']=='failed'
        assert 'offline' not in s.pm.items and (root/'data/sessions'/started['session']/'raw_bag').exists()


def test_shutdown_finalizes_scan_and_recovery_never_claims_success(root):
    with TestClient(make_app(root,True)) as client:
        sid=capture(client,'start_scan')['session'];wait_state(client,'SCANNING');time.sleep(.2)
    assert read_json(root/'data/sessions'/sid/'metadata.json')['bag_finalized']
    state=root/'.state/capture.json';m=read_json(state);m['state']='SCANNING';atomic_json(state,m)
    with TestClient(make_app(root,True)) as client:
        assert client.get('/api/capture').json()['outcome']=='interrupted'


def test_advanced_stop_remains_compatible_and_settings_guarded(root):
    with TestClient(make_app(root,True)) as client:
        client.put('/api/capture/settings',json={'live_glim':False,'auto_process':False})
        capture(client,'start_scan');wait_state(client,'SCANNING')
        assert client.post('/api/action',json={'action':'camera_stop'}).status_code==409
        assert client.put('/api/capture/settings',json={'live_glim':True,'auto_process':True}).status_code==409
        assert client.post('/api/action',json={'action':'session_stop'}).status_code==200
        wait_state(client,'COMPLETE')


def test_archive_from_capture_imports_and_processes_elsewhere(root,tmp_path,monkeypatch):
    import shutil
    with TestClient(make_app(root,True)) as client:
        monkeypatch.setattr(client.app.state.service,'glim_available',lambda:False)
        sid=capture(client,'start_scan')['session'];wait_state(client,'SCANNING');time.sleep(.2)
        capture(client,'stop_scan');wait_state(client,'COMPLETE')
        source=root/'data/sessions'/sid/'raw_bag/MOCK_ONLY.txt';before=source.read_bytes()
        archive=client.get('/api/sessions/'+sid+'/project');assert archive.status_code==200
    target=tmp_path/'workstation';shutil.copytree(root/'config',target/'config');(target/'ui/frontend').mkdir(parents=True)
    with TestClient(make_app(target,True)) as client:
        result=client.post('/api/projects/import',content=archive.content,headers={'Content-Type':'application/zip'})
        assert result.status_code==200,result.text
        assert (target/'data/sessions'/sid/'raw_bag/MOCK_ONLY.txt').read_bytes()==before
        response=client.post('/api/action',json={'action':'process','session':sid});assert response.status_code==200,response.text
        client.portal.call(lambda:client.app.state.service.pm.items['offline']['watcher'])
        assert read_json(target/'data/sessions'/sid/'processing/run_001/job.json')['state']=='completed'
        assert source.read_bytes()==before


def test_camera_preference_can_change_after_capture(root):
    with TestClient(make_app(root,True)) as client:
        capture(client,'start_camera_recording');wait_state(client,'SCANNING')
        capture(client,'stop_camera_recording');wait_state(client,'COMPLETE')
        assert client.app.state.service.pm.active('camera')
        assert client.post('/api/camera/enabled',json={'enabled':False}).status_code==200
        assert not client.app.state.service.pm.active('camera')


def test_wizard_rejects_busy_job_before_starting_sensors(root,monkeypatch):
    with TestClient(make_app(root,True)) as client:
        s=client.app.state.service
        async def forbidden():raise AssertionError('Must reject before touching hardware')
        monkeypatch.setattr(s,'prepare_recording',forbidden)
        # Unknown dataset is rejected before sensor preparation, too.
        response=client.post('/api/calibrations/cal_1234567890123456/wizard',json={'action':'capture'})
        assert response.status_code==409
        sid=client.post('/api/sessions',json={'name':'legacy'}).json()['id']
        # Use low-level recording to verify compatibility mutual exclusion.
        monkeypatch.undo()
        assert client.post('/api/action',json={'action':'record_start','session':sid}).status_code==200
        monkeypatch.setattr(s,'prepare_recording',forbidden)
        response=client.post('/api/calibrations/cal_1234567890123456/wizard',json={'action':'capture'})
        assert response.status_code==409
        assert s.pm.active('recording')
        assert client.post('/api/projects/import',content=b'invalid',headers={'Content-Type':'application/zip'}).status_code==409
        assert not list((root/'.state').glob('project-import*'))


def test_incomplete_topic_recording_is_not_marked_successful(root):
    import yaml
    with TestClient(make_app(root,True)) as client:
        s=client.app.state.service
        client.put('/api/capture/settings',json={'live_glim':False,'auto_process':False})
        sid=capture(client,'start_scan')['session'];wait_state(client,'SCANNING')
        # Switch only the metadata validation path, after the mock recorder stops.
        client.portal.call(s.pm.stop,'recording',2)
        bag=root/'data/sessions'/sid/'raw_bag'
        (bag/'metadata.yaml').write_text(yaml.safe_dump({'rosbag2_bagfile_information':{
            'duration':{'nanoseconds':1000000000},'topics_with_message_count':[
            {'topic_metadata':{'name':'/livox/lidar'},'message_count':10}]}}))
        s.mock=False
        client.portal.call(s.stop_recording)
        s.mock=True
        m=read_json(bag.parent/'metadata.json')
        assert m['state']=='failed' and any('imu' in e for e in m['recording_errors'])


def test_startup_preview_starts_camera_even_if_lidar_fails(root,monkeypatch):
    from factory_mapping.service import Service
    service=Service(root,False)
    starts=[]
    monkeypatch.setattr(service.pm,'active',lambda key:False)
    async def fail_lidar():
        starts.append('lidar')
        raise RuntimeError('sensor unavailable')
    async def start_camera(force=False):starts.append(('camera',force))
    monkeypatch.setattr(service,'start_driver',fail_lidar)
    monkeypatch.setattr(service,'start_camera',start_camera)
    asyncio.run(service.start_preview_sensors())
    assert starts==['lidar',('camera',True)]
    assert any('LiDAR preview unavailable at startup' in error for error in service.errors)
    assert service.sessions.list()==[] and service.active is None


def test_app_startup_invokes_preview_without_creating_session(root,monkeypatch):
    from factory_mapping.service import Service
    started=[]
    async def preview_startup(self):started.append(True)
    monkeypatch.setattr(Service,'start_preview_sensors',preview_startup)
    with TestClient(make_app(root,True)) as client:
        assert started==[True]
        assert client.get('/api/sessions').json()==[]
        assert client.app.state.service.active is None


def test_rgb_disabled_warns_without_blocking_scan(root):
    with TestClient(make_app(root,True)) as client:
        client.put('/api/capture/settings',json={'live_glim':False,'auto_process':False})
        started=capture(client,'start_scan')
        scanning=wait_state(client,'SCANNING')
        assert scanning['error'] is None
        assert any('RGB recording is disabled' in warning for warning in scanning['warnings'])
        time.sleep(.2)
        capture(client,'stop_scan')
        assert wait_state(client,'COMPLETE')['outcome']=='recorded'
        metadata=read_json(root/'data/sessions'/started['session']/'metadata.json')
        assert set(metadata['topics'])=={'points_topic','imu_topic'}


def test_default_rgb_scan_records_camera_and_lidar(root):
    from conftest import ROOT
    import shutil
    shutil.copyfile(ROOT/'config/system.yaml',root/'config/system.yaml')
    with TestClient(make_app(root,True)) as client:
        assert client.get('/api/status').json()['config']['system']['camera']['enabled'] is True
        client.put('/api/capture/settings',json={'live_glim':False,'auto_process':False})
        started=capture(client,'start_scan')
        scanning=wait_state(client,'SCANNING')
        assert not any('RGB recording is disabled' in warning for warning in scanning['warnings'])
        time.sleep(.2)
        capture(client,'stop_scan')
        assert wait_state(client,'COMPLETE')['outcome']=='recorded'
        metadata=read_json(root/'data/sessions'/started['session']/'metadata.json')
        assert set(metadata['topics'])=={'points_topic','imu_topic','image_topic','camera_info_topic'}


def test_dense_preset_persists_and_drives_automatic_processing(root):
    with TestClient(make_app(root,True)) as client:
        response=client.put('/api/capture/settings',json={'live_glim':False,'auto_process':True,'mapping_preset':'pc_dense'})
        assert response.status_code==200 and response.json()['preset']=='pc_dense'
        # Older clients updating only switches must preserve the selected preset.
        response=client.put('/api/capture/settings',json={'live_glim':False,'auto_process':True})
        assert response.json()['preset']=='pc_dense'
        started=capture(client,'start_scan');wait_state(client,'SCANNING');time.sleep(.2)
        capture(client,'stop_scan');wait_state(client,'COMPLETE')
        job=root/'data/sessions'/started['session']/'processing/run_001'
        assert read_json(job/'job.json')['preset']=='pc_dense'
        config=read_json(job/'config/config.json')['global']
        assert config['config_odometry']=='config_odometry_cpu.json'
        assert read_json(job/'config/config_sub_mapping_passthrough.json')['sub_mapping']['min_dist_in_voxel']==.05
    with TestClient(make_app(root,True)) as client:
        assert client.get('/api/capture').json()['capabilities']['preset']=='pc_dense'


def test_dense_cpu_preset_does_not_require_cuda(root,monkeypatch):
    from factory_mapping.service import Service
    service=Service(root,False)
    monkeypatch.setattr(service,'glim_available',lambda:True)
    service.check_preset('pc_dense')
    with pytest.raises(ValueError,match='CUDA'):service.check_preset('offline_quality')


def test_invalid_mapping_preset_rejected(root):
    with TestClient(make_app(root,True)) as client:
        response=client.put('/api/capture/settings',json={'live_glim':False,'auto_process':True,'mapping_preset':'unknown'})
        assert response.status_code==422
