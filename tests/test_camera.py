import copy
import json
import time
import asyncio
import pytest
import yaml
from factory_mapping.config import load
from factory_mapping.camera_config import validate_camera
from factory_mapping import commands


def enable(root):
    path=root/'config/system.yaml'; s=yaml.safe_load(path.read_text()); s['camera']['enabled']=True;path.write_text(yaml.safe_dump(s));return load(root)


def test_disabled_camera_needs_no_calibration_or_hardware(root):
    (root/'config/calibration/camera_intrinsics.yaml').unlink()
    assert not load(root)['system']['camera']['enabled']
    (root/'config/camera/dfk33ux287.yaml').unlink()
    assert 'camera' not in load(root)


def test_camera_configuration(root):
    c=enable(root)
    assert c['camera']['image_topic']=='/camera/image_raw'


@pytest.mark.parametrize('key,value',[('image_topic','/bad;touch'),('camera_info_topic','/camera/image_raw'),('width',0),('height',True),('fps',-1),('fps',float('nan')),('time_offset_sec',float('inf')),('frame_id','bad frame'),('preview_hz',20),('driver','shell'),('intrinsics_file','../other')])
def test_bad_camera_config(root,key,value):
    c=load(root)['camera'];c[key]=value
    with pytest.raises(ValueError):validate_camera(c)


def test_record_topics_and_snapshot(root):
    from factory_mapping.storage import Sessions,read_json
    c=load(root);assert list(commands.acquisition_topics(c).values())==['/livox/lidar','/livox/imu']
    c=enable(root);s=Sessions(root);m=s.create('camera','',c);p=s.get(m['id'])
    args=commands.record(p,c)
    assert '/camera/image_raw' in args and '/camera/camera_info' in args
    qos=yaml.safe_load((p/'config_snapshot/record_qos.yaml').read_text())
    assert qos['/camera/image_raw']['reliability']=='reliable'
    assert qos['/livox/lidar']['reliability']=='best_effort'
    assert m['camera']['enabled'] and m['camera']['frame_id']=='camera_optical_frame'
    before=(p/'config_snapshot/camera/dfk33ux287.yaml').read_bytes()
    (root/'config/camera/dfk33ux287.yaml').write_text('changed')
    assert (p/'config_snapshot/camera/dfk33ux287.yaml').read_bytes()==before


def test_camera_lifecycle_and_preview(root):
    from fastapi.testclient import TestClient
    from factory_mapping.api import make_app
    enable(root)
    with TestClient(make_app(root,True)) as client:
        assert client.post('/api/action',json={'action':'camera_start','pipeline':'$(touch /tmp/evil)'}).status_code==422
        assert client.post('/api/action',json={'action':'camera_start'}).status_code==200
        s=client.get('/api/status').json();c=s['health']['camera']
        assert c['camera_running'] and c['healthy'] and not c['camera_info_valid']
        assert client.get('/api/camera/preview').headers['content-type']=='image/jpeg'
        assert client.post('/api/action',json={'action':'camera_stop'}).status_code==200
        assert client.get('/api/status').json()['health']['camera']['state']=='stopped'
        assert client.get('/api/camera/preview').status_code==404


def test_camera_metrics_reject_zero_info(root):
    from factory_mapping.camera import CameraMetrics
    c=load(root)['camera'];m=CameraMetrics(c)
    m.info(c['width'],c['height'],[0]*9,[0]*5,'plumb_bob',c['frame_id'])
    m.image(time.time(),c['width'],c['height'],c['frame_id'])
    assert m.view()['camera_info_seen'] and not m.view()['camera_info_valid']
    json.dumps(m.view(),allow_nan=False)


def test_camera_requires_validated_static_pipeline(root):
    c=enable(root);c['camera']['pipeline_validated']=False
    with pytest.raises(ValueError,match='hardware-validated'):commands.camera(root,c)
    c['camera'].update(pipeline_validated=True,gstreamer_pipeline='test-source ! converter')
    args=commands.camera(root,c)
    assert args[:4]==['ros2','run','gscam2','gscam_main']
    params=yaml.safe_load((root/'.state/camera.params.yaml').read_text())['/**']['ros__parameters']
    assert params['use_gst_timestamps'] is True and params['camera_info_url']==''


def test_required_camera_blocks_mapping_but_optional_does_not(root):
    from factory_mapping.service import Service
    async def go():
        c=enable(root);s=Service(root,True)
        try:
            await s.start_driver();s.require_health()
            s.config['camera']['required_for_mapping']=True
            with pytest.raises(ValueError,match='Required camera'):s.require_health()
            await s.start_camera();s.require_health()
            await s.stop_camera()
            assert s.pm.active('driver')
        finally:await s.close()
    asyncio.run(go())


def test_rgb_is_never_a_glim_input(root):
    from factory_mapping.storage import Sessions
    c=enable(root);sessions=Sessions(root);m=sessions.create('rgb','',c);p=sessions.get(m['id'])
    cfg=commands.preset_snapshot(root,p,'jetson_cpu',p/'processing/test-config')
    ros=json.loads((cfg/'config_ros.json').read_text())['glim_ros']
    assert ros['image_topic']!=c['camera']['image_topic']
    assert ros['points_topic']==c['sensor']['points_topic']
    assert ros['imu_topic']==c['sensor']['imu_topic']


def test_camera_stop_does_not_stop_live_glim(root):
    from factory_mapping.service import Service
    async def go():
        enable(root);s=Service(root,True)
        try:
            await s.start_driver();await s.start_camera()
            m=s.sessions.create('live','',s.config);await s.start_glim(m['id'],'jetson_cpu')
            await s.stop_camera()
            assert s.pm.active('glim') and s.pm.active('driver')
        finally:await s.close()
    asyncio.run(go())


def test_physical_detection_matches_model_and_serial(tmp_path):
    from factory_mapping.camera import detect_camera
    d=tmp_path/'6-1';d.mkdir()
    for k,v in {'idVendor':'199e','product':'DFK 33UX287','serial':'12345678','speed':'5000'}.items():(d/k).write_text(v)
    assert detect_camera({},tmp_path)['detected']
    assert not detect_camera({'serial_number':'another'},tmp_path)['detected']
    (d/'idVendor').write_text('8086')
    assert not detect_camera({},tmp_path)['detected']


def test_record_button_starts_both_and_saves_one_bag(root):
    from fastapi.testclient import TestClient
    from factory_mapping.api import make_app
    from factory_mapping.storage import read_json
    with TestClient(make_app(root,True)) as client:
        assert client.post('/api/camera/enabled',json={'enabled':True}).status_code==200
        m=client.post('/api/sessions',json={'name':'combined'}).json()
        assert client.post('/api/action',json={'action':'record_start','session':m['id']}).status_code==200
        s=client.get('/api/status').json()
        assert all(s['processes'][k]['state']=='running' for k in ('camera','driver','recording'))
        assert not s['detection']['camera']['detected'] # mock never claims physical hardware
        assert client.post('/api/camera/enabled',json={'enabled':False}).status_code==409
        assert client.post('/api/action',json={'action':'session_stop'}).status_code==200
        meta=read_json(root/'data/sessions'/m['id']/'metadata.json',{})
        assert set(meta['topics'].values())=={'/livox/lidar','/livox/imu','/camera/image_raw','/camera/camera_info'}
    assert load(root)['system']['camera']['enabled']


def test_camera_start_failure_prevents_incomplete_combined_bag(root):
    from factory_mapping.service import Service
    async def go():
        enable(root);s=Service(root,True)
        async def failed():raise ValueError('Camera unavailable')
        s.start_camera=failed
        try:
            m=s.sessions.create('missing camera','',s.config)
            with pytest.raises(ValueError,match='Camera unavailable'):await s.start_recording(m['id'])
            assert not (s.sessions.get(m['id'])/'raw_bag').exists()
            assert not s.pm.active('recording')
        finally:await s.close()
    asyncio.run(go())
