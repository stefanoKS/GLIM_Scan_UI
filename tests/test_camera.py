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
    c=enable(root)
    with pytest.raises(ValueError,match='hardware-validated'):commands.camera(root,c)
    c['camera'].update(pipeline_validated=True,gstreamer_pipeline='test-source ! converter')
    args=commands.camera(root,c)
    assert args[:4]==['ros2','run','gscam2','gscam_main']
    params=yaml.safe_load((root/'.state/camera.params.yaml').read_text())['/**']['ros__parameters']
    assert params['use_gst_timestamps'] is True and params['camera_info_url']==''
