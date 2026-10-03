import copy
import json
import time
import asyncio
import pytest
import yaml
import numpy as np
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


def test_camera_parameters_preserve_spaces_in_calibration_path(root,tmp_path):
    import shutil
    from factory_mapping.commands import camera
    c=enable(root)
    spaced_root=tmp_path/'factory mapping'
    shutil.copytree(root/'config',spaced_root/'config')
    (spaced_root/'.state').mkdir()
    matrix=lambda rows,cols,data:dict(rows=rows,cols=cols,data=data)
    calibration=dict(image_width=720,image_height=540,camera_name='factory_rgb',distortion_model='plumb_bob',
                    camera_matrix=matrix(3,3,[600,0,360,0,600,270,0,0,1]),
                    distortion_coefficients=matrix(1,5,[0,0,0,0,0]),
                    rectification_matrix=matrix(3,3,[1,0,0,0,1,0,0,0,1]),
                    projection_matrix=matrix(3,4,[600,0,360,0,0,600,270,0,0,0,1,0]))
    (spaced_root/'config/calibration/camera_intrinsics.yaml').write_text(yaml.safe_dump(calibration))
    argv=camera(spaced_root,c)
    params=yaml.safe_load((spaced_root/'.state/camera.params.yaml').read_text())['/**']['ros__parameters']
    assert params['camera_info_url']=='file://'+str((spaced_root/c['camera']['intrinsics_file']).resolve())
    assert '%20' not in params['camera_info_url']
    assert argv[0:4]==['ros2','run','gscam2','gscam_main']


def test_charuco_board_capture_and_solver(root):
    cv2=pytest.importorskip("cv2",reason="Optional camera stack is not installed")
    from factory_mapping.camera_intrinsics import board, add_view, calibrate, detect, printable_board
    image=cv2.imdecode(np.frombuffer(printable_board(),np.uint8),cv2.IMREAD_COLOR)
    if hasattr(cv2.aruco,'CharucoDetector'):
        assert len(detect(image)[1])>100
        views=[]
        add_view(views,image,1)
        with pytest.raises(ValueError,match='new camera frame'):add_view(views,image,1)
        with pytest.raises(ValueError,match='Move or rotate'):add_view(views,image,2)
    target=board();points=target.getChessboardCorners() if hasattr(target,'getChessboardCorners') else target.chessboardCorners
    matrix=np.array([[630.,0,360.],[0,635.,270.],[0,0,1.]])
    samples=[]
    for index in range(9):
        projected,_=cv2.projectPoints(points,np.array([.08*index,.04*(index%3),-.03*index]),
                                      np.array([-.3+.015*index,-.2+.013*index,1.1+.04*index]),matrix,np.zeros(5))
        samples.append(dict(ids=np.arange(len(points)),corners=projected.reshape(-1,2).astype(np.float32),size=(720,540)))
    with pytest.raises(ValueError,match='at least 4'):calibrate(samples[:3],load(root)['camera'])
    measured,quality=calibrate(samples[:4],load(root)['camera'])
    assert quality['views']==4
    assert quality['rms']<.01
    assert abs(measured['camera_matrix']['data'][0]-630)<1


@pytest.mark.parametrize('rms',[12.01,float('nan'),float('inf')])
def test_intrinsic_solver_rejects_bad_quality(root,monkeypatch,rms):
    cv2=pytest.importorskip('cv2',reason='Optional camera stack is not installed')
    from factory_mapping.camera_intrinsics import calibrate
    samples=[dict(ids=np.arange(12),corners=np.zeros((12,2),np.float32),size=(720,540)) for index in range(4)]
    monkeypatch.setattr(cv2,'calibrateCameraExtended',lambda *args:(rms,np.eye(3),np.zeros((1,5))))
    with pytest.raises(ValueError,match='Reset views and recapture at least 4 sharp, distinct views'):
        calibrate(samples,load(root)['camera'])


def test_intrinsic_solver_accepts_rms_below_new_limit(root,monkeypatch):
    cv2=pytest.importorskip('cv2',reason='Optional camera stack is not installed')
    from factory_mapping.camera_intrinsics import calibrate
    samples=[dict(ids=np.arange(12),corners=np.zeros((12,2),np.float32),size=(720,540)) for index in range(4)]
    matrix=np.array([[630.,0,360.],[0,635.,270.],[0,0,1.]])
    monkeypatch.setattr(cv2,'calibrateCameraExtended',lambda *args:(11.99,matrix,np.zeros((1,5))))
    _,quality=calibrate(samples,load(root)['camera'])
    assert quality['rms']==11.99


def test_intrinsic_browser_capture_save_and_delete(root,monkeypatch):
    cv2=pytest.importorskip("cv2",reason="Optional camera stack is not installed")
    from factory_mapping.camera_intrinsics import board
    from fastapi.testclient import TestClient
    from factory_mapping.api import make_app
    from factory_mapping.calibration_data import intrinsics_status
    enable(root)
    with TestClient(make_app(root,True)) as client:
        assert client.get('/api/camera/intrinsics/board').headers['content-type']=='image/png'
        progress=client.get('/api/camera/intrinsics/views').json()
        assert progress['views']==0 and progress['required']==4
        assert client.post('/api/camera/intrinsics/views').status_code==409
        assert client.post('/api/action',json={'action':'camera_start'}).status_code==200
        service=client.app.state.service
        original_active=service.pm.active
        monkeypatch.setattr(service.pm,'active',lambda key:key=='camera_preview' or original_active(key))
        image=board().generateImage((690,460),marginSize=0) if hasattr(board(),'generateImage') else board().draw((690,460),marginSize=0)
        image=cv2.copyMakeBorder(image,40,40,15,15,cv2.BORDER_CONSTANT,value=255)
        success,encoded=cv2.imencode('.jpg',image)
        assert success
        (root/'.state/camera_calibration_frame.jpg').write_bytes(encoded.tobytes())
        if hasattr(cv2.aruco,'CharucoDetector'):
            captured=client.post('/api/camera/intrinsics/views')
            assert captured.status_code==200,captured.text
            assert captured.json()['corners']>=12
            assert client.post('/api/camera/intrinsics/views').status_code==409
        assert client.delete('/api/camera/intrinsics/views').json()['views']==0
        target=board();points=target.getChessboardCorners() if hasattr(target,'getChessboardCorners') else target.chessboardCorners;matrix=np.array([[630.,0,360.],[0,635.,270.],[0,0,1.]])
        for index in range(4):
            projected,_=cv2.projectPoints(points,np.array([.08*index,.04*(index%3),-.03*index]),
                                          np.array([-.3+.015*index,-.2+.013*index,1.1+.04*index]),matrix,np.zeros(5))
            service.intrinsic_samples.append(dict(ids=np.arange(len(points)),corners=projected.reshape(-1,2).astype(np.float32),size=(720,540)))
        previous_pid=service.pm.items['camera']['pid']
        result=client.post('/api/camera/intrinsics/calibrate')
        assert result.status_code==200,result.text
        assert result.json()['intrinsics']['status']=='VALID'
        assert result.json()['quality']['rms']<.01
        assert service.pm.items['camera']['pid']!=previous_pid
        assert client.get('/api/camera/intrinsics/views').json()['views']==0
        assert intrinsics_status(root,service.config['camera'])['status']=='VALID'
        session=client.post('/api/sessions',json={'name':'busy'}).json()
        assert client.post('/api/action',json={'action':'record_start','session':session['id']}).status_code==200
        assert client.delete('/api/camera/intrinsics').status_code==409
        assert client.post('/api/action',json={'action':'session_stop'}).status_code==200
        deleted=client.delete('/api/camera/intrinsics')
        assert deleted.status_code==200,deleted.text
        assert deleted.json()['intrinsics']['status']=='MISSING'
        assert list((root/'config/calibration/history').glob('*camera_intrinsics.yaml'))
        assert service.pm.active('camera')


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
    m.info(c['width'],c['height'],np.array([800,0,c['width']/2,0,800,c['height']/2,0,0,1],dtype=np.float64),[0.]*5,'plumb_bob',c['frame_id'])
    assert m.view()['camera_info_valid']
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
