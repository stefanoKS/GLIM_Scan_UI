import re, time
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from factory_mapping.api import make_app

def test_unauthenticated_api_origin_guard_and_session_escape(root):
    app=make_app(root,True)
    with TestClient(app) as c:
        assert c.get('/api/status').json()['mock']
        assert not (root/'.state/operator.token').exists()
        assert c.post('/api/action',json={'action':'rm -rf /'}).status_code==409
        assert c.post('/api/action',json={'action':'process','session':'../'}).status_code==409
        assert c.post('/api/action',json={'action':'driver_start'},headers={'Origin':'http://evil.test'}).status_code==403
        with pytest.raises(WebSocketDisconnect) as rejected:
            with c.websocket_connect('/ws/preview',headers={'Origin':'http://evil.test'}): pass
        assert rejected.value.code==1008

def test_session_name_defaults_to_datetime(root):
    app=make_app(root,True)
    with TestClient(app) as client:
        for payload in ({}, {'name':'   '}):
            response=client.post('/api/sessions',json=payload)
            assert response.status_code==200,response.text
            session=response.json()
            assert re.fullmatch(r'\d{8}_\d{6}_session(?:_[0-9a-f]{6})?',session['id'])
            assert re.fullmatch(r'Session \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',session['name'])
        named=client.post('/api/sessions',json={'name':'Zone A'}).json()
        assert named['name']=='Zone A'
        assert named['id'].endswith('_Zone_A')

def test_project_transfer_http_and_active_guard(root,tmp_path):
    import shutil
    app=make_app(root,True)
    with TestClient(app) as client:
        session=client.post('/api/sessions',json={'name':'browser transfer'}).json()
        assert client.post('/api/action',json={'action':'record_start','session':session['id']}).status_code==200
        assert client.get('/api/sessions/'+session['id']+'/project').status_code==409
        time.sleep(.3)
        assert client.post('/api/action',json={'action':'session_stop'}).status_code==200
        archive=client.get('/api/sessions/'+session['id']+'/project')
        assert archive.status_code==200 and archive.headers['content-type']=='application/zip'
        assert client.post('/api/projects/import',content=archive.content,headers={'content-type':'application/zip'}).status_code==409
    target=tmp_path/'server';shutil.copytree(root/'config',target/'config')
    (target/'ui/frontend').mkdir(parents=True);(target/'ui/frontend/index.html').write_text('test')
    with TestClient(make_app(target,True)) as client:
        imported=client.post('/api/projects/import',content=archive.content,headers={'content-type':'application/zip'})
        assert imported.status_code==200,imported.text
        assert imported.json()['id']==session['id']
        assert (target/'data/sessions'/session['id']/'raw_bag/MOCK_ONLY.txt').exists()

def test_record_only_session_never_calls_glim(root, monkeypatch):
    app=make_app(root,True)
    with TestClient(app) as c:
        service=app.state.service
        def forbidden(*args,**kwargs): raise AssertionError('Record-only must not invoke GLIM or validate its preset')
        monkeypatch.setattr(service,'start_glim',forbidden)
        monkeypatch.setattr(service,'check_preset',forbidden)
        sid=c.post('/api/sessions',json={'name':'acquisition only'}).json()['id']
        assert not (root/'ros2_ws/install/glim_ros').exists()
        response=c.post('/api/action',json={'action':'session_record_start','session':sid,'preset':'jetson_gpu'})
        assert response.status_code==200, response.text
        status=c.get('/api/status').json()
        assert status['processes']['driver']['state']=='running'
        assert status['processes']['recording']['state']=='running'
        assert 'glim' not in status['processes']
        assert c.post('/api/action',json={'action':'session_record_start','session':sid}).status_code==409
        assert c.get('/api/status').json()['active_session']==sid
        bag=root/'data/sessions'/sid/'raw_bag/MOCK_ONLY.txt'
        log=bag.parent.parent/'logs/recording.log'
        for _ in range(30):
            if 'MOCK recording started' in log.read_text(): break
            time.sleep(.1)
        assert 'MOCK recording started' in log.read_text()
        assert c.post('/api/action',json={'action':'session_stop'}).status_code==200
        assert bag.exists()
        meta=c.get('/api/sessions').json()[0]
        assert meta['state']=='recorded' and meta['bag_finalized'] and meta['glim_live'] is False
        assert meta['end_time'] and meta['finalized_at']
        assert c.get('/api/status').json()['active_session'] is None

def test_missing_glim_rejected_before_recording_or_driver_start(root):
    app=make_app(root,False)
    with TestClient(app) as c:
        sid=c.post('/api/sessions',json={'name':'no glim installed'}).json()['id']
        assert c.get('/api/status').json()['glim_available'] is False
        response=c.post('/api/action',json={'action':'session_start','session':sid})
        assert response.status_code==409 and 'not installed' in response.json()['detail']
        assert c.get('/api/status').json()['processes']=={}

def test_mock_record_process_and_raw_protection(root):
    app=make_app(root,True)
    with TestClient(app) as c:
        sid=c.post('/api/sessions',json={'name':'mock flow'}).json()['id']
        def act(action,**kw):return c.post('/api/action',json={'action':action,'session':sid,**kw})
        assert act('driver_start').status_code==200
        assert act('record_start').status_code==200
        time.sleep(.4)
        assert act('session_stop').status_code==200
        assert (root/'data/sessions'/sid/'raw_bag/MOCK_ONLY.txt').exists()
        assert act('process').status_code==200
        assert act('process').status_code==409
        time.sleep(.4)
        assert act('cancel').status_code==200
        runs=c.get('/api/sessions').json()[0]['processing'];assert runs[0]['state']=='cancelled'
        assert act('delete_derived',run=runs[0]['id']).status_code==200
        assert (root/'data/sessions'/sid/'raw_bag/MOCK_ONLY.txt').exists()
        assert act('export',run='../../raw_bag').status_code==409

def test_host_ip_change_finalizes_recording_and_restarts_driver(root,monkeypatch):
    app=make_app(root,True)
    with TestClient(app) as c:
        service=app.state.service
        service.config['sensor']['host_ip']='192.168.1.135'
        sid=c.post('/api/sessions',json={'name':'network change'}).json()['id']
        assert c.post('/api/action',json={'action':'record_start','session':sid}).status_code==200
        old_pid=service.pm.items['driver']['pid']
        monkeypatch.setattr('factory_mapping.service.wired_host_ip',lambda sensor:'192.168.1.240')
        c.portal.call(service.refresh_host_ip)
        status=c.get('/api/status').json()
        assert status['config']['sensor']['host_ip']=='192.168.1.240'
        assert status['active_session'] is None
        assert status['processes']['driver']['pid']!=old_pid
        assert status['processes']['driver']['state']=='running'
        assert c.get('/api/sessions').json()[0]['bag_finalized']

def test_network_save_omits_detected_host_ip(root):
    app=make_app(root,True)
    with TestClient(app) as c:
        sensor=c.get('/api/status').json()['config']['sensor']
        body={key:value for key,value in sensor.items() if key not in ('host_ip','serial_number','frame_id','imu_frame_id','expected_imu_hz','T_lidar_imu')}
        response=c.put('/api/network',json=body)
        assert response.status_code==200,response.text
        assert 'host_ip' not in (root/'config/livox/mid360.yaml').read_text()

def test_binary_websocket(root):
    from factory_mapping.preview import decode
    app=make_app(root,True)
    with TestClient(app) as c:
        c.post('/api/action',json={'action':'driver_start'})
        with c.websocket_connect('/ws/preview') as ws:
            points,stamp=decode(ws.receive_bytes());assert len(points)>100 and stamp>0
