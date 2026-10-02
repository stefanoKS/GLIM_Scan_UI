import time
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

def test_binary_websocket(root):
    from factory_mapping.preview import decode
    app=make_app(root,True)
    with TestClient(app) as c:
        c.post('/api/action',json={'action':'driver_start'})
        with c.websocket_connect('/ws/preview') as ws:
            points,stamp=decode(ws.receive_bytes());assert len(points)>100 and stamp>0
