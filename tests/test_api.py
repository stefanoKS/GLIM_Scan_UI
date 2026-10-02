import time
from fastapi.testclient import TestClient
from factory_mapping.api import make_app

def test_unauthenticated_api_origin_guard_and_session_escape(root):
    app=make_app(root,True)
    with TestClient(app) as c:
        assert c.get('/api/status').json()['mock']
        assert not (root/'.state/operator.token').exists()
        assert c.post('/api/action',json={'action':'rm -rf /'}).status_code==409
        assert c.post('/api/action',json={'action':'process','session':'../'}).status_code==409
        assert c.post('/api/action',json={'action':'driver_start'},headers={'Origin':'http://evil.test'}).status_code==403

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
