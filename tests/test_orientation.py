import socket
from types import SimpleNamespace as N
import numpy as np
import pytest
from factory_mapping.config import wired_connection
from factory_mapping.orientation import GravityWindow,leveling_quaternion


def test_auto_ethernet_excludes_wifi_virtual_down_and_ambiguous(tmp_path,monkeypatch):
    addresses={}
    stats={}
    def device(name,ip='192.168.1.135',physical=True,wifi=False,up=True):
        node=tmp_path/name;node.mkdir();(node/'type').write_text('1')
        if physical:(node/'device').mkdir()
        if wifi:(node/'wireless').mkdir()
        addresses[name]=[N(family=socket.AF_INET,address=ip,netmask='255.255.255.0')]
        stats[name]=N(isup=up)
    device('eth_jetson');device('wifi',wifi=True);device('bridge',physical=False);device('down',up=False);device('other','10.0.0.2')
    monkeypatch.setattr('factory_mapping.config.psutil.net_if_addrs',lambda:addresses)
    monkeypatch.setattr('factory_mapping.config.psutil.net_if_stats',lambda:stats)
    sensor={'interface':'auto','lidar_ip':'192.168.1.120'}
    assert wired_connection(sensor,tmp_path)==('eth_jetson','192.168.1.135')
    device('second','192.168.1.136')
    assert wired_connection(sensor,tmp_path)==('auto',None)
    sensor['interface_setting']='second'
    assert wired_connection(sensor,tmp_path)==('second','192.168.1.136')
    sensor['interface_setting']='auto';addresses.pop('second');addresses.pop('eth_jetson')
    assert wired_connection(sensor,tmp_path)==('auto',None)


@pytest.mark.parametrize('a',[[0,0,1],[1,0,1],[0,0,-1],[.2,-.3,.8]])
def test_rotation_levels_display_only(a):
    original=list(a);q=np.array(leveling_quaternion(a,[0,0,0,1]));v=np.array(a,dtype=float);v/=np.linalg.norm(v)
    rotated=v+2*np.cross(q[:3],np.cross(q[:3],v)+q[3]*v)
    assert np.allclose(rotated,[0,0,1]) and a==original
    assert np.isclose(np.linalg.norm(q),1)


def test_gravity_rejects_motion_and_stale_data():
    g=GravityWindow()
    for i in range(401):g.add(i*.005,[.5,0,.866],[0,0,0])
    assert g.view(2)['stable']
    assert g.view(3)['stable']
    assert not g.view(3.6)['stable']
    assert not g.view(5)['stable']
    g.add(2.001,[.5,0,.866],[0,0,.5])
    assert not g.view(2.001)['stable']


def test_orientation_api_persists_reset_and_keeps_config(root,monkeypatch):
    import asyncio,time
    from fastapi.testclient import TestClient
    from factory_mapping.api import make_app
    with TestClient(make_app(root,True)) as c:
        s=c.app.state.service
        before=(root/'config/livox/mid360.yaml').read_bytes()
        assert c.post('/api/preview/orientation').status_code==409 # no fabricated mock gravity
        s.mock=False
        monkeypatch.setattr(s.pm,'active',lambda k:k=='driver')
        original_sleep=asyncio.sleep
        async def no_measurement_wait(delay):
            assert delay!=.25,'Orientation must not wait for new measurements'
            await original_sleep(delay)
        monkeypatch.setattr('factory_mapping.api.asyncio.sleep',no_measurement_wait)
        monkeypatch.setattr(s,'health',lambda:{'gravity':{'stable':True,'first':time.time()-2,'last':time.time()-1,'acceleration':[1,0,1]}})
        result=c.post('/api/preview/orientation');assert result.status_code==200,result.text
        assert c.get('/api/preview/orientation').json()['quaternion']==result.json()['quaternion']
        assert result.json()['scope']=='live_preview_only'
        assert c.delete('/api/preview/orientation').json()['quaternion']==[0,0,0,1]
        assert (root/'config/livox/mid360.yaml').read_bytes()==before
        s.mock=True


@pytest.mark.parametrize('stable,age',[(False,0),(True,2),(True,-1)])
def test_orientation_rejects_unstable_or_stale_data(root,monkeypatch,stable,age):
    import time
    from fastapi.testclient import TestClient
    from factory_mapping.api import make_app
    with TestClient(make_app(root,True)) as client:
        service=client.app.state.service
        service.mock=False
        monkeypatch.setattr(service.pm,'active',lambda key:key=='driver')
        monkeypatch.setattr(service,'health',lambda:{'gravity':{'stable':stable,'last':time.time()-age,'acceleration':[1,0,1]}})
        result=client.post('/api/preview/orientation')
        assert result.status_code==409
        assert 'Fresh, stable IMU data is unavailable' in result.json()['detail']
        assert not (root/'.state/view_orientation.json').exists()
        service.mock=True
