import asyncio
import os
import signal
import subprocess
import sys
import time

import psutil
import pytest
from fastapi.testclient import TestClient

from factory_mapping.api import make_app
from factory_mapping.processes import ProcessManager, group_alive
from factory_mapping.storage import atomic_json, read_json


@pytest.fixture
def abandoned(root):
    processes=[]
    def launch(code='import time; time.sleep(60)',key='driver'):
        process=subprocess.Popen([sys.executable,'-c',code],start_new_session=True)
        processes.append(process)
        atomic_json(root/f'.state/process_{key}.json',dict(key=key,pid=process.pid,
                    created=psutil.Process(process.pid).create_time(),forced=False,
                    argv=[sys.executable,'-c',code],state='running'))
        return process
    yield launch
    for process in processes:
        try: os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait(timeout=5)


@pytest.mark.parametrize('stubborn',[False,True])
def test_orphan_recovery_graceful_and_forced(root,abandoned,stubborn):
    ready=root/'ready'
    code=('import signal,time,pathlib; '
          +('signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); ' if stubborn else '')
          +f'pathlib.Path({str(ready)!r}).touch(); time.sleep(60)')
    process=abandoned(code)
    for _ in range(100):
        if ready.exists(): break
        time.sleep(.02)
    assert ready.exists()
    pm=ProcessManager(root/'.state')
    assert pm.items['driver']['state']=='orphaned'
    asyncio.run(pm.recover('driver',timeout=.1,terminate_timeout=.1))
    assert not group_alive(process.pid) and not pm.active('driver')
    assert pm.items['driver']['forced']==stubborn
    assert not (root/'.state/process_driver.json').exists()
    asyncio.run(pm.recover('driver'))  # Repeated stop is harmless.


def test_recovery_rejects_changed_identity(root,abandoned,monkeypatch):
    process=abandoned()
    pm=ProcessManager(root/'.state')
    pm.items['driver']['created']-=100
    signals=[]
    with monkeypatch.context() as patch:
        patch.setattr(os,'killpg',lambda *args:signals.append(args))
        with pytest.raises(ValueError,match='Cannot verify ownership'):
            asyncio.run(pm.recover('driver',timeout=.1))
    assert signals==[] and group_alive(process.pid)
    assert pm.active('driver') and (root/'.state/process_driver.json').exists()


def test_recovery_keeps_verified_children_after_launcher_exits(root,abandoned):
    ready=root/'child_ready'
    child=('import signal,time,pathlib; signal.signal(signal.SIGINT,signal.SIG_IGN); '
           'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
           f'pathlib.Path({str(ready)!r}).touch(); time.sleep(60)')
    parent=f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child!r}]); time.sleep(60)'
    process=abandoned(parent)
    for _ in range(100):
        if ready.exists(): break
        time.sleep(.02)
    assert ready.exists()
    pm=ProcessManager(root/'.state')
    asyncio.run(pm.recover('driver',timeout=.1,terminate_timeout=.1))
    assert not group_alive(process.pid) and pm.items['driver']['forced']


@pytest.mark.parametrize('restart',[False,True])
def test_recovery_api_unblocks_next_scan(root,abandoned,restart):
    process=abandoned()
    with TestClient(make_app(root,True)) as client:
        service=client.app.state.service
        assert not client.get('/api/capture').json()['can_start']
        response=client.post('/api/action',json={'action':'processes_restart' if restart else 'processes_stop'})
        assert response.status_code==200,response.text
        assert not group_alive(process.pid)
        assert service.pm.active('driver')==restart and service.pm.active('camera')==restart
        assert client.get('/api/capture').json()['can_start']
        assert not any('still alive' in error for error in service.errors)
        assert not service.pm.active('recording') and not service.sessions.list()


def test_recovery_finalizes_current_scan_without_auto_processing(root):
    with TestClient(make_app(root,True)) as client:
        service=client.app.state.service
        response=client.post('/api/capture/action',json={'action':'start_scan'})
        sid=response.json()['session']
        for _ in range(100):
            if client.get('/api/capture').json()['state']=='SCANNING': break
            time.sleep(.05)
        assert service.capture.data['state']=='SCANNING'
        time.sleep(.2)
        response=client.post('/api/action',json={'action':'processes_stop'})
        assert response.status_code==200,response.text
        metadata=read_json(root/f'data/sessions/{sid}/metadata.json')
        assert metadata['bag_finalized'] and metadata['state']=='recorded'
        assert (root/f'data/sessions/{sid}/raw_bag/MOCK_ONLY.txt').exists()
        assert not any(service.pm.active(key) for key in service.pm.items)
        assert 'offline' not in service.pm.items and service.capture.view()['can_start']


def test_recovery_failure_does_not_restart_sensors(root,abandoned,monkeypatch):
    abandoned()
    with TestClient(make_app(root,True)) as client:
        service=client.app.state.service
        service.pm.items['driver']['created']-=100
        async def forbidden(*args,**kwargs): raise AssertionError('Must not restart after failed stop')
        monkeypatch.setattr(service,'start_driver',forbidden)
        monkeypatch.setattr(service,'start_camera',forbidden)
        response=client.post('/api/action',json={'action':'processes_restart'})
        assert response.status_code==409 and 'Cannot verify ownership' in response.text
        assert not service.recovering and not service.capture.view()['can_start']


def test_partial_sensor_restart_reports_failure(root,monkeypatch):
    with TestClient(make_app(root,True)) as client:
        service=client.app.state.service
        async def fail(): raise ValueError('LiDAR unplugged')
        monkeypatch.setattr(service,'start_driver',fail)
        response=client.post('/api/action',json={'action':'processes_restart'})
        assert response.status_code==409 and 'restart was incomplete' in response.text
        assert service.pm.active('camera') and not service.recovering
