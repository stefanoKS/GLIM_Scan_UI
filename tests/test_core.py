import asyncio, json, os, signal, socket, sys, time
from types import SimpleNamespace
import numpy as np
import pytest
from factory_mapping.config import load, validate_sensor, wired_host_ip
from factory_mapping.storage import Sessions, read_json
from factory_mapping.processes import ProcessManager
from factory_mapping.preview import encode, decode
from factory_mapping.health import Rates, system_status
from factory_mapping import commands

def test_configuration(root):
    c=load(root); assert c['sensor']['points_topic']=='/livox/lidar'
    c['sensor']['lidar_ip']='$(reboot)'
    with pytest.raises(ValueError): validate_sensor(c['sensor'])

def test_wired_host_ip_uses_matching_subnet(monkeypatch):
    addresses=[SimpleNamespace(family=socket.AF_INET,address='10.0.0.2',netmask='255.255.255.0'),
               SimpleNamespace(family=socket.AF_INET,address='192.168.1.240',netmask='255.255.255.0')]
    monkeypatch.setattr('factory_mapping.config.psutil.net_if_addrs',lambda:{'enP8p1s0':addresses})
    sensor={'interface':'enP8p1s0','lidar_ip':'192.168.1.120'}
    assert wired_host_ip(sensor)=='192.168.1.240'
    sensor['interface']='missing'
    assert wired_host_ip(sensor) is None

def test_session_snapshot_and_raw_protection(root):
    s=Sessions(root); a=s.create('factory / zone','hello',load(root)); b=s.create('factory / zone','other',load(root))
    assert a['id']!=b['id']; p=s.get(a['id']); assert not (p/'raw_bag').exists()
    assert read_json(p/'metadata.json')['notes']=='hello'
    with pytest.raises(ValueError): s.get('../')
    with pytest.raises(ValueError): Sessions(root,True).get(a['id'])
    before=(p/'config_snapshot/system.yaml').read_text(); (root/'config/system.yaml').write_text('changed')
    assert (p/'config_snapshot/system.yaml').read_text()==before

def test_preview_bounds_and_finite():
    rng=np.random.default_rng(1);p=rng.normal(size=(100000,4));p[0,0]=np.nan
    packet=encode(p,123.4,1000,.1);v,t=decode(packet)
    assert 0<len(v)<=1000 and np.isfinite(v).all() and t==123.4
    with pytest.raises(ValueError): decode(packet[:-1])

def test_system(root):
    s=system_status(root); assert s['disk_free']>0 and s['ram_available']>0 and isinstance(s['interfaces'],dict)

def test_message_health_not_process_presence():
    r=Rates(); assert r.view()['hz']==0
    r.add(1);time.sleep(.01);r.add(2); assert r.view()['hz']>0
    r.last=time.monotonic()-3; assert r.view()['hz']==0

def test_process_crash_and_restart(root):
    async def go():
        pm=ProcessManager(root/'.state')
        await pm.start('x',[sys.executable,'-c','raise SystemExit(7)'],root/'x.log')
        await pm.items['x']['watcher'];assert pm.items['x']['state']=='failed'
        await pm.start('x',[sys.executable,'-c','print("ok")'],root/'x.log')
        await pm.items['x']['watcher']; assert pm.items['x']['state']=='completed'
    asyncio.run(go())

def test_process_cancel_and_duplicate(root):
    async def go():
        pm=ProcessManager(root/'.state')
        await pm.start('x',[sys.executable,'-c','import time; time.sleep(60)'],root/'x.log')
        with pytest.raises(ValueError): await pm.start('x',['false'],root/'x.log')
        await pm.stop('x',2,cancel=True);assert pm.items['x']['state']=='cancelled'
        assert not (root/'.state/process_x.json').exists()
    asyncio.run(go())

def test_stale_pid_removed_without_signalling(root):
    (root/'.state/process_x.json').write_text(json.dumps({'key':'x','pid':os.getpid(),'created':0}))
    pm=ProcessManager(root/'.state');assert not pm.active('x');assert not (root/'.state/process_x.json').exists()

def test_commands_have_real_upstream_arguments(root):
    c=load(root);s=Sessions(root);m=s.create('test','',c);p=s.get(m['id'])
    args=commands.record(p,c);assert '/livox/lidar' in args and '/livox/imu' in args and '/image' not in args
    cfg=commands.preset_snapshot(root,p,'jetson_cpu',p/'processing/config')
    args=commands.glim(cfg,p/'glim_dump',p/'raw_bag');assert args[:4]==['ros2','run','glim_ros','glim_rosbag'];assert 'auto_quit:=true' in args
    assert '--export_path' in commands.export(p/'glim_dump',p/'exports/a.ply',cfg)


def test_session_browser_ignores_non_directories(root):
    s=Sessions(root);(s.base/'.gitkeep').touch(); assert s.list()==[]

@pytest.mark.parametrize('cancel', [False, True])
def test_parent_exit_does_not_abandon_child(root, cancel):
    from factory_mapping.processes import group_alive
    async def go():
        pm=ProcessManager(root/'.state')
        child="import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); time.sleep(60)"
        parent="import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); time.sleep(.3)" % child
        await pm.start('tree',[sys.executable,'-c',parent],root/'tree.log')
        await asyncio.sleep(.5)
        pid=pm.items['tree']['pid'];assert group_alive(pid);assert pm.active('tree')
        await pm.stop('tree',.2,cancel=cancel)
        assert not group_alive(pid) and pm.items['tree']['forced']
        assert pm.items['tree']['state']==('cancelled' if cancel else 'failed')
    asyncio.run(go())
