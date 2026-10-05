"""Hardware-independent selection, cleanup, snapshot and lock regressions."""
import asyncio
import json
import time
import pytest
from fastapi.testclient import TestClient
from factory_mapping.service import Service
from factory_mapping.config import load
from factory_mapping.storage import atomic_json, read_json
from factory_mapping import commands
from factory_mapping.api import make_app
from test_capture import capture, wait_state


@pytest.fixture
def rig(root, monkeypatch):
    s=Service(root,False)
    s.config['system']['camera'].update(profile='auto',enabled=True)
    s.camera_selection.startup_timeout=.025
    detected={'d405':True,'dfk33ux287':True}
    failures={}; calls=[]; stopped=[]
    def profile(c): return 'd405' if c['source']=='realsense' else 'dfk33ux287'
    monkeypatch.setattr('factory_mapping.camera_selection.detect_camera',lambda c:dict(detected=detected[profile(c)],devices=[]))
    async def deps(c,root):
        if failures.get(profile(c))=='dependency':raise ValueError('missing SDK')
    monkeypatch.setattr('factory_mapping.camera_selection.check_camera_dependencies',deps)
    async def start(key,args,log,out=None,done=None):
        p=profile(s.config['camera']);calls.append((p,key))
        if key=='camera':assert not any(s.pm.active(k) for k in ('camera','camera_monitor','camera_preview'))
        s.pm.items[key]={'state':'running'}
        if failures.get(p)=='exit' and key=='camera':s.pm.items[key]['state']='failed'
        return s.pm.items[key]
    async def stop(key,*args,**kwargs):
        stopped.append(key)
        if key in s.pm.items:s.pm.items[key]['state']='completed'
    monkeypatch.setattr(s,'start_process',start);monkeypatch.setattr(s.pm,'stop',stop)
    def health():
        c=s.config['camera'];failure=failures.get(profile(c))
        return dict(healthy=failure not in ('stream','exit','geometry','info'),image_hz=c['expected_hz'],
                    image_age=.01,width=c['width']+(1 if failure=='geometry' else 0),height=c['height'],
                    camera_info_seen=failure!='info',camera_info_age=.01,camera_info_valid=True)
    monkeypatch.setattr(s,'camera_health',health)
    return s,detected,failures,calls,stopped


@pytest.mark.parametrize('preference,devices,expected',[
 ('auto',['d405'],'d405'),('auto',['dfk33ux287'],'dfk33ux287'),('auto',['d405','dfk33ux287'],'d405'),
 ('dfk33ux287',['d405'],'d405'),('d405',['dfk33ux287'],'dfk33ux287'),
])
def test_candidate_order(rig,preference,devices,expected):
    s,detected,failures,calls,stopped=rig
    s.config['system']['camera']['profile']=preference
    detected.update({p:p in devices for p in detected})
    assert asyncio.run(s.prepare_camera())
    view=s.camera_selection.view()
    assert view['active_profile']==expected
    assert view['preferred_profile']==preference
    assert view['fallback_used']==(expected!=('d405' if preference=='auto' else preference))
    assert view['candidates'][expected]['stream_healthy']


@pytest.mark.parametrize('failure',['stream','dependency','exit','geometry','info'])
def test_failed_preferred_candidate_cleans_up_and_falls_back(rig,failure):
    s,detected,failures,calls,stopped=rig
    failures['d405']=failure
    for name in ('camera_health.json','camera_preview.jpg','camera_calibration_frame.jpg'):
        (s.root/'.state'/name).write_text('stale D405')
    before=time.monotonic();asyncio.run(s.prepare_camera())
    assert time.monotonic()-before<1.5 # immediate exits do not consume a 6-second timeout
    assert s.camera_selection.active_profile=='dfk33ux287'
    assert s.camera_selection.fallback_reason
    assert stopped[-3:]==['camera_preview','camera_monitor','camera']
    assert all(s.pm.active(k) for k in ('camera','camera_monitor','camera_preview'))
    assert {p for p,k in calls if k=='camera'}==({'dfk33ux287'} if failure=='dependency' else {'d405','dfk33ux287'})
    assert not (s.root/'.state/camera_health.json').exists()
    assert not (s.root/'.state/camera_preview.jpg').exists()
    assert read_json(s.root/'.state/camera_active.json')['camera']['source']=='tiscamera'


def test_neither_camera_required_failure_and_optional_success(rig):
    s,detected,failures,_,_=rig
    detected.update(d405=False,dfk33ux287=False)
    assert not asyncio.run(s.camera_selection.resolve(required=False))
    with pytest.raises(ValueError,match='RGB unavailable'):asyncio.run(s.prepare_camera())
    assert not any(s.pm.active(k) for k in ('camera','camera_monitor','camera_preview'))


@pytest.mark.parametrize('active',['d405','dfk33ux287'])
def test_snapshot_and_acquisition_lock(rig,active):
    s,detected,_,calls,_=rig
    detected.update({p:p==active for p in detected})
    asyncio.run(s.prepare_camera())
    config=s.camera_selection.acquisition(True,True)
    assert config['system']['camera']['profile']==active
    assert config['camera_selection']['active_profile']==active
    prefix='d405_' if active=='d405' else ''
    assert config['camera']['extrinsics_file']==f'config/calibration/{prefix}lidar_camera.yaml'
    assert config['camera']['intrinsics_file']==('config/calibration/d405_intrinsics.yaml' if active=='d405' else 'config/calibration/camera_intrinsics.yaml')
    s.active=s.root/'locked';count=len(calls)
    other='dfk33ux287' if active=='d405' else 'd405'
    with pytest.raises(ValueError,match='locked'):asyncio.run(s.prepare_camera(profile=other))
    assert len(calls)==count
    s.config['camera']['width']=123
    assert config['camera']['width']!=123 # immutable copy


@pytest.mark.parametrize('preferred',['d405','dfk33ux287','auto'])
def test_legacy_and_auto_preferences_persist(root,preferred):
    atomic_json(root/'.state/camera_profile.json',dict(profile=preferred))
    assert load(root)['system']['camera']['profile']==preferred
    assert Service(root,True).config['system']['camera']['profile']==preferred


@pytest.mark.parametrize('data',['{bad','{}','null','[]','{"profile":"retired"}'])
def test_bad_preference_fails_soft(root,data):
    p=root/'.state/camera_profile.json';p.write_text(data)
    config=load(root)
    assert config['system']['camera']['profile']=='auto'
    assert 'Invalid camera preference' in config['warnings'][0]
    assert p.read_text()==data


def test_auto_preference_api(root):
    with TestClient(make_app(root,True)) as client:
        assert client.put('/api/camera/profile',json={'profile':'auto'}).status_code==200
    assert load(root)['system']['camera']['profile']=='auto'


def test_no_camera_scan_warns_and_snapshot_omits_rgb(root,monkeypatch):
    with TestClient(make_app(root,True)) as client:
        s=client.app.state.service;s.config['system']['camera']['enabled']=True
        async def broken():raise ValueError('publisher failed')
        monkeypatch.setattr(s,'_start_camera_candidate',broken)
        sid=capture(client,'start_scan')['session'];state=wait_state(client,'SCANNING')
        assert any('RGB unavailable' in x for x in state['warnings'])
        config=read_json(s.sessions.get(sid)/'active_config.json')
        assert not config['system']['camera']['enabled']
        assert config['camera_selection']['rgb_requested'] and not config['camera_selection']['rgb_recorded']
        assert set(commands.acquisition_topics(config))=={'points_topic','imu_topic'}
        assert s.config['system']['camera']['enabled']
        capture(client,'stop_scan');wait_state(client,'COMPLETE')
        assert not read_json(s.sessions.get(sid)/'metadata.json')['camera_selection']['rgb_recorded']


@pytest.mark.parametrize('operation',['camera','calibration'])
def test_no_camera_rejects_camera_only_and_calibration(root,monkeypatch,operation):
    with TestClient(make_app(root,True)) as client:
        s=client.app.state.service;s.config['system']['camera']['enabled']=True
        async def broken():raise ValueError('publisher failed')
        monkeypatch.setattr(s,'_start_camera_candidate',broken)
        if operation=='camera':
            sid=capture(client,'start_camera_recording')['session']
            assert 'RGB unavailable' in wait_state(client,'COMPLETE')['error']
            assert not (s.sessions.get(sid)/'raw_bag').exists()
        else:
            response=client.post('/api/calibrations',json={'name':'unavailable'})
            assert response.status_code==409 and 'RGB unavailable' in response.text
            assert not s.calibrations.list()


@pytest.mark.parametrize('profile',['d405','dfk33ux287'])
def test_actual_profile_frozen_in_recording(root,monkeypatch,profile):
    with TestClient(make_app(root,True)) as client:
        s=client.app.state.service;s.config['system']['camera']['enabled']=True
        s.config['system']['camera']['profile']='d405' if profile=='dfk33ux287' else 'dfk33ux287'
        original=s._start_camera_candidate
        async def candidate():
            if s.camera_selection.active_profile!=profile:raise ValueError('preferred unavailable')
            return await original()
        monkeypatch.setattr(s,'_start_camera_candidate',candidate)
        sid=capture(client,'start_scan')['session'];wait_state(client,'SCANNING')
        path=s.sessions.get(sid)/'active_config.json';original_bytes=path.read_bytes()
        assert read_json(path)['system']['camera']['profile']==profile
        assert client.put('/api/camera/profile',json={'profile':'auto'}).status_code==409
        async def switch():await s.prepare_camera(profile='d405' if profile=='dfk33ux287' else 'dfk33ux287')
        with pytest.raises(ValueError,match='locked'):client.portal.call(switch)
        assert path.read_bytes()==original_bytes
        capture(client,'stop_scan');wait_state(client,'COMPLETE')
        assert read_json(s.sessions.get(sid)/'metadata.json')['camera_selection']['rgb_recorded']


def test_polling_does_not_lose_candidate_failure_details(rig,monkeypatch):
    s,_,failures,_,_=rig;failures['d405']='stream'
    health=s.camera_health
    def polling_health():
        s.camera_selection.refresh_detection()
        return health()
    monkeypatch.setattr(s,'camera_health',polling_health)
    asyncio.run(s.prepare_camera())
    assert s.camera_selection.view()['candidates']['d405']['stream_healthy'] is False
    assert s.camera_selection.view()['candidates']['d405']['reason']


def test_old_health_generation_cannot_make_new_camera_ready(root):
    s=Service(root,False);s.config['system']['camera']['enabled']=True
    s.pm.items={'camera':{'state':'running'},'camera_monitor':{'state':'running'}}
    s.camera_generation='new'
    atomic_json(root/'.state/camera_health.json',dict(updated_at=time.time(),healthy=True,camera_generation='old'))
    assert not s.camera_health()['healthy']
    assert s.camera_health()['state']=='monitor_stale'


def test_choosing_auto_replaces_invalid_preference(root):
    (root/'.state/camera_profile.json').write_text('{bad')
    with TestClient(make_app(root,True)) as client:
        assert client.put('/api/camera/profile',json={'profile':'auto'}).status_code==200
    assert read_json(root/'.state/camera_profile.json')=={'profile':'auto'}


def test_calibration_dataset_cannot_fall_back_to_another_camera(rig):
    from test_calibration import measured_intrinsics
    s,detected,_,_,_=rig
    measured_intrinsics(s.root)
    s.config['system']['camera']['profile']='dfk33ux287'
    asyncio.run(s.prepare_camera())
    dataset=s.calibrations.create('DFK alignment')
    frozen=read_json(s.calibrations.get(dataset['id'])/'active_config.json')
    assert frozen['system']['camera']['profile']=='dfk33ux287'
    assert frozen['camera']['intrinsics_file']=='config/calibration/camera_intrinsics.yaml'
    asyncio.run(s.stop_camera());detected['dfk33ux287']=False
    with pytest.raises(ValueError,match='RGB unavailable'):asyncio.run(s.calibrations.capture_start(dataset['id']))
    assert not s.pm.active('calibration_record')
    assert s.camera_selection.active_profile is None
