import asyncio
import json
import time
import yaml
import pytest
from fastapi.testclient import TestClient
from factory_mapping.api import make_app
from factory_mapping.config import load
from factory_mapping.calibration_data import parse_intrinsics,parse_result,require_intrinsics
from factory_mapping.calibration import Calibrations,hashes,tool_command,bag_statistics
from factory_mapping.service import Service
from factory_mapping.storage import atomic_json,read_json


def measured_intrinsics(root):
    # Synthetic measured values used only in tests, never shipped as calibration.
    c=load(root)['camera'];w,h=c['width'],c['height']
    obj={'image_width':w,'image_height':h,'camera_name':'factory_rgb','distortion_model':'plumb_bob','camera_matrix':{'rows':3,'cols':3,'data':[800.,0,w/2,0,800.,h/2,0,0,1]},'distortion_coefficients':{'rows':1,'cols':5,'data':[0.]*5},'rectification_matrix':{'rows':3,'cols':3,'data':[1,0,0,0,1,0,0,0,1]},'projection_matrix':{'rows':3,'cols':4,'data':[800,0,w/2,0,0,800,h/2,0,0,0,1,0]}}
    (root/'config/calibration/camera_intrinsics.yaml').write_text(yaml.safe_dump(obj));return obj


def enabled(root):
    p=root/'config/system.yaml';c=yaml.safe_load(p.read_text());c['camera']['enabled']=True;p.write_text(yaml.safe_dump(c));return load(root)


def result(root,pose):
    intr=require_intrinsics(root,load(root)['camera'])
    obj={'camera':{'camera_model':intr['model'],'intrinsics':intr['intrinsics'],'distortion_coeffs':intr['distortion']},'results':{'T_lidar_camera':pose}}
    path=root/'calib.json';atomic_json(path,obj);return path,intr


@pytest.mark.parametrize('pose',[[0]*6,[0,0,0,0,0,0,0],[0,0,0,0,0,0,2],['x',0,0,0,0,0,1]])
def test_result_rejects_bad_transform(root,pose):
    measured_intrinsics(root);path,intr=result(root,pose)
    with pytest.raises(ValueError):parse_result(path,intr)


def test_result_convention_and_model(root):
    measured_intrinsics(root);pose=[1,2,3,0,0,0,1];path,intr=result(root,pose)
    assert parse_result(path,intr)==pose
    obj=json.loads(path.read_text());obj['camera']['camera_model']='fisheye';atomic_json(path,obj)
    with pytest.raises(ValueError,match='model'):parse_result(path,intr)


def test_intrinsics_required_before_calibration(root):
    enabled(root);s=Service(root,True)
    with pytest.raises(ValueError,match='intrinsics'):s.calibrations.create('test')


def test_capture_immutable_and_independent_from_mapping(root):
    enabled(root);measured_intrinsics(root)
    with TestClient(make_app(root,True)) as c:
        cid=c.post('/api/calibrations',json={'name':'static views'}).json()['id']
        def act(a):return c.post(f'/api/calibrations/{cid}/action',json={'action':a})
        c.post('/api/action',json={'action':'driver_start'});c.post('/api/action',json={'action':'camera_start'})
        assert act('capture_start').status_code==200
        p=root/'data/calibrations'/cid
        log=p/'captures/capture_001/capture.log'
        for _ in range(30):
            if 'started' in log.read_text():break
            time.sleep(.1)
        assert c.get('/api/sessions').json()==[]
        assert act('capture_start').status_code==409
        assert act('capture_stop').json()['state']=='CAPTURED'
        before=hashes(p/'captures/capture_001/raw_bag')
        assert act('capture_start').status_code==200
        time.sleep(.2);assert act('capture_stop').status_code==200
        assert hashes(p/'captures/capture_001/raw_bag')==before
        assert len(c.get(f'/api/calibrations/{cid}').json()['captures'])==2
        assert act('initial_guess_auto').status_code==409
        (p/'captures/capture_001/raw_bag/MOCK_ONLY.txt').write_text('tampered')
        with pytest.raises(ValueError,match='changed'):c.app.state.service.calibrations.verify_captures(p)


def test_real_result_import_preserves_original_and_does_not_validate(root):
    enabled(root);measured_intrinsics(root);s=Service(root,False);cal=s.calibrations
    m=cal.create('import fixture');p=cal.get(m['id']);cap=p/'captures/capture_001';(cap/'raw_bag').mkdir(parents=True);(cap/'raw_bag/example').write_text('raw')
    atomic_json(cap/'metadata.json',dict(state='CAPTURED',raw_hashes=hashes(cap/'raw_bag')))
    work=p/'jobs/job_fixture/work';work.mkdir(parents=True)
    path,intr=result(root,[1,2,3,0,0,0,1]);(work/'calib.json').write_bytes(path.read_bytes());cal.update(p,state='CALIBRATED',work=str(work.relative_to(p)))
    before=hashes(cap/'raw_bag');out=cal.import_result(m['id'])
    assert out['state']=='IMPORTED' and out['validated'] is False
    canonical=yaml.safe_load((root/'config/calibration/lidar_camera.yaml').read_text())
    assert canonical['T_lidar_camera']==[1,2,3,0,0,0,1]
    assert canonical['transform_convention']=='p_lidar = T_lidar_camera * p_camera'
    assert (p/'result/calib.json').read_bytes()==path.read_bytes()
    assert hashes(cap/'raw_bag')==before
    with pytest.raises(ValueError):cal.import_result(m['id'])
    cal.validate(m['id'],'Independent overlay reviewed on held-out views; residual criteria documented by operator.')
    assert read_json(p/'metadata.json')['state']=='VALIDATED'


def test_fixed_manual_tool_commands(root):
    c=enabled(root);measured_intrinsics(root);intr=require_intrinsics(root,c['camera'])
    args=tool_command('preprocess',root/'out',c,intr,root/'inputs')
    assert '--image_topic' in args and '--camera_info_topic' in args and '--points_topic' in args
    assert '--auto_topic' not in args
    with pytest.raises(ValueError):tool_command('initial_guess_auto',root,c,intr)


def test_nonfinite_result_rejected(root):
    measured_intrinsics(root);p,intr=result(root,[0,0,0,0,0,0,1]);obj=json.loads(p.read_text());obj['results']['T_lidar_camera'][0]=float('nan');p.write_text(json.dumps(obj))
    with pytest.raises(ValueError):parse_result(p,intr)


def test_capture_statistics_reject_missing_or_slow_camera(root):
    c=enabled(root);bag=root/'bag';bag.mkdir()
    info={'duration':{'nanoseconds':10_000_000_000},'topics_with_message_count':[]}
    entries=info['topics_with_message_count']
    for name,kind,count in [(c['sensor']['points_topic'],'PointCloud2',100),(c['camera']['image_topic'],'Image',150),(c['camera']['camera_info_topic'],'CameraInfo',150)]:
        entries.append({'topic_metadata':{'name':name,'type':'sensor_msgs/msg/'+kind},'message_count':count})
    def save(): (bag/'metadata.yaml').write_text(yaml.safe_dump({'rosbag2_bagfile_information':info}))
    save();assert bag_statistics(bag,c)['topics']['image_topic']['hz']==15
    entries[1]['message_count']=1;save()
    with pytest.raises(ValueError,match='rate'):bag_statistics(bag,c)
    entries[1]['message_count']=150;entries.pop();save()
    with pytest.raises(ValueError,match='CameraInfo'):bag_statistics(bag,c)


def test_intrinsics_import_archives_previous_and_rejects_api_injection(root):
    enabled(root);obj=measured_intrinsics(root)
    with TestClient(make_app(root,True)) as c:
        assert c.post('/api/camera/intrinsics',json={'yaml_text':yaml.safe_dump(obj),'command':'touch evil'}).status_code==422
        assert c.post('/api/camera/intrinsics',json={'yaml_text':'!!python/object/apply:os.system [touch evil]'}).status_code==409
        assert c.post('/api/camera/intrinsics',json={'yaml_text':yaml.safe_dump(obj)}).status_code==200
        assert list((root/'config/calibration/history').glob('*'))
        c.post('/api/action',json={'action':'camera_start'})
        assert c.post('/api/camera/intrinsics',json={'yaml_text':yaml.safe_dump(obj)}).status_code==409


def test_calibration_tool_requires_display_and_failed_job_can_retry(root,monkeypatch):
    enabled(root);measured_intrinsics(root);s=Service(root,False);cal=s.calibrations;m=cal.create('native');p=cal.get(m['id'])
    cal.update(p,state='PREPROCESSED')
    monkeypatch.delenv('DISPLAY',raising=False)
    with pytest.raises(ValueError,match='DISPLAY'):asyncio.run(cal.run(m['id'],'initial_guess_manual'))
    assert read_json(p/'metadata.json')['state']=='PREPROCESSED'


def test_manual_calibration_stages_preserve_prior_outputs(root,monkeypatch):
    enabled(root);measured_intrinsics(root);s=Service(root,False);cal=s.calibrations;m=cal.create('stages');p=cal.get(m['id'])
    cap=p/'captures/capture_001';(cap/'raw_bag').mkdir(parents=True);(cap/'raw_bag/data.db3').write_bytes(b'fixture raw')
    atomic_json(cap/'metadata.json',dict(state='CAPTURED',raw_hashes=hashes(cap/'raw_bag')));cal.update(p,state='CAPTURED')
    for stage in ('preprocess','initial_guess_manual','calibrate'):
        exe=root/'ros2_ws/install/direct_visual_lidar_calibration/lib/direct_visual_lidar_calibration'/stage;exe.parent.mkdir(parents=True,exist_ok=True);exe.touch()
    monkeypatch.setenv('DISPLAY',':test')
    async def display_ok():pass
    monkeypatch.setattr('factory_mapping.calibration.ensure_display',display_ok)
    async def fake_start(key,args,log,out=None,done=None):
        assert key=='calibration_tool'
        stage=args[3];work=__import__('pathlib').Path(args[5] if stage=='preprocess' else args[4])
        if stage=='preprocess':
            work.mkdir();obj={'meta':{'bag_names':['capture_001']},'camera':{'camera_model':'plumb_bob','intrinsics':m['intrinsics']['intrinsics'],'distortion_coeffs':m['intrinsics']['distortion']}}
            (work/'capture_001.ply').write_bytes(b'fixture cloud');(work/'capture_001.png').write_bytes(b'fixture image')
        else:obj=read_json(work/'calib.json')
        if stage=='initial_guess_manual':obj['results']={'init_T_lidar_camera':[0,0,0,0,0,0,1]}
        if stage=='calibrate':obj['results']['T_lidar_camera']=[1,2,3,0,0,0,1]
        atomic_json(work/'calib.json',obj)
        await done({'state':'completed','returncode':0,'forced':False})
    monkeypatch.setattr(s,'start_process',fake_start)
    async def go():
        await cal.run(m['id'],'preprocess');first=p/read_json(p/'metadata.json')['work'];before=hashes(first)
        await cal.run(m['id'],'initial_guess_manual');assert hashes(first)==before
        second=p/read_json(p/'metadata.json')['work'];second_before=hashes(second)
        await cal.run(m['id'],'calibrate');assert hashes(second)==second_before
        assert read_json(p/'metadata.json')['state']=='CALIBRATED'
        assert (cap/'raw_bag/data.db3').read_bytes()==b'fixture raw'
    asyncio.run(go())
