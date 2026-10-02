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
