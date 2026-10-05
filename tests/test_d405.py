"""D405 integration keeps the existing DFK regression fixtures intact."""
import json
import yaml
import numpy as np
import pytest
from factory_mapping.config import load
from factory_mapping import commands
from factory_mapping.calibration_data import intrinsics_status
from factory_mapping.realsense_camera import rectification_maps, save_calibration


def test_default_profile_and_rgb_topics(root):
    path=root/'config/system.yaml'
    system=yaml.safe_load(path.read_text());system['camera'].pop('profile')
    path.write_text(yaml.safe_dump(system))
    config=load(root);c=config['camera']
    assert c['source']=='realsense'
    assert c['serial_number']=='230322276078'
    args=commands.camera(root,config)
    assert args[1:3]==['-m','factory_mapping.realsense_camera']
    assert json.loads(args[-1])==c
    assert intrinsics_status(root,c)['status']=='VALID'
    assert c['extrinsics_file']!='config/calibration/lidar_camera.yaml'


def test_rectification_matches_sdk_rays():
    rs=pytest.importorskip('pyrealsense2')
    intr=rs.intrinsics();intr.width=32;intr.height=24
    intr.fx=16.;intr.fy=16.;intr.ppx=16.;intr.ppy=12.
    intr.model=rs.distortion.inverse_brown_conrady
    intr.coeffs=[-.0504,.0605,-.00028,.00202,-.0209]
    mx,my=rectification_maps(intr)
    for x,y in [(0,0),(31,23),(16,12),(3,20)]:
        ray=rs.rs2_deproject_pixel_to_point(intr,[float(mx[y,x]),float(my[y,x])],1.)
        np.testing.assert_allclose(ray,[(x-intr.ppx)/intr.fx,(y-intr.ppy)/intr.fy,1.],atol=2e-5)


def test_unchanged_factory_intrinsics_preserve_alignment(root):
    c=yaml.safe_load((root/'config/camera/d405.yaml').read_text())
    path=root/c['intrinsics_file']; obj=yaml.safe_load(path.read_text())
    ext=root/c['extrinsics_file'];ext.write_text('calibrated: true\nvalidated: true\n')
    save_calibration(root,c,obj)
    assert yaml.safe_load(ext.read_text())['validated']
    obj['factory_calibration']['serial_number']='replacement'
    save_calibration(root,c,obj)
    assert not yaml.safe_load(ext.read_text())['calibrated']
    assert list((ext.parent/'history').glob('*d405_lidar_camera.yaml'))


def test_d405_usb_detection(root,tmp_path):
    from factory_mapping.camera import detect_camera
    c=yaml.safe_load((root/'config/camera/d405.yaml').read_text())
    usb=tmp_path/'usb'/'4-1';usb.mkdir(parents=True)
    for key,value in dict(idVendor='8086',product='Intel(R) RealSense(TM) Depth Camera 405',serial=c['usb_serial_number'],speed='10000').items():
        (usb/key).write_text(value)
    assert detect_camera(c,usb.parent)['detected']
    c['usb_serial_number']='other'
    assert not detect_camera(c,usb.parent)['detected']


@pytest.mark.parametrize('model',['brown_conrady','inverse_brown_conrady'])
def test_vectorized_maps_match_native_projection(model):
    rs=pytest.importorskip('pyrealsense2')
    intr=rs.intrinsics();intr.width=1280;intr.height=720
    intr.fx=660.;intr.fy=662.;intr.ppx=639.5;intr.ppy=359.5
    intr.model=getattr(rs.distortion,model)
    intr.coeffs=[-.0504,.0605,-.00028,.00202,-.0209]
    mx,my=rectification_maps(intr)
    for x,y in [(0,0),(1279,719),(640,360),(0,719),(1279,0),(91,603),(1110,115)]:
        expected=rs.rs2_project_point_to_pixel(intr,[(x-intr.ppx)/intr.fx,(y-intr.ppy)/intr.fy,1.])
        # float32 SIMD/SDK rounding at 1280 pixels; much smaller than the
        # 1/32 pixel interpolation table quantization used by OpenCV remap.
        np.testing.assert_allclose([mx[y,x],my[y,x]],expected,rtol=0,atol=3e-4)


def test_no_distortion_skips_remap():
    rs=pytest.importorskip('pyrealsense2')
    intr=rs.intrinsics();intr.model=rs.distortion.none
    assert rectification_maps(intr) is None


def test_d405_delivered_rate_is_measured(root,monkeypatch):
    from factory_mapping.camera import CameraMetrics
    c=yaml.safe_load((root/'config/camera/d405.yaml').read_text())
    assert c['fps']==c['expected_hz']==30
    metrics=CameraMetrics(c)
    clock=[100.]
    monkeypatch.setattr('factory_mapping.health.time.monotonic',lambda:clock[0])
    for i in range(60):
        clock[0]=100+i/30;metrics.image(clock[0],c['width'],c['height'],c['frame_id'])
    assert metrics.view()['healthy']
    assert metrics.view()['image_hz']==pytest.approx(30)
    clock[0]+=3
    assert not metrics.view()['healthy']
