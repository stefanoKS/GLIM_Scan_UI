"""Projection validation requires no ROS, devices, GLIM, GPU or NKSR runtime."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from factory_mapping import colorization as c


@pytest.fixture
def session(tmp_path):
    root = Path(__file__).resolve().parents[1]
    camera = yaml.safe_load((root/'config/camera/d405.yaml').read_text())
    s = tmp_path/'session'
    calibration = s/'config_snapshot/calibration'
    calibration.mkdir(parents=True)
    intr = (root/'config/calibration/d405_intrinsics.yaml').read_bytes()
    (calibration/'d405_intrinsics.yaml').write_bytes(intr)
    ext = dict(calibrated=True, validated=True, camera_name=camera['camera_name'],
               width=camera['width'], height=camera['height'],
               intrinsics_sha256=hashlib.sha256(intr).hexdigest(),
               T_lidar_camera=[0, 0, 0, 0, 0, 0, 1],
               transform_convention='p_lidar = T_lidar_camera * p_camera', time_offset_sec=0.)
    (calibration/'d405_lidar_camera.yaml').write_text(yaml.safe_dump(ext))
    (s/'active_config.json').write_text(json.dumps(dict(camera=camera, system=dict(camera=dict(enabled=True, profile='d405')))))
    trajectory = s/'processing/run_001/glim_dump/traj_lidar.txt'
    trajectory.parent.mkdir(parents=True)
    np.savetxt(trajectory, [[1, 0, 0, 0, 0, 0, 0, 1], [3, 2, 0, 0, 0, 0, 0, 1]])
    run = s/'reconstruction/run_123456789abc'
    (run/'input').mkdir(parents=True)
    np.savez(run/'input/nksr_input.npz', points=np.array([[1., 0, 2], [1, 0, -2]]), timestamps=[1., 2.])
    (run/'job.json').write_text(json.dumps(dict(state='PREPARED', created_at='2026-10-06', trajectory=str(trajectory.relative_to(s)))))
    (s/'raw_bag').mkdir()
    (s/'raw_bag/metadata.yaml').write_text('')
    return s


def test_composition_world_to_camera_and_corrected_interpolation(monkeypatch):
    q = np.sqrt(.5)
    trajectory = np.array([[1, 0, 0, 0, 0, 0, 0, 1], [3, 2, 0, 0, 0, 0, 1, 0]])
    original = c.transform_points
    called = []
    def spy(points, timestamps, traj):
        called.extend(timestamps)
        return original(points, timestamps, traj)
    monkeypatch.setattr(c, 'transform_points', spy)
    # Interpolated LiDAR rotation +90 deg; extrinsic +90 deg => camera +180 deg.
    rotation, origin, timestamp = c.camera_pose(trajectory, 1.75, .25, [1, 0, 0, 0, 0, q, q])
    np.testing.assert_allclose(rotation, np.diag([-1, -1, 1]), atol=1e-14)
    np.testing.assert_allclose(origin, [1, 1, 0], atol=1e-14)
    np.testing.assert_allclose((np.array([0, 1, 2])-origin) @ rotation, [1, 0, 2], atol=1e-14)
    assert timestamp == 2 and called == [2]*4


def test_pinhole_filters_and_zbuffer():
    points = np.array([[0, 0, 2], [0, 0, 3], [1, 0, 2], [0, 0, -1],
                       [0, 0, 0], [2, 0, 1], [-2, 0, 1], [0, 2, 1], [0, -2, 1], [0, 0, 21]])
    pixels, depth = c.project_points(points, np.eye(3), np.zeros(3), [10, 10, 10, 10], 20, 20, 20)
    np.testing.assert_array_equal(pixels, [[10, 10], [15, 10]])
    np.testing.assert_array_equal(depth, [2, 2])


def test_snapshot_and_rectified_intrinsics(session):
    # Conflicting current/global files must never influence this session.
    global_path = session.parent/'config/calibration'
    global_path.mkdir(parents=True)
    (global_path/'d405_intrinsics.yaml').write_text('calibrated: false\n')
    camera, intr, _, meta = c.session_calibration(session)
    assert intr['distortion'] == [0]*5
    obj = yaml.safe_load(Path(meta['intrinsics_file']).read_text())
    assert any(obj['factory_calibration']['coefficients'])
    pixels, _ = c.project_points(np.array([[0, 0, 2]]), np.eye(3), np.zeros(3),
                                intr['intrinsics'], camera['width'], camera['height'], 20)
    assert pixels.tolist() == [[int(intr['intrinsics'][2]), int(intr['intrinsics'][3])]]
    assert 'config_snapshot' in meta['intrinsics_file']
    Path(meta['intrinsics_file']).unlink()
    with pytest.raises(ValueError, match='Missing session calibration snapshot'):
        c.session_calibration(session)


@pytest.mark.parametrize('changes,allow,message', [
    ({'calibrated': False}, True, 'uncalibrated'),
    ({'validated': False}, False, 'unvalidated'),
    ({'transform_convention': 'inverse'}, True, 'convention'),
    ({'intrinsics_sha256': 'wrong'}, False, 'hash'),
    ({'time_offset_sec': .1}, False, 'offsets disagree'),
    ({'T_lidar_camera': [0]*7}, False, 'quaternion'),
])
def test_invalid_calibration(session, changes, allow, message):
    path = session/'config_snapshot/calibration/d405_lidar_camera.yaml'
    ext = yaml.safe_load(path.read_text()); ext.update(changes)
    path.write_text(yaml.safe_dump(ext))
    with pytest.raises(ValueError, match=message):
        c.session_calibration(session, allow)


def test_unvalidated_override(session):
    path = session/'config_snapshot/calibration/d405_lidar_camera.yaml'
    ext = yaml.safe_load(path.read_text()); ext['validated'] = False
    path.write_text(yaml.safe_dump(ext))
    with pytest.warns(UserWarning, match='unvalidated'):
        assert not c.session_calibration(session, True)[3]['calibration_validated']


def test_selected_images_metadata_and_preserved_inputs(session, monkeypatch):
    import cv2
    camera, _, _, calibration = c.session_calibration(session)
    width, height = camera['width'], camera['height']
    rgb = np.zeros((height, width, 3), np.uint8); rgb[:, :, 0] = 40
    msg = SimpleNamespace(width=width, height=height, encoding='rgb8', step=width*3, data=rgb.tobytes())
    def images(bag, topic):
        assert bag == session/'raw_bag' and topic == camera['image_topic']
        for stamp in np.linspace(1, 3, 9):
            yield float(stamp), int(stamp*1e9), msg
    monkeypatch.setattr(c, 'bag_images', images)
    inputs = [*session.glob('config_snapshot/calibration/*'), *session.glob('reconstruction/*/input/*')]
    before = {p: p.read_bytes() for p in inputs}
    output = c.validate_session(session, frames=3, time_offset=.25)
    meta = json.loads((output/'metadata.json').read_text())
    assert len(list(output.glob('*.png'))) == 3
    assert meta['time_offset_sec'] == .25 and meta['calibration_time_offset_sec'] == 0
    assert meta['extrinsics_sha256'] == calibration['extrinsics_sha256']
    assert meta['extrinsics_file'] == calibration['extrinsics_file']
    assert meta['number_source_points'] == 2
    assert meta['frames'][0]['camera_timestamp'] == 1
    assert all(f['lidar_pose_timestamp'] == f['camera_timestamp']+.25 for f in meta['frames'])
    assert all(f['number_projected'] == 1 for f in meta['frames'])
    assert cv2.imread(str(output/'frame_000000.png'))[0, 0].tolist() == [0, 0, 40]
    assert before == {p: p.read_bytes() for p in inputs}
    assert c.validate_session(session, frames=1, time_offset=-.25) != output


@pytest.mark.parametrize('missing,message', [
    ('reconstruction/run_123456789abc/input/nksr_input.npz', 'No completed reconstruction'),
    ('processing/run_001/glim_dump/traj_lidar.txt', 'valid trajectory'),
    ('raw_bag/metadata.yaml', 'rosbag is missing'),
])
def test_missing_inputs(session, missing, message):
    (session/missing).unlink()
    with pytest.raises(ValueError, match=message):
        c.validate_session(session)


def test_no_rgb_and_no_overlap(session, monkeypatch):
    monkeypatch.setattr(c, 'bag_images', lambda *args: iter([]))
    with pytest.raises(ValueError, match='No recorded RGB frames'):
        c.validate_session(session)
    with pytest.raises(ValueError, match='outside the trajectory'):
        c.camera_pose(np.array([[1, 0, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0, 0, 1]]), 0, 0, [0, 0, 0, 0, 0, 0, 1])


def test_image_padding():
    msg = SimpleNamespace(width=1, height=2, encoding='rgb8', step=4, data=bytes([1, 2, 3, 0, 4, 5, 6, 0]))
    assert c.image_bgr(msg).tolist() == [[[3, 2, 1]], [[6, 5, 4]]]


def test_bag_reader_uses_header_timestamp_and_saved_topic(monkeypatch):
    import sys
    msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=500000000)))
    class Reader:
        def open(self, *args):
            self.pending = True
        def get_all_topics_and_types(self):
            return [SimpleNamespace(name='/saved/rgb', type='sensor_msgs/msg/Image')]
        def set_filter(self, value):
            assert value.topics == ['/saved/rgb']
        def has_next(self):
            return self.pending
        def read_next(self):
            self.pending = False
            return '/saved/rgb', b'image', 13000000000
    monkeypatch.setitem(sys.modules, 'rosbag2_py', SimpleNamespace(
        SequentialReader=Reader, StorageOptions=SimpleNamespace,
        ConverterOptions=lambda *args: args, StorageFilter=SimpleNamespace))
    monkeypatch.setitem(sys.modules, 'rclpy.serialization', SimpleNamespace(deserialize_message=lambda *args: msg))
    monkeypatch.setitem(sys.modules, 'sensor_msgs.msg', SimpleNamespace(Image=object))
    assert list(c.bag_images(Path('/bag'), '/saved/rgb')) == [(12.5, 13000000000, msg)]
    with pytest.raises(ValueError, match='No recorded RGB Image topic'):
        list(c.bag_images(Path('/bag'), '/wrong'))


# ---------------------------------------------------------------------------
# Colorization pipeline unit and integration tests.
# ---------------------------------------------------------------------------

def test_project_to_pixels_rejects_behind_and_out_of_bounds():
    intrinsics = [100.0, 100.0, 50.0, 50.0]
    points = np.array([[0, 0, 2], [0, 0, -1], [0, 0, 0], [1, 0, 1], [0, 0, 30]])
    _, pixels, depth, masks = c.project_to_pixels(
        points, np.eye(3), np.zeros(3), intrinsics, 100, 100, 0.1, 20.0)
    assert masks['valid'].tolist() == [True, False, False, False, False]
    assert masks['in_front'].tolist() == [True, False, False, True, True]
    assert not masks['in_bounds'][3]  # u == 150 outside the 100-wide image
    assert not masks['in_range'][4]   # 30 m beyond max_depth
    np.testing.assert_allclose(pixels[0], [50.0, 50.0])
    assert depth[0] == 2.0


def test_bilinear_sample_rgb_corners_and_center():
    image = np.zeros((2, 2, 3), np.uint8)
    image[0, 0] = [0, 0, 255]    # BGR -> red
    image[0, 1] = [0, 255, 0]    # -> green
    image[1, 0] = [255, 0, 0]    # -> blue
    image[1, 1] = [255, 255, 255]
    rgb = c.bilinear_sample_rgb(image, np.array([0.0, 1.0, 0.0, 1.0]),
                                np.array([0.0, 0.0, 1.0, 1.0]))
    np.testing.assert_allclose(rgb, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], atol=1/255)
    center = c.bilinear_sample_rgb(image, np.array([0.5]), np.array([0.5]))
    np.testing.assert_allclose(center[0], [0.5, 0.5, 0.5], atol=1/255)


def test_depth_buffer_and_occlusion_front_surface_wins():
    pixels = np.array([[5.2, 5.1], [5.4, 5.3]])
    depth = np.array([1.0, 2.0])
    valid = np.array([True, True])
    buffer = c.build_depth_buffer(pixels, depth, valid, 10, 10)
    assert buffer[5, 5] == 1.0
    keep = c.occlusion_keep(pixels, depth, np.array([True, True]), buffer, 0.03, 0.0075)
    assert keep.tolist() == [True, False]
    # A point well behind the front surface is rejected even with tolerance.
    keep_far = c.occlusion_keep(np.array([[5.0, 5.0]]), np.array([3.0]),
                                np.array([True]), buffer, 0.03, 0.0075)
    assert keep_far.tolist() == [False]


def test_fuse_grouped_median_per_channel():
    group = np.array([0, 0, 0, 1, 1, 2])
    rgb = np.array([[1, 0, 0], [0.5, 0, 0], [0, 0, 0],
                    [0, 1, 0], [0, 0.2, 0], [0, 0, 1]], np.float32)
    fused, counts = c.fuse_grouped_median(group, rgb, 3)
    assert counts.tolist() == [3, 2, 1]
    np.testing.assert_allclose(fused[0], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(fused[1], [0.0, 0.6, 0.0])
    np.testing.assert_allclose(fused[2], [0.0, 0.0, 1.0])


def test_fuse_grouped_median_empty_group_is_nan():
    fused, counts = c.fuse_grouped_median(np.array([0]), np.array([[1.0, 1.0, 1.0]], np.float32), 3)
    assert counts.tolist() == [1, 0, 0]
    assert np.isnan(fused[1]).all() and np.isnan(fused[2]).all()
    np.testing.assert_allclose(fused[0], [1.0, 1.0, 1.0])


def test_voxel_grouping_representative_shares_voxel_observations():
    points = np.array([[0.0, 0, 0], [0.001, 0, 0], [0.1, 0, 0]])
    representative, voxel_of_point, voxel_of_rep = c._voxel_grouping(points, 0.01)
    assert representative.tolist() == [0, 2]
    assert voxel_of_point.tolist() == [0, 0, 1]
    # Voxel 0 (point 0) and voxel 1 (point 2) map back to the two output rows.
    assert voxel_of_rep.tolist() == [0, 1]
    # Observation collected from the non-representative point colors the voxel.
    fused, counts = c.fuse_grouped_median(np.array([0]), np.array([[1.0, 0.5, 0.25]], np.float32), 2)
    np.testing.assert_allclose(fused[0], [1.0, 0.5, 0.25])
    np.testing.assert_allclose(points[representative], [[0, 0, 0], [0.1, 0, 0]])


def test_write_colored_ply_and_npz(tmp_path):
    points = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    rgb = np.array([[255, 0, 0], [0, 0, 255]], np.uint8)
    intensity = np.array([0.5, 1.0], np.float32)
    ply = tmp_path/'out.ply'
    c.write_colored_ply(ply, points, rgb, intensity)
    from plyfile import PlyData
    vertex = PlyData.read(str(ply))['vertex'].data
    assert list(vertex.dtype.names) == ['x', 'y', 'z', 'red', 'green', 'blue', 'intensity']
    np.testing.assert_array_equal(vertex['red'], [255, 0])
    np.testing.assert_array_equal(vertex['blue'], [0, 255])
    np.testing.assert_allclose(vertex['intensity'], [0.5, 1.0])

    cd = np.array([[0.25, 0.5, 0.75]], np.float32)
    npz = tmp_path/'out.npz'
    c.write_colored_npz(npz, points[:1], cd, np.array([0.5], np.float32), np.array([2], np.int32),
                        intensity=np.array([1.0], np.float32))
    data = np.load(npz, allow_pickle=False)
    assert data['points'].shape == (1, 3)
    assert data['Cd'].shape == (1, 3)
    assert data['Cd'].min() >= 0.0 and data['Cd'].max() <= 1.0
    assert data['color_count'].shape == (1,)
    assert data['color_confidence'].shape == (1,)
    np.testing.assert_allclose(data['Cd'][0], [0.25, 0.5, 0.75])


def test_deterministic_fusion_and_grouping():
    points = np.array([[0.0, 0, 0], [0.001, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])
    first = c._voxel_grouping(points, 0.01)
    second = c._voxel_grouping(points, 0.01)
    assert first[0].tolist() == second[0].tolist()
    assert first[1].tolist() == second[1].tolist()
    assert first[2].tolist() == second[2].tolist()
    group = np.tile(np.arange(2), 3)
    rgb = np.random.default_rng(0).random((6, 3)).astype(np.float32)
    f1, n1 = c.fuse_grouped_median(group, rgb, 2)
    f2, n2 = c.fuse_grouped_median(group, rgb, 2)
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(n1, n2)


def test_colorize_session_end_to_end(session, monkeypatch):
    camera, intr, _, _ = c.session_calibration(session)
    width, height = camera['width'], camera['height']
    fx, fy, cx, cy = intr['intrinsics']

    def make_image(color):
        rgb = np.zeros((height, width, 3), np.uint8)
        rgb[:, :, 0] = color[0]
        rgb[:, :, 1] = color[1]
        rgb[:, :, 2] = color[2]
        return SimpleNamespace(width=width, height=height, encoding='rgb8',
                               step=width * 3, data=rgb.tobytes())

    frames = [(1.5, make_image((255, 0, 0))), (2.5, make_image((0, 0, 255)))]

    def images(bag, topic):
        for stamp, msg in frames:
            yield stamp, int(stamp * 1e9), msg
    monkeypatch.setattr(c, 'bag_images', images)

    records = dict(
        points=np.array([[0.5, 0, 2], [0.5, 0, -1], [0.5, 0, 6]], np.float64),
        sensor_origins=np.zeros((3, 3)),
        intensity=np.array([0.2, 0.4, 0.6], np.float32),
        timestamps=np.array([1.5, 1.5, 2.5], np.float64))
    monkeypatch.setattr(c, 'read_measurements', lambda *a, **k: records)

    output = c.colorize_session(session, voxel_size=0.01, max_time_delta=0.15,
                                max_depth=20.0, validation_frames=1)
    meta = json.loads((output/'metadata.json').read_text())
    assert meta['statistics']['input_point_count'] == 3
    assert meta['statistics']['final_point_count'] == 3
    assert meta['statistics']['colored_point_count'] == 2
    assert meta['statistics']['uncolored_point_count'] == 1
    assert meta['statistics']['behind_camera_count'] == 1
    assert meta['statistics']['colored_observation_count'] == 2

    data = np.load(output/'output/colored_points.npz', allow_pickle=False)
    assert data['points'].shape == (3, 3)
    assert data['Cd'].shape == (3, 3)
    assert data['color_count'].tolist() == [1, 0, 1]
    assert np.isnan(data['Cd'][1]).all()
    np.testing.assert_allclose(data['Cd'][0], [1.0, 0.0, 0.0], atol=1/255)
    np.testing.assert_allclose(data['Cd'][2], [0.0, 0.0, 1.0], atol=1/255)
    assert data['Cd'][data['color_count'] > 0].min() >= 0.0
    assert data['Cd'][data['color_count'] > 0].max() <= 1.0

    from plyfile import PlyData
    vertex = PlyData.read(str(output/'output/colored_points.ply'))['vertex'].data
    assert list(vertex['red']) == [255, 128, 0]
    assert list(vertex['blue']) == [0, 128, 255]

    stats = json.loads((output/'output/colorization_stats.json').read_text())
    assert stats['final_point_count'] == 3 and stats['colored_point_count'] == 2
    assert (output/'calibration_snapshot.yaml').is_file()

    overlays = list((output/'validation').glob('frame_*_overlay.jpg'))
    assert len(overlays) == 1
    assert meta['statistics']['validation_frames_written'] == 1
