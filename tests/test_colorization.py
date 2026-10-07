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


# ---------------------------------------------------------------------------
# Image-border quantization: one convention for depth-buffer build and lookup.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('offset', [0.01, 0.49, 0.51, 0.99])
@pytest.mark.parametrize('axis', ['u', 'v'])
def test_pixel_indices_never_exceed_the_image(axis, offset):
    width = height = 8
    pixels = np.array([[width-offset, 4.0]]) if axis == 'u' else np.array([[4.0, height-offset]])
    u, v = c.pixel_indices(pixels, width, height)
    assert 0 <= int(u[0]) < width and 0 <= int(v[0]) < height
    assert int(u[0]) == (width-1 if axis == 'u' else 4)
    assert int(v[0]) == (height-1 if axis == 'v' else 4)


@pytest.mark.parametrize('offset', [0.01, 0.49, 0.51, 0.99])
def test_depth_buffer_and_visibility_agree_at_the_border(offset):
    # Regression: `0 <= u < width` accepted these pixels, but rounding produced
    # the invalid index `width` both when building and when testing the buffer.
    width = height = 8
    pixels = np.array([[width-offset, height-offset], [1.0, 1.0]])
    depth = np.array([1.0, 2.0])
    valid = np.array([True, True])
    buffer = c.build_depth_buffer(pixels, depth, valid, height, width)
    assert buffer.shape == (height, width)
    assert buffer[height-1, width-1] == 1.0
    keep = c.occlusion_keep(pixels, depth, valid, buffer, 0.03, 0.0075)
    assert keep.tolist() == [True, True]  # distinct pixels, neither is occluded


@pytest.mark.parametrize('offset', [0.01, 0.49, 0.51, 0.99])
def test_near_border_points_share_one_quantized_pixel(offset):
    width = height = 8
    pixels = np.array([[width-offset, height-offset], [width-offset, height-offset]])
    depth = np.array([1.0, 3.0])
    valid = np.array([True, True])
    buffer = c.build_depth_buffer(pixels, depth, valid, height, width)
    keep = c.occlusion_keep(pixels, depth, valid, buffer, 0.03, 0.0075)
    assert keep.tolist() == [True, False]  # same pixel: near survives, far rejected


def test_border_quantization_is_the_same_function_everywhere():
    pixels = np.array([[7.4, 3.2], [0.0, 0.0]])
    u, v = c.pixel_indices(pixels, 8, 8)
    buffer = c.build_depth_buffer(pixels, np.array([1.0, 2.0]), np.array([True, True]), 8, 8)
    assert buffer[int(v[0]), int(u[0])] == 1.0
    keep = c.occlusion_keep(pixels, np.array([1.0, 2.0]), np.array([True, True]), buffer, 0.0, 0.0)
    assert keep.tolist() == [True, True]


def test_projection_rejects_outside_and_accepts_near_border():
    intrinsics = [100.0, 100.0, 50.0, 50.0]
    # depth 10: u = 50 + 100*x/10, so x = 4.999 -> u = 99.99 (near border).
    points = np.array([[4.999, 0, 10], [-5.5, 0, 10], [0, -5.5, 10], [5.5, 0, 10], [0, 5.5, 10]])
    _, pixels, depth, masks = c.project_to_pixels(points, np.eye(3), np.zeros(3), intrinsics,
                                                 100, 100, 0.1, 50.0)
    assert masks['valid'].tolist() == [True, False, False, False, False]
    assert pixels[0, 0] > 99.0 and pixels[0, 0] < 100.0
    # The accepted near-border sample must not index outside the depth buffer.
    buffer = c.build_depth_buffer(pixels, depth, masks['valid'], 100, 100)
    keep = c.occlusion_keep(pixels, depth, masks['valid'], buffer, 0.03, 0.0075)
    assert keep.tolist() == [True, False, False, False, False]


def test_project_points_validation_path_border_is_safe():
    intrinsics = [100.0, 100.0, 50.0, 50.0]
    # depth 1: u = 50 + 100*x -> x = 0.4999 gives u = 99.99, x = 0.505 gives u = 100.5.
    points = np.array([[0.4999, 0, 1.0], [0.505, 0, 1.0]])
    pixels, depth = c.project_points(points, np.eye(3), np.zeros(3), intrinsics, 100, 100, 20.0)
    assert len(depth) == 1
    assert pixels.tolist() == [[99, 50]]  # floor convention, still vertically centered
    assert int(pixels[0, 0]) < 100


# ---------------------------------------------------------------------------
# Bounded, deterministic per-voxel RGB observation storage.
# ---------------------------------------------------------------------------

def _observations(store, voxel, values):
    for value in values:
        store.add(np.array([voxel]), np.array([[value, value, value]], np.float32))


def test_bounded_store_keeps_the_first_observations_deterministically():
    store = c.BoundedObservationStore(2, 2)
    _observations(store, 0, [0.1, 0.4, 0.9])
    _observations(store, 1, [0.2, 0.6])
    fused, counts = store.voxel_observations()
    assert counts.tolist() == [2, 2]
    assert store.dropped == 1
    np.testing.assert_allclose(fused[0], [0.25]*3)  # median of the retained 0.1 and 0.4
    np.testing.assert_allclose(fused[1], [0.4]*3)
    assert store.stored == 4


def test_bounded_store_matches_grouped_median_when_unbounded():
    rng = np.random.default_rng(7)
    groups = rng.integers(0, 40, 800)
    rgb = rng.random((800, 3)).astype(np.float32)
    store = c.BoundedObservationStore(40, 64)
    store.add(groups, rgb)
    fused, counts = store.voxel_observations()
    expected, expected_counts = c.fuse_grouped_median(groups, rgb, 40)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(fused, expected, equal_nan=True)


def test_bounded_store_handles_same_batch_duplicates():
    store = c.BoundedObservationStore(1, 3)
    store.add(np.array([0, 0, 0, 0, 0]),
              np.array([[0.1]*3, [0.2]*3, [0.3]*3, [0.4]*3, [0.5]*3], np.float32))
    fused, counts = store.voxel_observations()
    assert counts.tolist() == [3] and store.dropped == 2
    np.testing.assert_allclose(fused[0], [0.2]*3)


def test_bounded_store_never_exceeds_the_limit():
    rng = np.random.default_rng(11)
    store = c.BoundedObservationStore(5, 4)
    for _ in range(20):
        store.add(rng.integers(0, 5, 50), rng.random((50, 3)).astype(np.float32))
    _, counts = store.voxel_observations()
    assert counts.max() <= 4
    # Stored memory never scales with the number of accepted observations.
    assert store.stored <= 5*4
    assert store.dropped == 1000 - store.stored


def test_bounded_store_validates_inputs():
    with pytest.raises(ValueError, match='max_observations'):
        c.BoundedObservationStore(4, 0)
    store = c.BoundedObservationStore(4, 2)
    with pytest.raises(ValueError, match='float RGB'):
        store.add(np.array([0, 1]), np.zeros((3, 3), np.float32))
    with pytest.raises(ValueError, match='out of range'):
        store.add(np.array([4]), np.zeros((1, 3), np.float32))


def test_fuse_bounded_observations_batches_like_one_pass():
    rng = np.random.default_rng(13)
    table = rng.random((1000, 6, 3)).astype(np.float32)
    counts = rng.integers(0, 7, 1000).astype(np.int64)
    whole = c.fuse_bounded_observations(table, counts)
    batched = c.fuse_bounded_observations(table, counts, batch_rows=7)
    np.testing.assert_array_equal(whole, batched)


# ---------------------------------------------------------------------------
# Global per-image depth buffer across projection chunks.
# ---------------------------------------------------------------------------

def _uniform_frame(camera, color):
    width, height = camera['width'], camera['height']
    rgb = np.zeros((height, width, 3), np.uint8)
    for channel in range(3):
        rgb[:, :, channel] = color[channel]
    return SimpleNamespace(width=width, height=height, encoding='rgb8',
                           step=width*3, data=rgb.tobytes())


def _frames(camera, colors):
    frames = [(_frame_stamp(index), _uniform_frame(camera, color))
              for index, color in enumerate(colors)]
    def images(bag, topic):
        for stamp, msg in frames:
            yield stamp, int(stamp*1e9), msg
    return images


def _frame_stamp(index):
    return 1.5 + 0.5*index


def _records(points, timestamps):
    points = np.asarray(points, np.float64)
    return dict(points=points, sensor_origins=np.zeros_like(points),
                intensity=np.zeros(len(points), np.float32),
                timestamps=np.asarray(timestamps, np.float64))


def test_depth_buffer_is_global_across_projection_chunks(session, monkeypatch):
    camera, intr, _, _ = c.session_calibration(session)
    monkeypatch.setattr(c, 'bag_images', _frames(camera, [(255, 0, 0)]))
    # Same camera ray, 0.2 m apart: the near point fills chunk A, the far point chunk B.
    monkeypatch.setattr(c, 'read_measurements', lambda *a, **k: _records(
        [(0.5, 0, 2.0), (0.5, 0, 2.2)], [1.5, 1.5]))
    output = c.colorize_session(session, voxel_size=0.01, chunk_points=1)
    stats = json.loads((output/'output/colorization_stats.json').read_text())
    assert stats['valid_projections'] == 2
    assert stats['occlusion_rejected_count'] == 1
    assert stats['colored_point_count'] == 1
    data = np.load(output/'output/colored_points.npz', allow_pickle=False)
    assert data['color_count'].tolist() == [1, 0]
    np.testing.assert_allclose(data['Cd'][0], [1.0, 0.0, 0.0], atol=1/255)
    assert np.isnan(data['Cd'][1]).all()


def test_median_fusion_over_multiple_frames_and_bounded_limit(session, monkeypatch):
    camera, _, _, _ = c.session_calibration(session)
    monkeypatch.setattr(c, 'bag_images', _frames(camera, [(51, 51, 51), (102, 102, 102), (230, 230, 230)]))
    monkeypatch.setattr(c, 'read_measurements', lambda *a, **k: _records(
        [(0.5, 0, 2.0)]*3, [1.5, 2.0, 2.5]))
    output = c.colorize_session(session, voxel_size=0.01)
    stats = json.loads((output/'output/colorization_stats.json').read_text())
    data = np.load(output/'output/colored_points.npz', allow_pickle=False)
    assert stats['colored_observation_count'] == 3
    assert stats['observations_dropped_due_to_per_voxel_limit'] == 0
    assert data['color_count'].tolist() == [3]
    np.testing.assert_allclose(data['Cd'][0], [0.4]*3, atol=1/255)  # per-channel median
    assert stats['max_color_observations_per_voxel'] == 3

    bounded = c.colorize_session(session, voxel_size=0.01, max_color_observations_per_voxel=2)
    bounded_stats = json.loads((bounded/'output/colorization_stats.json').read_text())
    bounded_data = np.load(bounded/'output/colored_points.npz', allow_pickle=False)
    assert bounded_stats['observations_dropped_due_to_per_voxel_limit'] == 1
    assert bounded_data['color_count'].tolist() == [2]
    # First two accepted observations (deterministic), median of 0.2 and 0.4.
    np.testing.assert_allclose(bounded_data['Cd'][0], [0.3]*3, atol=1/255)


def test_non_representative_observation_colors_the_voxel(session, monkeypatch):
    camera, _, _, _ = c.session_calibration(session)
    monkeypatch.setattr(c, 'bag_images', _frames(camera, [(0, 0, 255)]))
    # Both measurements share one 0.5 m voxel; the representative is out of camera time.
    monkeypatch.setattr(c, 'read_measurements', lambda *a, **k: _records(
        [(0.05, 0, 2.0), (0.10, 0, 2.0)], [3.0, 1.5]))
    output = c.colorize_session(session, voxel_size=0.5)
    stats = json.loads((output/'output/colorization_stats.json').read_text())
    data = np.load(output/'output/colored_points.npz', allow_pickle=False)
    assert stats['final_voxels'] == 1 and stats['colored_voxels'] == 1
    assert data['color_count'].tolist() == [1]
    np.testing.assert_allclose(data['Cd'][0], [0.0, 0.0, 1.0], atol=1/255)
    np.testing.assert_allclose(data['points'][0], [0.05, 0, 2.0], atol=1e-6)


def test_reported_statistics_cover_the_documented_set(session, monkeypatch):
    camera, _, _, _ = c.session_calibration(session)
    monkeypatch.setattr(c, 'bag_images', _frames(camera, [(255, 255, 255)]*3))
    monkeypatch.setattr(c, 'read_measurements', lambda *a, **k: _records(
        [(0.5, 0, 2.0), (0.5, 0, 8.0), (0.5, 0, -1.0)], [1.5, 1.5, 1.5]))
    output = c.colorize_session(session, voxel_size=0.01, validation_frames=2)
    stats = json.loads((output/'output/colorization_stats.json').read_text())
    required = ['raw_lidar_observations', 'world_observations', 'camera_frames_seen',
                'camera_frames_used', 'candidate_projections', 'behind_camera', 'out_of_frame',
                'depth_rejected', 'occlusion_rejected', 'valid_rgb_observations',
                'observations_dropped_due_to_per_voxel_limit', 'final_voxels', 'colored_voxels',
                'uncolored_voxels', 'coverage_percent', 'mean_color_observations_per_colored_voxel',
                'median_color_observations_per_colored_voxel', 'max_color_observations_per_voxel',
                'voxel_size', 'max_time_delta', 'time_offset_sec', 'occlusion_base_tolerance',
                'occlusion_range_scale']
    missing = [key for key in required if key not in stats]
    assert not missing, missing
    assert stats['raw_lidar_observations'] == 3 and stats['world_observations'] == 3
    assert stats['behind_camera'] == 1
    assert stats['final_voxels'] == stats['colored_voxels'] + stats['uncolored_voxels']
    assert stats['coverage_percent'] == pytest.approx(stats['percentage_colored'])


def test_validation_frames_span_the_scan_and_record_provenance(session, monkeypatch):
    camera, _, _, _ = c.session_calibration(session)
    stamps = np.linspace(1.0, 3.0, 21)
    frames = [(float(stamp), _uniform_frame(camera, (10, 20, 30))) for stamp in stamps]
    def images(bag, topic):
        for stamp, msg in frames:
            yield stamp, int(stamp*1e9), msg
    monkeypatch.setattr(c, 'bag_images', images)
    monkeypatch.setattr(c, 'read_measurements', lambda *a, **k: _records(
        [(0.5, 0, 2.0)], [2.0]))
    output = c.colorize_session(session, voxel_size=0.01, validation_frames=3)
    metadata = json.loads((output/'metadata.json').read_text())
    records = metadata['validation']
    assert len(records) == 3
    assert records[0]['camera_timestamp'] == pytest.approx(1.0)
    assert records[-1]['camera_timestamp'] == pytest.approx(3.0)
    timestamps = [record['camera_timestamp'] for record in records]
    assert timestamps == sorted(timestamps) and len(set(timestamps)) == 3
    assert all(record['image_topic'] == metadata['image_topic'] for record in records)
    assert all(record['time_offset_sec'] == metadata['time_offset_sec'] for record in records)
    assert all(record['lidar_pose_timestamp'] == pytest.approx(record['camera_timestamp'])
               for record in records)
    assert len(list((output/'validation').glob('frame_*_overlay.jpg'))) == 3


# ---------------------------------------------------------------------------
# Optional depth-discontinuity rejection.
# ---------------------------------------------------------------------------

def _edge_buffer(depth_values):
    buffer = np.full((5, 5), np.inf, dtype=np.float32)
    for (u, v), value in depth_values.items():
        buffer[v, u] = value
    return buffer


def test_depth_edge_rejection_is_disabled_by_zero_radius():
    keep = np.array([True])
    assert c.depth_edge_keep(np.array([[2.0, 2.0]]), np.array([2.0]), keep,
                             _edge_buffer({}), 0, 0.05).tolist() == [True]


def test_depth_edge_rejection_detects_a_neighbour_discontinuity():
    buffer = _edge_buffer({(2, 2): 2.0, (3, 2): 1.0})
    pixels = np.array([[2.0, 2.0]])
    depth = np.array([2.0])
    keep = np.array([True])
    assert c.depth_edge_keep(pixels, depth, keep, buffer, 1, 0.05).tolist() == [False]
    assert c.depth_edge_keep(pixels, depth, keep, buffer, 1, 1.5).tolist() == [True]


def test_depth_edge_rejection_ignores_the_sample_pixel_and_empty_neighbours():
    # Only the sample's own pixel is occupied: an occlusion-margin difference at
    # that pixel is not a discontinuity (empty neighbours are infinite).
    buffer = _edge_buffer({(2, 2): 1.0})
    keep = c.depth_edge_keep(np.array([[2.0, 2.0]]), np.array([2.0]), np.array([True]),
                             buffer, 1, 0.05)
    assert keep.tolist() == [True]


def test_colorize_session_depth_edge_rejection_statistics(session, monkeypatch):
    camera, intr, _, _ = c.session_calibration(session)
    fx, fy = intr['intrinsics'][0], intr['intrinsics'][1]
    monkeypatch.setattr(c, 'bag_images', _frames(camera, [(255, 255, 255)]))
    # Two measurements on diagonally adjacent pixels at clearly different depths.
    far_depth = 2.5
    near = (0.5, 0.0, 2.0)
    far = (0.5 + far_depth/fx, far_depth/fy, far_depth)
    monkeypatch.setattr(c, 'read_measurements', lambda *a, **k: _records([near, far], [1.5, 1.5]))
    output = c.colorize_session(session, voxel_size=0.01, depth_edge_rejection=True,
                                depth_edge_radius=1, depth_edge_threshold=0.05)
    stats = json.loads((output/'output/colorization_stats.json').read_text())
    assert stats['depth_edge_rejected_count'] == 2
    assert stats['colored_observation_count'] == 0
    # The same scene without the optional heuristic keeps both observations.
    plain = c.colorize_session(session, voxel_size=0.01)
    plain_stats = json.loads((plain/'output/colorization_stats.json').read_text())
    assert plain_stats['depth_edge_rejected_count'] == 0
    assert plain_stats['colored_observation_count'] == 2


def test_colorization_config_validates_the_new_switches():
    config = c.ColorizationConfig(max_color_observations_per_voxel=8).validated()
    assert config.as_dict()['max_color_observations_per_voxel'] == 8
    assert config.as_dict()['depth_edge_rejection'] is False
    for kwargs, match in [({'max_color_observations_per_voxel': 0}, 'max_color_observations_per_voxel'),
                          ({'max_color_observations_per_voxel': 65}, 'max_color_observations_per_voxel'),
                          ({'depth_edge_radius': 4}, 'depth_edge_radius'),
                          ({'depth_edge_threshold': 0.0}, 'depth_edge_threshold')]:
        with pytest.raises(ValueError, match=match):
            c.ColorizationConfig(**kwargs).validated()


def test_lidar_time_is_camera_time_plus_the_calibrated_offset():
    """Verified repository convention: t_lidar = t_camera + time_offset_sec."""
    trajectory = np.array([[1, 0, 0, 0, 0, 0, 0, 1], [3, 0, 0, 0, 0, 0, 1, 0]])
    for camera_time, offset in [(2.0, 0.0), (2.0, 0.25), (2.0, -0.25), (1.5, 0.25)]:
        _, _, lidar_time = c.camera_pose(trajectory, camera_time, offset, [0, 0, 0, 0, 0, 0, 1])
        assert lidar_time == pytest.approx(camera_time + offset)
    with pytest.raises(ValueError, match='outside the trajectory'):
        c.camera_pose(trajectory, 1.0, -0.25, [0, 0, 0, 0, 0, 0, 1])


def test_colorization_metadata_records_lineage_and_calibration(session, monkeypatch):
    camera, _, _, calibration = c.session_calibration(session)
    monkeypatch.setattr(c, 'bag_images', _frames(camera, [(255, 255, 255)]))
    monkeypatch.setattr(c, 'read_measurements', lambda *a, **k: _records([(0.5, 0, 2.0)], [1.5]))
    output = c.colorize_session(session, voxel_size=0.01)
    metadata = json.loads((output/'metadata.json').read_text())
    assert metadata['source_session'] == session.name
    assert metadata['source_bag'] == str(session/'raw_bag')
    assert metadata['source_processing_run'] == 'run_001'
    assert metadata['source_trajectory'].endswith('processing/run_001/glim_dump/traj_lidar.txt')
    assert metadata['source_glim_ply'] is None and metadata['source_nksr_mesh'] is None
    assert metadata['source_reconstruction_run'] is None
    assert metadata['camera_calibration'] == calibration['extrinsics_file']
    assert metadata['image_topic'] == camera['image_topic']
    assert metadata['camera_info_topic'] == camera['camera_info_topic']
    assert metadata['source_processing_run'] == c.processing_run_id(
        session, metadata['source_trajectory'])
