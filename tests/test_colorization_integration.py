"""End-to-end colorization and color transfer through the real bag reader.

This is the only test that exercises the actual CLI: a synthetic sqlite3 rosbag
with recorded ``Image``/``PointCloud2`` messages, the real GLIM trajectory
interpolation, the real projection, and both transfers. It is skipped where the
ROS 2 Python libraries are unavailable, because every other colorization test
uses in-memory frames and needs no ROS.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

pytest.importorskip('rosbag2_py')
pytest.importorskip('rclpy.serialization')
pytest.importorskip('sensor_msgs.msg')

import rosbag2_py  # noqa: E402
from plyfile import PlyData, PlyElement  # noqa: E402
from rclpy.serialization import serialize_message  # noqa: E402
from sensor_msgs.msg import Image, PointCloud2, PointField  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'tools/glim_colorize.py'
IMAGE_TOPIC = '/camera/image_raw'
LIDAR_TOPIC = '/livox/lidar'


def _image_message(stamp, color):
    msg = Image()
    msg.header.stamp.sec = int(stamp)
    msg.header.stamp.nanosec = int(round((stamp - int(stamp))*1e9))
    msg.width, msg.height, msg.encoding, msg.step = 1280, 720, 'rgb8', 1280*3
    msg.data = bytes(color)*1280*720
    return msg


def _lidar_message(points, intensity, timestamps):
    fields = [('x', 0), ('y', 4), ('z', 8), ('intensity', 12), ('timestamp', 16)]
    formats = ['<f4', '<f4', '<f4', '<f4', '<f8']
    records = np.empty(len(points), dtype=list(zip([f[0] for f in fields], formats)))
    for index, axis in enumerate('xyz'):
        records[axis] = points[:, index]
    records['intensity'] = intensity
    records['timestamp'] = np.asarray(timestamps, np.float64)*1e9  # Livox absolute nanoseconds
    msg = PointCloud2()
    msg.height, msg.width, msg.point_step, msg.row_step = 1, len(points), 24, 24*len(points)
    msg.is_dense = True
    msg.fields = [PointField(name=name, offset=offset,
                             datatype=(PointField.FLOAT64 if name == 'timestamp' else PointField.FLOAT32),
                             count=1) for name, offset in fields]
    msg.data = records.tobytes()
    return msg


def _write_bag(path, frames, clouds):
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    writer.create_topic(rosbag2_py.TopicMetadata(name=IMAGE_TOPIC, type='sensor_msgs/msg/Image',
                                                 serialization_format='cdr'))
    writer.create_topic(rosbag2_py.TopicMetadata(name=LIDAR_TOPIC, type='sensor_msgs/msg/PointCloud2',
                                                 serialization_format='cdr'))
    for stamp, msg in frames:
        writer.write(IMAGE_TOPIC, serialize_message(msg), int(stamp*1e9))
    for stamp, msg in clouds:
        writer.write(LIDAR_TOPIC, serialize_message(msg), int(stamp*1e9))
    del writer


def _write_points_ply(path, points, intensity):
    vertex = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                          ('intensity', '<f4')])
    for index, axis in enumerate('xyz'):
        vertex[axis] = points[:, index].astype(np.float32)
    vertex['intensity'] = np.asarray(intensity, np.float32)
    PlyData([PlyElement.describe(vertex, 'vertex')], text=False, byte_order='<').write(str(path))


def _write_mesh_ply(path, points, faces):
    vertex = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    for index, axis in enumerate('xyz'):
        vertex[axis] = points[:, index].astype(np.float32)
    face = np.empty(len(faces), dtype=[('vertex_indices', '<i4', (3,))])
    face['vertex_indices'] = faces
    PlyData([PlyElement.describe(vertex, 'vertex'), PlyElement.describe(face, 'face')],
            text=False, byte_order='<').write(str(path))


def _faces(mesh):
    faces = np.asarray(mesh['face']['vertex_indices'])
    return np.stack(faces) if faces.dtype.kind == 'O' else faces


@pytest.fixture
def scene(tmp_path):
    """A session with a real bag, one GLIM run, one export and one NKSR mesh."""
    session = (tmp_path/'20261007_120000_Scan_2026-10-07_12_00_00')
    session.mkdir()
    camera = yaml.safe_load((ROOT/'config/camera/d405.yaml').read_text())
    intrinsics_bytes = (ROOT/'config/calibration/d405_intrinsics.yaml').read_bytes()
    snapshot = session/'config_snapshot/calibration'
    snapshot.mkdir(parents=True)
    (snapshot/'d405_intrinsics.yaml').write_bytes(intrinsics_bytes)
    extrinsic = dict(calibrated=True, validated=True, camera_name=camera['camera_name'],
                     width=camera['width'], height=camera['height'],
                     intrinsics_sha256=hashlib.sha256(intrinsics_bytes).hexdigest(),
                     T_lidar_camera=[0, 0, 0, 0, 0, 0, 1],
                     transform_convention='p_lidar = T_lidar_camera * p_camera',
                     time_offset_sec=0.0)
    (snapshot/'d405_lidar_camera.yaml').write_text(yaml.safe_dump(extrinsic))
    (session/'active_config.json').write_text(json.dumps(dict(
        camera=camera,
        system=dict(camera=dict(enabled=True, profile='d405')),
        sensor=dict(points_topic=LIDAR_TOPIC))))

    trajectory = session/'processing/run_001/glim_dump/traj_lidar.txt'
    trajectory.parent.mkdir(parents=True)
    np.savetxt(trajectory, [[1.0, 0, 0, 0, 0, 0, 0, 1], [3.0, 2, 0, 0, 0, 0, 0, 1]])

    # Camera frames at t=1.5 (red) and t=2.5 (blue); the LiDAR sensor sits at the
    # interpolated trajectory position and its points are already in world axes.
    frames = [(1.5, _image_message(1.5, (255, 0, 0))),
              (2.5, _image_message(2.5, (0, 0, 255)))]
    clouds = [(1.5, _lidar_message(np.array([[0.0, 0, 2.0], [0.0, 0, 2.2]]),
                                   np.array([0.5, 0.6], np.float32), [1.5, 1.5])),
              (2.5, _lidar_message(np.array([[0.0, 0, 3.0]]),
                                   np.array([0.7], np.float32), [2.5]))]
    _write_bag(session/'raw_bag', frames, clouds)

    # Master world points: (0.5,0,2.0) occluding (0.5,0,2.2), and (1.5,0,3.0).
    world = np.array([[0.5, 0, 2.0], [0.5, 0, 2.2], [1.5, 0, 3.0]], np.float32)
    export = session/'exports/run_001_abcd1234.ply'
    export.parent.mkdir(parents=True)
    _write_points_ply(export, world, np.array([0.5, 0.6, 0.7], np.float32))
    run = session/'reconstruction/run_abc123abc123'
    (run/'output').mkdir(parents=True)
    (run/'job.json').write_text(json.dumps(dict(
        state='PREPARED', created_at='2026-10-07',
        trajectory='processing/run_001/glim_dump/traj_lidar.txt')))
    (run/'mesh_job.json').write_text(json.dumps(dict(state='COMPLETED')))
    _write_mesh_ply(run/'output/mesh.ply', world, np.array([[0, 1, 2]], np.int32))
    return session, export, run/'output/mesh.ply', world


def test_cli_colorizes_and_transfers_without_touching_inputs(scene, tmp_path):
    session, export, mesh, world = scene
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (export, mesh, session/'raw_bag/metadata.yaml')
              if path.is_file()}
    command = [sys.executable, str(SCRIPT), '--session', str(session),
               '--voxel-size', '0.01', '--validation-frames', '2',
               '--max-color-observations-per-voxel', '2',
               '--transfer-glim', '--glim-ply', 'auto',
               '--transfer-nksr', '--nksr-mesh', 'auto']
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    output = Path(result.stdout.strip().splitlines()[-1])
    assert output.is_dir() and output.name.startswith('run_')

    # Master cloud: standard PLY, occlusion applied, fallback for uncolored points.
    vertex = PlyData.read(str(output/'output/colored_points.ply'))['vertex'].data
    assert list(vertex.dtype.names) == ['x', 'y', 'z', 'red', 'green', 'blue', 'intensity']
    np.testing.assert_allclose(np.column_stack([vertex[axis] for axis in 'xyz']), world)
    assert list(vertex['red']) == [255, 128, 0]
    assert list(vertex['blue']) == [0, 128, 255]
    np.testing.assert_allclose(vertex['intensity'], [0.5, 0.6, 0.7])
    data = np.load(output/'output/colored_points.npz', allow_pickle=False)
    assert data['color_count'].tolist() == [1, 0, 1]
    assert np.isnan(data['Cd'][1]).all()
    assert data['color_confidence'][1] == 0.0

    stats = json.loads((output/'output/colorization_stats.json').read_text())
    assert stats['raw_lidar_observations'] == 3
    assert stats['occlusion_rejected'] == 1
    assert stats['colored_voxels'] == 2 and stats['uncolored_voxels'] == 1
    assert stats['observations_dropped_due_to_per_voxel_limit'] == 0
    assert stats['max_observations_per_voxel_limit'] == 2

    # Lineage: the export of run_001 and the reconstruction of its trajectory.
    metadata = json.loads((output/'metadata.json').read_text())
    assert metadata['source_processing_run'] == 'run_001'
    assert metadata['source_glim_ply'] == str(export)
    assert metadata['source_reconstruction_run'] == 'run_abc123abc123'
    assert metadata['source_nksr_mesh'] == str(mesh)
    assert metadata['source_bag'] == str(session/'raw_bag')
    assert metadata['raw_bag'] == str(session/'raw_bag')
    assert metadata['image_topic'] == IMAGE_TOPIC
    assert metadata['camera_info_topic'] == '/camera/camera_info'
    assert metadata['camera_calibration'].endswith('d405_lidar_camera.yaml')
    # The occluded GLIM/NKSR vertex has no valid master color, so it stays neutral.
    assert metadata['glim_transfer']['coverage_percent'] == pytest.approx(200/3)
    assert metadata['glim_transfer']['vertices_colored'] == 2
    assert metadata['glim_transfer']['vertices_uncolored'] == 1
    assert metadata['glim_transfer']['max_transfer_radius'] == 0.025
    assert metadata['nksr_transfer']['coverage_percent'] == pytest.approx(200/3)
    assert len(metadata['validation']) == 2

    # GLIM geometry and attributes survive the transfer bit for bit.
    source = PlyData.read(str(export))['vertex'].data
    colored = PlyData.read(str(output/'output/glim_colored.ply'))['vertex'].data
    for name in source.dtype.names:
        np.testing.assert_array_equal(np.asarray(colored[name]), np.asarray(source[name]))
    assert list(colored.dtype.names)[-3:] == ['red', 'green', 'blue']
    assert list(colored['red']) == [255, 128, 0]

    # NKSR vertices and topology are unchanged.
    mesh_source = PlyData.read(str(mesh))
    mesh_colored = PlyData.read(str(output/'output/nksr_colored.ply'))
    np.testing.assert_array_equal(
        np.column_stack([mesh_colored['vertex'][axis] for axis in 'xyz']),
        np.column_stack([mesh_source['vertex'][axis] for axis in 'xyz']))
    np.testing.assert_array_equal(_faces(mesh_colored), _faces(mesh_source))

    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
    assert (output/'calibration_snapshot.yaml').is_file()


def test_cli_reports_an_ambiguous_export_without_guessing(scene):
    session, export, _, _ = scene
    (session/'exports/run_001_ffff9999.ply').write_bytes(export.read_bytes())
    result = subprocess.run([sys.executable, str(SCRIPT), '--session', str(session),
                             '--transfer-glim', '--glim-ply', 'auto'],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 1
    assert 'Multiple GLIM exports exist for run_001' in result.stderr
