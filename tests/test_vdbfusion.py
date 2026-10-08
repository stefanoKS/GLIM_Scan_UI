"""VDBFusion engine tests: settings, motion-aware origins, streaming and mesh output.

The real native library is exercised through the isolated worker interpreter, so a
missing installation skips those tests instead of pretending to pass.
"""
import json
import math
import subprocess
from pathlib import Path
import numpy as np
import pytest

from factory_mapping import vdbfusion as V
from factory_mapping.engines import (DEFAULT_ALGORITHM, NKSR, PREPARED_MARKERS, VDBFUSION,
                                     normalize_algorithm, process_key)
from factory_mapping.nksr_mesh import inspect_mesh


def translating_trajectory(steps=21, speed=1.0, rotation_deg=0.0, scale_quaternion=1.0):
    """Analytical trajectory: constant velocity along +x with a constant yaw rate.

    ``rotation_deg`` is the total yaw applied over the whole trajectory, so pose
    ``i`` has yaw ``rotation_deg * i / (steps - 1)``.
    """
    stamps = np.arange(steps, dtype=np.float64)
    xyz = np.column_stack((speed * stamps, np.zeros(steps), np.zeros(steps)))
    yaw = np.radians(rotation_deg) * np.arange(steps) / max(steps - 1, 1)
    quaternions = scale_quaternion * np.column_stack(
        (np.zeros(steps), np.zeros(steps), np.sin(yaw / 2.0), np.cos(yaw / 2.0)))
    return np.column_stack((stamps, xyz, quaternions))


# --------------------------------------------------------------------------- #
# GATE 6: settings, presets and physical validation
# --------------------------------------------------------------------------- #

def test_defaults_presets_and_truncation_are_independent():
    assert V.DEFAULT_VOXEL_SIZE_M == 0.02 and V.DEFAULT_SDF_TRUNC_M == 0.06
    assert V.DEFAULT_SPACE_CARVING is False
    defaults = V.resolve_settings({})
    assert (defaults['voxel_size_m'], defaults['sdf_trunc_m']) == (0.02, 0.06)
    assert defaults['origin_error_budget_m'] == 0.02  # one TSDF voxel, conservatively
    for preset, (voxel, trunc) in [('fast', (0.02, 0.06)), ('detailed', (0.01, 0.03)),
                                   ('experimental', (0.005, 0.015))]:
        resolved = V.resolve_settings({'preset': preset})
        assert (resolved['voxel_size_m'], resolved['sdf_trunc_m']) == (voxel, trunc)
    # Truncation is an independent control, not a fixed 3x multiple of the voxel.
    custom = V.resolve_settings({'voxel_size_m': 0.02, 'sdf_trunc_m': 0.2})
    assert custom['sdf_trunc_m'] == 0.2
    assert V.resolve_settings({'space_carving': True})['space_carving'] is True


@pytest.mark.parametrize('settings', [
    {'voxel_size_m': 0}, {'voxel_size_m': -0.02}, {'voxel_size_m': float('nan')},
    {'voxel_size_m': float('inf')}, {'voxel_size_m': 5.0}, {'voxel_size_m': 0.0001},
    {'sdf_trunc_m': 0}, {'sdf_trunc_m': -1}, {'sdf_trunc_m': float('nan')},
    {'sdf_trunc_m': 0.01},  # below the voxel size: no surface band exists
    {'sdf_trunc_m': 10.0}, {'preset': 'ultra'}, {'mesh_output_mode': 'tiles'},
    {'space_carving': 'yes'}, {'origin_error_budget_m': 0.5}, {'origin_error_budget_m': 0},
    {'batch_points': 10}, {'batch_points': 10 ** 9},
])
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        V.resolve_settings(settings)


def test_region_of_interest_validation():
    resolved = V.resolve_settings({'roi_min_m': [-5, -5, 0], 'roi_max_m': [5, 5, 3]})
    assert resolved['roi_min_m'] == [-5.0, -5.0, 0.0] and resolved['roi_max_m'] == [5.0, 5.0, 3.0]
    for bad in ({'roi_min_m': [0, 0, 0]}, {'roi_max_m': [1, 1, 1]},
                {'roi_min_m': [0, 0, 0], 'roi_max_m': [0, 1, 1]},
                {'roi_min_m': [0, 0], 'roi_max_m': [1, 1, 1]},
                {'roi_min_m': [0, 0, 0], 'roi_max_m': [1, 1, float('nan')]}):
        with pytest.raises(ValueError):
            V.resolve_settings(bad)


def test_engine_identifiers_and_legacy_default():
    assert DEFAULT_ALGORITHM == NKSR
    assert normalize_algorithm(None) == NKSR
    assert normalize_algorithm('vdbfusion') == VDBFUSION
    assert process_key(VDBFUSION) == 'vdbfusion' and process_key(NKSR) == 'nksr'
    with pytest.raises(ValueError):
        normalize_algorithm('poisson')
    assert PREPARED_MARKERS[NKSR] != PREPARED_MARKERS[VDBFUSION]


# --------------------------------------------------------------------------- #
# GATE 4: motion-aware sensor origins
# --------------------------------------------------------------------------- #

def test_motion_aware_grouping_covers_every_observation_within_budget():
    rng = np.random.default_rng(7)
    origins = np.cumsum(rng.normal(scale=0.01, size=(4000, 3)), axis=0)
    budget = 0.05
    covered = np.zeros(len(origins), dtype=bool)
    groups = []
    for start, stop, origin, error in V.motion_aware_groups(origins, budget):
        assert stop > start and error <= budget
        # The representative is a real trajectory origin, never a fabricated average.
        assert origin is not None and np.allclose(origin, origins[start])
        assert np.max(np.linalg.norm(origins[start:stop] - origin, axis=1)) <= budget
        covered[start:stop] = True
        groups.append((start, stop))
    assert covered.all(), 'grouping must never discard an observation'
    assert [stop for start, stop in groups[:-1]] == [start for start, stop in groups[1:]], 'groups are contiguous'
    assert len(groups) > 1, 'a moving sensor must be split into several groups'


def test_motion_aware_grouping_splits_when_movement_exceeds_the_budget():
    origins = np.zeros((10, 3))
    origins[5:] = 1.0
    groups = list(V.motion_aware_groups(origins, 0.02))
    assert [len(range(start, stop)) for start, stop, _, _ in groups] == [5, 5]
    worst, count = V.max_origin_error(origins, 0.02)
    assert count == 2 and worst == 0.0
    # A budget below every step yields single-observation groups, so nothing is merged.
    stepping = np.arange(10, dtype=np.float64)[:, None]*np.ones((1, 3))
    assert [stop-start for start, stop, _, _ in V.motion_aware_groups(stepping, 1e-9)] == [1]*10


def test_origin_budget_scales_with_tsdf_resolution():
    fine = V.resolve_settings({'voxel_size_m': 0.005})
    coarse = V.resolve_settings({'voxel_size_m': 0.05})
    assert fine['origin_error_budget_m'] == 0.005 and coarse['origin_error_budget_m'] == 0.05
    with pytest.raises(ValueError):
        V.resolve_settings({'voxel_size_m': 0.01, 'origin_error_budget_m': 0.02})


def test_world_points_and_origins_match_analytical_values():
    """A synthetic translating/rotating trajectory must reproduce closed-form results."""
    from factory_mapping.reconstruction import transform_points
    trajectory = translating_trajectory(steps=11, speed=2.0, rotation_deg=90.0)
    # One observation on the sensor's forward axis at each pose, taken exactly at pose times.
    local = np.array([[1.0, 0.0, 0.0]])
    xyz = np.repeat(local, 11, axis=0)
    stamps = trajectory[:, 0].copy()
    world, origins, valid = transform_points(xyz, stamps, trajectory)
    assert valid.all() and len(world) == 11
    # Yaw rate 90 deg over 10 s, so pose i yaw is 9 degrees * i; the point is 1 m along +x locally.
    for index in range(11):
        yaw = math.radians(9.0 * index)
        expected_world = np.array([2.0 * index + math.cos(yaw), math.sin(yaw), 0.0])
        expected_origin = np.array([2.0 * index, 0.0, 0.0])
        assert np.allclose(world[index], expected_world, atol=1e-9), (index, world[index], expected_world)
        assert np.allclose(origins[index], expected_origin, atol=1e-9)
    # The origins are the trajectory positions, never a fabricated [0, 0, 0].
    assert not np.allclose(origins, 0.0)


def test_interpolation_between_poses_uses_the_real_origin():
    from factory_mapping.reconstruction import transform_points
    trajectory = translating_trajectory(steps=3, speed=1.0)
    world, origins, valid = transform_points(np.zeros((1, 3)), np.array([0.5]), trajectory)
    assert valid.all()
    assert np.allclose(origins[0], [0.5, 0.0, 0.0]), origins
    assert np.allclose(world[0], [0.5, 0.0, 0.0])


def test_trajectory_validation_rejects_invalid_input(tmp_path):
    good = translating_trajectory(steps=5)
    path = tmp_path/'traj_lidar.txt'
    np.savetxt(path, good)
    trajectory, report = V.load_trajectory(path)
    assert report['poses'] == 5 and report['quaternion_normalized'] is False
    assert np.allclose(np.linalg.norm(trajectory[:, 4:8], axis=1), 1.0)
    scaled = tmp_path/'scaled_traj.txt'
    np.savetxt(scaled, translating_trajectory(steps=5, scale_quaternion=1.5))
    normalized, scaled_report = V.load_trajectory(scaled)
    assert scaled_report['quaternion_normalized'] is True
    assert np.allclose(np.linalg.norm(normalized[:, 4:8], axis=1), 1.0)
    assert report['path_length_m'] == 4.0 and report['duration_s'] == 4.0

    backwards = good.copy()
    backwards[2, 0] = 1.0  # timestamp goes backwards
    cases = {'non_monotonic': backwards, 'short': good[:1],
             'zero_quaternion': np.column_stack((good[:, :4], np.zeros((5, 4)))),
             'not_finite': np.column_stack((good[:, :7], np.full(5, np.nan))),
             'wrong_width': good[:, :6]}
    for name, values in cases.items():
        np.savetxt(path, values)
        with pytest.raises(ValueError):
            V.load_trajectory(path)


def test_observations_outside_the_trajectory_are_reported_not_clamped():
    trajectory = translating_trajectory(steps=5)
    frames = [(np.zeros((3, 3)), np.ones(3, dtype=np.float32), np.array([-10.0, 1.0, 99.0]))]
    counters = {}
    batch = next(V.batch_world_observations(frames, trajectory, V.MIN_BATCH_POINTS, counters=counters))
    assert len(batch['points']) == 1, 'only the in-range observation is transformed'
    assert counters['outside_trajectory'] == 2, 'out-of-range observations are counted, not clamped'
    assert counters['frames'] == 1 and counters['observations'] == 1


# --------------------------------------------------------------------------- #
# GATE 5: bounded streaming
# --------------------------------------------------------------------------- #

def test_batches_are_bounded_and_deterministic():
    trajectory = translating_trajectory(steps=6)
    rng = np.random.default_rng(3)
    frame_points = 5000
    frames = [(rng.normal(scale=0.5, size=(frame_points, 3)),
               rng.random(frame_points).astype(np.float32),
               np.full(frame_points, float(index))) for index in range(6)]
    limit = 12000
    batches = list(V.batch_world_observations(frames, trajectory, limit))
    assert len(batches) == 2, 'batches are flushed at the bounded limit'
    sizes = [len(batch['points']) for batch in batches]
    assert max(sizes) < 2 * limit, sizes
    for batch in batches:
        assert len(batch['points']) == len(batch['origins']) == len(batch['intensity']) == len(batch['timestamps'])
        assert batch['points'].dtype == np.float64
        assert np.isfinite(batch['points']).all()
    assert sum(sizes) == 6 * frame_points, 'no observation is dropped'


def test_preflight_reports_real_resources_and_a_bounded_estimate(tmp_path):
    report = V.preflight(tmp_path, V.resolve_settings({}), observed_points=90_000_000,
                         observed_bbox=([-88.0, -4.0, -1.0], [-17.0, 68.0, 11.0]))
    assert report['free_ram_bytes'] > 0 and report['free_disk_bytes'] > 0
    assert report['disk_path'] == str(tmp_path)
    # The estimate is capped by the observation count, not by the whole bounding box.
    assert report['estimated_tsdf_voxels'] == 90_000_000
    assert report['band_voxels_upper_bound'] > report['estimated_tsdf_voxels']
    assert 0 < report['estimated_tsdf_bytes'] <= 90_000_000 * V.TSDF_BYTES_PER_TOUCHED_VOXEL
    assert 'not memory bounded' in report['estimate_note']


def run_isolated(vdbfusion_python, code, tmp_path):
    """Run a snippet in the isolated VDBFusion interpreter; return its parsed JSON."""
    root = Path(__file__).resolve().parents[1]
    script = tmp_path/'isolated_probe.py'
    script.write_text('import json,sys\nsys.path.insert(0,str(sys.argv[1]))\n'+code)
    result = subprocess.run([str(vdbfusion_python), str(script), str(root/'ui/backend')],
                            capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_memory_budget_failure_is_clean(tmp_path, vdbfusion_python):
    """A soft memory budget fails the run instead of lowering the requested resolution.

    The native library lives only in the isolated interpreter, so this runs there.
    """
    trajectory = translating_trajectory(steps=5)
    bag = write_synthetic_bag(tmp_path, trajectory, points_per_frame=2000, frames=3)
    traj = tmp_path/'traj_lidar.txt'
    np.savetxt(traj, trajectory)
    result = subprocess.run(
        [str(vdbfusion_python), str(Path(__file__).resolve().parents[1]/'tools/vdbfusion_worker.py'),
         '--bag', str(bag), '--trajectory', str(traj), '--output', str(tmp_path/'mesh.ply'),
         '--progress', str(tmp_path/'progress.json'), '--voxel-size', '0.02', '--memory-budget-gib', '0.001'],
        env=vdbfusion_env(), capture_output=True, text=True, timeout=600)
    assert result.returncode != 0
    failure = json.loads(result.stderr.strip().splitlines()[-1])
    assert failure['error_type'] == 'MEMORY_BUDGET_EXCEEDED'
    assert 'not changed' in failure['message']
    assert not (tmp_path/'mesh.ply').exists()


# --------------------------------------------------------------------------- #
# Synthetic bag helpers and end-to-end native runs
# --------------------------------------------------------------------------- #

def write_synthetic_bag(directory, trajectory, points_per_frame=2000, frames=4, plane_z=3.0, seed=11):
    """Write a real ROS 2 sqlite3 bag of synthetic LiDAR frames on /livox/lidar."""
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import PointCloud2, PointField
    rng = np.random.default_rng(seed)
    # rosbag2 refuses to write into an existing directory, so the bag root is created by it.
    path = Path(directory)/'raw_bag'
    Path(directory).mkdir(parents=True, exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    writer.create_topic(rosbag2_py.TopicMetadata(name='/livox/lidar', type='sensor_msgs/msg/PointCloud2',
                                                 serialization_format='cdr'))
    stamps = np.linspace(float(trajectory[0, 0]), float(trajectory[-1, 0]), frames)
    for stamp in stamps:
        cloud = np.zeros(points_per_frame, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                                  ('intensity', '<f4'), ('timestamp', '<f8')])
        cloud['x'] = rng.uniform(-2.0, 2.0, points_per_frame)
        cloud['y'] = rng.uniform(-2.0, 2.0, points_per_frame)
        cloud['z'] = plane_z
        cloud['intensity'] = 1.0
        cloud['timestamp'] = stamp * 1e9
        message = PointCloud2()
        message.header.frame_id = 'livox_frame'
        message.height = 1
        message.width = points_per_frame
        message.is_bigendian = False
        message.point_step = cloud.dtype.itemsize
        message.row_step = message.point_step*points_per_frame
        message.fields = [PointField(name=name, offset=int(cloud.dtype.fields[name][1]),
                                     datatype=PointField.FLOAT32, count=1)
                          for name in ('x', 'y', 'z', 'intensity')]
        message.fields[-1].datatype = PointField.FLOAT32
        message.fields.append(PointField(name='timestamp', offset=int(cloud.dtype.fields['timestamp'][1]),
                                        datatype=PointField.FLOAT64, count=1))
        message.data = cloud.tobytes()
        writer.write('/livox/lidar', serialize_message(message), int(stamp*1e9))
    del writer
    return path


def test_synthetic_bag_streams_and_integrates_motion_aware(tmp_path, vdbfusion_python):
    """The real native integrator consumes bounded batches of a real synthetic bag."""
    trajectory = translating_trajectory(steps=9, speed=3.0)
    bag = write_synthetic_bag(tmp_path, trajectory, points_per_frame=4000, frames=5)
    traj = tmp_path/'traj_lidar.txt'
    np.savetxt(traj, trajectory)
    report = run_isolated(vdbfusion_python, f"""
from pathlib import Path
import numpy as np
from factory_mapping import vdbfusion as V
trajectory, _ = V.load_trajectory(Path({str(traj)!r}))
settings = V.resolve_settings(dict(voxel_size_m=0.05, sdf_trunc_m=0.15))
volume, stats = V.integrate_bag(Path({str(bag)!r}), trajectory, '/livox/lidar', settings)
vertices, faces, extraction = V.extract_and_mask(volume)
mesh = V.validate_and_report(vertices, faces, settings)
print(json.dumps(dict(stats={{k: v for k, v in stats.items() if not isinstance(v, dict)}},
                      extraction=extraction,
                      bounding_box_max=mesh['bounding_box_max'], units=mesh['units'],
                      coordinate_system=mesh['coordinate_system'],
                      face_count=int(len(faces)),
                      index_max=int(faces.max()), vertex_count=int(len(vertices)))))
""", tmp_path)
    stats = report['stats']
    assert stats['raw_frames'] == 5
    assert stats['integrated_observations'] == 5*4000
    assert stats['origin_groups'] >= 5, 'a moving sensor needs several origin groups'
    assert stats['max_origin_error_m'] <= stats['origin_error_budget_m']
    assert stats['outside_trajectory'] == 0
    assert report['face_count'] > 0 and report['index_max'] < report['vertex_count']
    assert report['units'] == 'meters' and report['coordinate_system'] == 'GLIM_world'
    # The synthetic plane sits at z = 3 m in the sensor frame; the sensor stays at z = 0.
    assert abs(report['bounding_box_max'][2] - 3.0) < 0.2


def vdbfusion_env():
    from factory_mapping import vdbfusion_jobs
    return vdbfusion_jobs.worker_environment()


def test_real_worker_check_and_mesh_output_modes(tmp_path, vdbfusion_python):
    from factory_mapping import vdbfusion_jobs
    trajectory = translating_trajectory(steps=9, speed=3.0)
    bag = write_synthetic_bag(tmp_path, trajectory, points_per_frame=3000, frames=4)
    traj = tmp_path/'traj_lidar.txt'
    np.savetxt(traj, trajectory)
    output = tmp_path/'output'/'mesh.ply'
    metadata = tmp_path/'output'/'vdbfusion_metadata.json'
    progress = tmp_path/'vdbfusion_progress.json'
    result = subprocess.run(
        [str(vdbfusion_python), str(vdbfusion_jobs.worker_path()),
         '--bag', str(bag), '--trajectory', str(traj), '--output', str(output),
         '--metadata', str(metadata), '--progress', str(progress),
         '--voxel-size', '0.05', '--sdf-trunc', '0.15', '--mesh-output-mode', 'both'],
        env=vdbfusion_env(), capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    stats = inspect_mesh(output)
    assert stats['face_count'] > 0 and stats['vertex_count'] > 0
    report = json.loads(metadata.read_text())
    assert report['engine'] == 'vdbfusion' and report['validation_status'] == 'PASS'
    assert report['mesh_output_mode'] == 'both'
    assert report['settings']['voxel_size_m'] == 0.05 and report['settings']['sdf_trunc_m'] == 0.15
    assert report['integration']['max_origin_error_m'] <= report['settings']['origin_error_budget_m']
    # Separate meshes are spatial sections of the one fused mesh; the validator re-reads them.
    manifest = json.loads((tmp_path/'output'/'mesh_chunks'/'chunks.json').read_text())
    assert manifest['export_strategy'] == 'spatial_split_of_final_mesh'
    assert manifest['reconstruction_mode'] == 'vdbfusion_fused_tsdf'
    assert manifest['source_faces'] == stats['face_count']
    assert sum(chunk['faces'] for chunk in manifest['chunks']) == stats['face_count']
    assert vdbfusion_jobs.validate_completed(tmp_path/'output', 0)['face_count'] == stats['face_count']


def test_real_worker_check_reports_a_real_mesh(tmp_path, vdbfusion_python):
    from factory_mapping import vdbfusion_jobs
    health = tmp_path/'health.json'
    result = subprocess.run([str(vdbfusion_python), str(vdbfusion_jobs.worker_path()),
                             '--check', '--health-output', str(health)],
                            env=vdbfusion_env(), capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(health.read_text())
    # Import success alone is not sufficient: a real nonempty mesh is validated.
    assert report['status'] == 'READY' and report['smoke_passed'] is True
    assert report['check_vertices'] > 0 and report['check_triangles'] > 0
    assert report['vdbfusion_version'] and report['numpy_version'].startswith('1.')
    assert report['native_backend'].startswith('upstream PRBonn VDBFusion')


def test_worker_rejects_empty_tsdf_and_bad_input(tmp_path, vdbfusion_python):
    from factory_mapping import vdbfusion_jobs
    traj = tmp_path/'traj_lidar.txt'
    np.savetxt(traj, translating_trajectory(steps=4))
    empty = tmp_path/'empty_bag'
    empty.mkdir()
    result = subprocess.run([str(vdbfusion_python), str(vdbfusion_jobs.worker_path()),
                             '--bag', str(empty), '--trajectory', str(traj),
                             '--output', str(tmp_path/'mesh.ply')],
                            env=vdbfusion_env(), capture_output=True, text=True, timeout=300)
    assert result.returncode != 0
    failure = json.loads(result.stderr.strip().splitlines()[-1])
    assert failure['error_type'] == 'RAW_BAG_UNREADABLE'
    assert not (tmp_path/'mesh.ply').exists()
