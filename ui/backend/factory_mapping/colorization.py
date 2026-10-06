"""Offline D405 projection validation; never modifies recorded/prepared inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import uuid
import warnings

import numpy as np
import yaml

from .calibration_data import parse_intrinsics, transform
from .camera_config import config_path
from .reconstruction import normalize_quaternion, rotate_vectors, transform_points
from .reconstruction_jobs import safe_trajectory
from .storage import atomic_json, read_json


def session_calibration(session, allow_unvalidated=False):
    config = json.loads((session/'active_config.json').read_text())
    camera = config['camera']
    if not config['system']['camera'].get('enabled'):
        raise ValueError('Session did not record RGB')
    if camera.get('source') != 'realsense' or 'D405' not in camera.get('model', ''):
        raise ValueError('Projection validation requires a recorded D405 profile')

    def snapshot(key):
        # Validate the original config path, then resolve ONLY inside this session.
        relative = config_path(session, camera[key]).relative_to(session/'config')
        path = session/'config_snapshot'/relative
        if not path.is_file() or (session/'config_snapshot').resolve() not in path.resolve().parents:
            raise ValueError(f'Missing session calibration snapshot: {path}')
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    ipath, ihash = snapshot('intrinsics_file')
    epath, ehash = snapshot('extrinsics_file')
    intr = parse_intrinsics(yaml.safe_load(ipath.read_text()), camera)
    # Published CameraInfo is zero-distortion; factory_calibration coefficients
    # describe the SDK source and must never be reapplied to rectified RGB.
    if intr['model'] != 'plumb_bob' or any(intr['distortion']):
        raise ValueError('Expected rectified D405 intrinsics with zero distortion')
    ext = yaml.safe_load(epath.read_text())
    if not isinstance(ext, dict) or ext.get('calibrated') is not True:
        raise ValueError('Session D405 extrinsic is missing or uncalibrated')
    if ext.get('validated') is not True:
        if not allow_unvalidated:
            raise ValueError('Calibration is unvalidated; use --allow-unvalidated-calibration to inspect it')
        warnings.warn('Using unvalidated session calibration for projection validation', stacklevel=2)
    if ext.get('transform_convention') != 'p_lidar = T_lidar_camera * p_camera':
        raise ValueError('Missing or unsupported extrinsic transform convention')
    pose = transform(ext.get('T_lidar_camera'))
    if ext.get('intrinsics_sha256') != ihash:
        raise ValueError('Extrinsic intrinsics hash does not match session snapshot')
    for key in ('camera_name', 'width', 'height'):
        if ext.get(key) != camera[key]:
            raise ValueError(f'Extrinsic {key} differs from recorded camera')
    offset = ext.get('time_offset_sec')
    if not isinstance(offset, (int, float)) or not np.isfinite(offset):
        raise ValueError('Extrinsic requires a finite time_offset_sec')
    if camera.get('time_offset_sec') != offset:
        raise ValueError('Session configuration and extrinsic time offsets disagree')
    return camera, intr, pose, dict(
        camera_profile=config['system']['camera'].get('profile', 'd405'),
        camera_configuration={key: camera.get(key) for key in
                              ('model', 'serial_number', 'camera_name', 'frame_id', 'fps')},
        intrinsics_file=str(ipath), intrinsics_sha256=ihash,
        extrinsics_file=str(epath), extrinsics_sha256=ehash,
        T_lidar_camera=pose, transform_convention=ext['transform_convention'],
        calibration_validated=ext.get('validated') is True,
        calibration_time_offset_sec=offset)


def prepared_input(session, run_id=None):
    candidates = []
    for run in (session/'reconstruction').glob('run_*'):
        if run_id is not None and run.name != run_id:
            continue
        job = read_json(run/'job.json', {})
        if job.get('state') in ('PREPARED', 'completed') and (run/'input/nksr_input.npz').is_file():
            candidates.append((job.get('created_at', ''), run.name, run, job))
    if not candidates:
        raise ValueError('No completed reconstruction with input/nksr_input.npz; prepare the session first')
    _, _, run, job = max(candidates)
    trajectory_path = safe_trajectory(session, job['trajectory'])
    trajectory = np.loadtxt(trajectory_path, ndmin=2)
    if (trajectory.shape[1] != 8 or len(trajectory) < 2 or
            not np.isfinite(trajectory).all() or np.any(np.diff(trajectory[:, 0]) <= 0) or
            np.any(np.linalg.norm(trajectory[:, 4:8], axis=1) == 0)):
        raise ValueError('Trajectory requires increasing timestamps and finite XYZ / XYZW poses')
    path = run/'input/nksr_input.npz'
    with np.load(path, allow_pickle=False) as data:
        points = data['points']
        timestamps = data['timestamps']
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points) or
            timestamps.shape != (len(points),) or not np.isfinite(points).all() or
            not np.isfinite(timestamps).all()):
        raise ValueError('Prepared points/timestamps are empty or invalid')
    return path, points, trajectory_path, trajectory


def camera_pose(trajectory, camera_time, offset, extrinsic):
    lidar_time = camera_time + offset  # docs/camera_calibration.md: t_lidar = t_camera + offset.
    # Reuse reconstruction's interpolation/SLERP by transforming an origin and basis.
    basis = np.vstack((np.zeros(3), np.eye(3)))
    world, _, valid = transform_points(basis, np.full(4, lidar_time), trajectory)
    if not valid.all():
        raise ValueError(f'Corrected image timestamp {lidar_time:.9f} is outside the trajectory')
    rotation = (world[1:] - world[0]).T
    camera_rotation = rotate_vectors(np.tile(normalize_quaternion(np.asarray(extrinsic[3:], dtype=float)), (3, 1)), np.eye(3)).T
    # Verified stored convention: p_lidar = T_lidar_camera * p_camera.
    # Thus T_world_camera = T_world_lidar @ T_lidar_camera (XYZ / XYZW).
    return rotation @ camera_rotation, rotation @ extrinsic[:3] + world[0], lidar_time


def project_points(points, rotation, origin, intrinsics, width, height, max_range):
    relative = points.astype(np.float64) - origin
    relative = relative[np.einsum('ij,ij->i', relative, relative) <= max_range**2]
    camera = relative @ rotation  # inverse rigid pose, for row-vector points
    camera = camera[camera[:, 2] > 0]
    fx, fy, cx, cy = intrinsics
    pixels = camera[:, :2] / camera[:, 2:3] * [fx, fy] + [cx, cy]
    valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
             (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    pixels, depth = pixels[valid].astype(int), camera[valid, 2]
    # Simple per-pixel z-buffer; a radius-one display dot is not an occlusion model.
    order = np.argsort(depth)
    _, first = np.unique(pixels[order, 1]*width + pixels[order, 0], return_index=True)
    keep = order[first]
    return pixels[keep], depth[keep]


def bag_images(bag, topic):
    # Same lazy ROS reader as reconstruction; no ROS/hardware needed for math tests.
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    topics = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if topics.get(topic) != 'sensor_msgs/msg/Image':
        raise ValueError(f'No recorded RGB Image topic: {topic}')
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    while reader.has_next():
        _, data, bag_ns = reader.read_next()
        msg = deserialize_message(data, Image)
        # Use the recorded image header on the ROS/LiDAR clock, not recorder arrival.
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        yield stamp, bag_ns, msg


def image_bgr(msg):
    if msg.encoding not in ('rgb8', 'bgr8') or msg.step < msg.width*3:
        raise ValueError('Expected recorded rgb8/bgr8 image')
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    image = data[:, :msg.width*3].reshape(msg.height, msg.width, 3)
    return (image[:, :, ::-1] if msg.encoding == 'rgb8' else image).copy()


def validate_session(session, frames=8, max_range=20., time_offset=None,
                     allow_unvalidated_calibration=False, run=None):
    import cv2
    session = Path(session).resolve()
    if not 1 <= frames <= 100 or not np.isfinite(max_range) or max_range <= 0:
        raise ValueError('Use 1–100 frames and a finite positive maximum range')
    camera, intr, extrinsic, metadata = session_calibration(session, allow_unvalidated_calibration)
    offset = metadata['calibration_time_offset_sec'] if time_offset is None else time_offset
    if not np.isfinite(offset):
        raise ValueError('Time offset must be finite')
    path, points, trajectory_path, trajectory = prepared_input(session, run)
    bag = session/'raw_bag'
    if not (bag/'metadata.yaml').is_file():
        raise ValueError('Session raw RGB rosbag is missing')
    # Two streaming passes: retain only indices/timestamps, decode only selected RGB.
    eligible = [(i, stamp) for i, (stamp, _, _) in enumerate(bag_images(bag, camera['image_topic']))
                if trajectory[0, 0] <= stamp + offset <= trajectory[-1, 0]]
    if not eligible:
        raise ValueError('No recorded RGB frames overlap the trajectory at this time offset')
    selected = {eligible[i][0] for i in np.linspace(0, len(eligible)-1, min(frames, len(eligible)), dtype=int)}
    output = path.parents[1]/'colorization/projection_validation'/('run_'+uuid.uuid4().hex[:12])
    output.mkdir(parents=True)
    metadata.update(image_topic=camera['image_topic'], image_resolution=[camera['width'], camera['height']],
                    intrinsics=intr, time_offset_sec=offset, time_offset_override=time_offset,
                    trajectory_path=str(trajectory_path), prepared_points_path=str(path),
                    number_source_points=len(points), max_range_m=max_range,
                    timestamp_source='recorded Image.header.stamp', frames=[])
    for index, (stamp, bag_ns, msg) in enumerate(bag_images(bag, camera['image_topic'])):
        if index not in selected:
            continue
        if (msg.width, msg.height) != (camera['width'], camera['height']):
            raise ValueError('Recorded image dimensions differ from session intrinsics')
        rotation, origin, lidar_time = camera_pose(trajectory, stamp, offset, extrinsic)
        pixels, depth = project_points(points, rotation, origin, intr['intrinsics'],
                                       msg.width, msg.height, max_range)
        image = image_bgr(msg)
        if len(depth):
            colors = cv2.applyColorMap(np.clip(depth/max_range*255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            for (u, v), color in zip(pixels, colors[:, 0]):
                cv2.circle(image, (int(u), int(v)), 1, tuple(map(int, color)), -1)
        filename = f'frame_{len(metadata["frames"]):06d}.png'
        if not cv2.imwrite(str(output/filename), image):
            raise ValueError(f'Could not write {output/filename}')
        metadata['frames'].append(dict(file=filename, bag_frame_index=index, camera_timestamp=stamp,
                                       bag_timestamp_ns=bag_ns, lidar_pose_timestamp=lidar_time,
                                       number_projected=len(depth)))
        if len(metadata['frames']) == len(selected):
            break
    atomic_json(output/'metadata.json', metadata)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--max-range', type=float, default=20.)
    parser.add_argument('--time-offset', type=float)
    parser.add_argument('--run', help='Reconstruction run ID; default: latest completed preparation')
    parser.add_argument('--allow-unvalidated-calibration', action='store_true')
    args = parser.parse_args()
    try:
        print(validate_session(**vars(args)))
    except (OSError, ValueError, KeyError, TypeError, ImportError, yaml.YAMLError) as error:
        parser.exit(1, f'Projection validation failed: {error}\n')


if __name__ == '__main__':
    main()
