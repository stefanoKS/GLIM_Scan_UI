"""Offline D405 colorization and projection validation.

Never modifies recorded or prepared inputs; every output lands in a fresh
``session/colorization/run_*`` directory. The projection-validation entry point
(``validate_session``) is preserved unchanged for compatibility.
"""
import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import uuid
import warnings

import numpy as np
import yaml

from .calibration_data import atomic_yaml, parse_intrinsics, transform
from .camera_config import config_path
from .reconstruction import (DEFAULT_VOXEL_SIZE_M, normalize_quaternion,
                             read_measurements, rotate_vectors, transform_points,
                             validate_voxel_size, voxel_keys)
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


# ---------------------------------------------------------------------------
# Colorization pipeline (additive; projection validation above is unchanged).
# ---------------------------------------------------------------------------

@dataclass
class ColorizationConfig:
    """User-configurable colorization settings.

    The occlusion tolerance formula ``base + range_scale * depth`` is a
    tunable default, not a physical calibration constant.
    """
    voxel_size: float = DEFAULT_VOXEL_SIZE_M
    max_time_delta: float = 0.15
    min_depth: float = 0.0
    max_depth: float = 20.0
    occlusion_base_tolerance: float = 0.03
    occlusion_range_scale: float = 0.0075
    validation_frames: int = 0
    chunk_points: int = 1_000_000
    fallback_color: tuple = (128, 128, 128)

    def validated(self):
        value = validate_voxel_size(self.voxel_size)
        if not np.isfinite(self.max_time_delta) or self.max_time_delta <= 0:
            raise ValueError('max_time_delta must be a positive, finite duration in seconds')
        if not np.isfinite(self.min_depth) or not np.isfinite(self.max_depth) or not 0 <= self.min_depth < self.max_depth:
            raise ValueError('min_depth/max_depth must satisfy 0 <= min_depth < max_depth')
        if not np.isfinite(self.occlusion_base_tolerance) or self.occlusion_base_tolerance < 0:
            raise ValueError('occlusion_base_tolerance must be finite and nonnegative')
        if not np.isfinite(self.occlusion_range_scale) or self.occlusion_range_scale < 0:
            raise ValueError('occlusion_range_scale must be finite and nonnegative')
        if type(self.validation_frames) is not int or not 0 <= self.validation_frames <= 1000:
            raise ValueError('validation_frames must be an integer between 0 and 1000')
        if type(self.chunk_points) is not int or self.chunk_points <= 0:
            raise ValueError('chunk_points must be a positive integer')
        color = tuple(self.fallback_color)
        if len(color) != 3 or not all(type(v) is int and 0 <= v <= 255 for v in color):
            raise ValueError('fallback_color must be three integers between 0 and 255')
        self.voxel_size = value
        self.fallback_color = color
        return self

    def as_dict(self):
        return dict(voxel_size=self.voxel_size, max_time_delta=self.max_time_delta,
                    min_depth=self.min_depth, max_depth=self.max_depth,
                    occlusion_base_tolerance=self.occlusion_base_tolerance,
                    occlusion_range_scale=self.occlusion_range_scale,
                    validation_frames=self.validation_frames, chunk_points=self.chunk_points,
                    fallback_color=list(self.fallback_color))


def load_trajectory(path):
    """Load and validate a GLIM ``traj_lidar.txt`` trajectory (t,x,y,z,qx,qy,qz,qw)."""
    path = Path(path)
    trajectory = np.loadtxt(path, ndmin=2)
    if (trajectory.shape[1] != 8 or len(trajectory) < 2 or
            not np.isfinite(trajectory).all() or np.any(np.diff(trajectory[:, 0]) <= 0) or
            np.any(np.linalg.norm(trajectory[:, 4:8], axis=1) == 0)):
        raise ValueError('Trajectory requires increasing timestamps and finite XYZ / XYZW poses')
    return trajectory


def resolve_trajectory(session, explicit, run=None):
    """Resolve the final optimized trajectory used for colorization.

    Prefer an explicit path, then the trajectory referenced by the latest
    prepared reconstruction run (which may be an edited saved map), and finally
    the latest GLIM processing dump.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            candidate = session/path
            path = candidate if candidate.is_file() else path.resolve()
        if not path.is_file():
            raise ValueError(f'Trajectory not found: {explicit}')
        return path
    candidates = []
    for folder in (session/'reconstruction').glob('run_*'):
        if run is not None and folder.name != run:
            continue
        job = read_json(folder/'job.json', {})
        if job.get('state') in ('PREPARED', 'completed') and job.get('trajectory'):
            try:
                candidates.append((job.get('created_at', ''), safe_trajectory(session, job['trajectory'])))
            except ValueError:
                continue
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    processing = sorted((session/'processing').glob('*/glim_dump/traj_lidar.txt'))
    if not processing:
        raise ValueError('No trajectory found; pass --trajectory or prepare a reconstruction first')
    return safe_trajectory(session, str(processing[-1].relative_to(session)))


def resolve_lidar_topic(session, explicit):
    if explicit:
        return explicit
    config = read_json(session/'active_config.json', {})
    topic = config.get('sensor', {}).get('points_topic', '/livox/lidar')
    if not isinstance(topic, str) or not topic:
        raise ValueError('Session sensor points_topic is missing')
    return topic


def transform_world_to_camera(points, rotation, origin):
    """Row-vector world points into camera coordinates: ``P_camera = (P_world - origin) @ R``."""
    return (points.astype(np.float64) - origin) @ rotation


def project_to_pixels(points, rotation, origin, intrinsics, width, height, min_depth, max_depth):
    """Project world points and return camera coords, float pixels, depth and masks."""
    fx, fy, cx, cy = intrinsics
    camera = transform_world_to_camera(points, rotation, origin)
    depth = camera[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        pixels = camera[:, :2] / depth[:, None] * np.array([fx, fy]) + np.array([cx, cy])
    finite = np.isfinite(camera).all(axis=1) & np.isfinite(pixels).all(axis=1)
    in_front = depth > 0
    in_range = (depth >= min_depth) & (depth <= max_depth)
    in_bounds = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
                 (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    valid = finite & in_front & in_range & in_bounds
    masks = dict(finite=finite, in_front=in_front, in_range=in_range, in_bounds=in_bounds, valid=valid)
    return camera, pixels, depth, masks


def build_depth_buffer(pixels, depth, valid, height, width):
    """Nearest projected depth per rounded pixel; ``inf`` marks empty pixels."""
    buffer = np.full((height, width), np.inf, dtype=np.float32)
    if valid.any():
        u = np.rint(pixels[valid, 0]).astype(np.int64)
        v = np.rint(pixels[valid, 1]).astype(np.int64)
        np.minimum.at(buffer, (v, u), depth[valid].astype(np.float32))
    return buffer


def occlusion_keep(pixels, depth, valid, depth_buffer, base_tolerance, range_scale):
    """Keep valid projections not significantly behind the front surface."""
    keep = np.zeros(depth.shape[0], dtype=bool)
    if not valid.any():
        return keep
    idx = np.flatnonzero(valid)
    u = np.rint(pixels[idx, 0]).astype(np.int64)
    v = np.rint(pixels[idx, 1]).astype(np.int64)
    front = depth_buffer[v, u]
    d = depth[idx]
    tolerance = base_tolerance + range_scale * d
    keep[idx] = d <= front + tolerance
    return keep


def bilinear_sample_rgb(image_bgr, u, v):
    """Bilinearly sample a BGR uint8 image and return float RGB in ``[0, 1]``."""
    height, width = image_bgr.shape[:2]
    u = np.clip(u, 0.0, width - 1.0)
    v = np.clip(v, 0.0, height - 1.0)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = np.minimum(u0 + 1, width - 1)
    v1 = np.minimum(v0 + 1, height - 1)
    du = (u - u0).astype(np.float32)[:, None]
    dv = (v - v0).astype(np.float32)[:, None]
    c00 = image_bgr[v0, u0].astype(np.float32)
    c01 = image_bgr[v0, u1].astype(np.float32)
    c10 = image_bgr[v1, u0].astype(np.float32)
    c11 = image_bgr[v1, u1].astype(np.float32)
    top = c00 * (1.0 - du) + c01 * du
    bottom = c10 * (1.0 - du) + c11 * du
    return (top * (1.0 - dv) + bottom * dv)[:, ::-1] / 255.0


def fuse_grouped_median(group_ids, rgb, n_groups):
    """Per-channel median of RGB observations grouped by voxel id.

    Returns ``(fused, counts)`` with ``fused`` shape ``[n_groups, 3]`` (NaN for
    empty groups) and ``counts`` shape ``[n_groups]``.
    """
    group_ids = np.asarray(group_ids, dtype=np.int64)
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or len(rgb) != len(group_ids):
        raise ValueError('Observation arrays must be [M,3] float RGB with matching group ids')
    if np.any(group_ids < 0) or np.any(group_ids >= n_groups):
        raise ValueError('Observation group ids are out of range')
    counts = np.bincount(group_ids, minlength=n_groups).astype(np.int64)
    fused = np.full((n_groups, 3), np.nan, dtype=np.float32)
    for channel in range(3):
        values = rgb[:, channel]
        order = np.lexsort((values, group_ids))  # group-major, value-minor
        g_sorted = group_ids[order]
        v_sorted = values[order]
        counts_sorted = np.bincount(g_sorted, minlength=n_groups).astype(np.int64)
        occupied = counts_sorted > 0
        starts = np.zeros(n_groups, dtype=np.int64)
        np.cumsum(counts_sorted[:-1], out=starts[1:])
        lower = starts[occupied] + (counts_sorted[occupied] - 1) // 2
        upper = starts[occupied] + counts_sorted[occupied] // 2
        fused[occupied, channel] = (v_sorted[lower] + v_sorted[upper]) * 0.5
    return fused, counts


def write_colored_ply(path, points, rgb_u8, intensity=None):
    """Write a portable binary little-endian PLY with standard RGB fields."""
    path = Path(path)
    points = np.asarray(points, dtype=np.float32)
    rgb_u8 = np.asarray(rgb_u8, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or rgb_u8.shape != (len(points), 3):
        raise ValueError('Colored PLY needs [N,3] points and matching [N,3] RGB')
    has_intensity = intensity is not None
    names = ['x', 'y', 'z', 'red', 'green', 'blue'] + (['intensity'] if has_intensity else [])
    formats = ['<f4', '<f4', '<f4', 'u1', 'u1', 'u1'] + (['<f4'] if has_intensity else [])
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              + ('property float intensity\n' if has_intensity else '') + 'end_header\n')
    dtype = np.dtype(list(zip(names, formats)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        for start in range(0, len(points), 250_000):
            stop = min(start + 250_000, len(points))
            records = np.empty(stop - start, dtype=dtype)
            records['x'] = points[start:stop, 0]
            records['y'] = points[start:stop, 1]
            records['z'] = points[start:stop, 2]
            records['red'] = rgb_u8[start:stop, 0]
            records['green'] = rgb_u8[start:stop, 1]
            records['blue'] = rgb_u8[start:stop, 2]
            if has_intensity:
                records['intensity'] = np.asarray(intensity[start:stop], dtype=np.float32)
            records.tofile(stream)


def write_colored_npz(path, points, cd, confidence, count, intensity=None,
                      timestamps=None, origins=None):
    data = dict(points=np.asarray(points, dtype=np.float32),
                Cd=np.asarray(cd, dtype=np.float32),
                color_confidence=np.asarray(confidence, dtype=np.float32),
                color_count=np.asarray(count, dtype=np.int32))
    if intensity is not None:
        data['intensity'] = np.asarray(intensity, dtype=np.float32)
    if timestamps is not None:
        data['timestamps'] = np.asarray(timestamps, dtype=np.float64)
    if origins is not None:
        data['sensor_origins'] = np.asarray(origins, dtype=np.float32)
    np.savez_compressed(Path(path), **data)


def render_projection_overlay(image_bgr, points, rotation, origin, intrinsics, width, height, max_range):
    """Annotate a BGR image with depth-colored projected LiDAR dots (JET)."""
    import cv2
    annotated = image_bgr.copy()
    pixels, depth = project_points(points, rotation, origin, intrinsics, width, height, max_range)
    if len(depth):
        scaled = np.clip(depth / max_range * 255, 0, 255).astype(np.uint8)
        colors = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        for (u, v), color in zip(pixels, colors[:, 0]):
            cv2.circle(annotated, (int(u), int(v)), 1, tuple(map(int, color)), -1)
    return annotated


def _voxel_grouping(points, voxel_size):
    """Representative indices and per-point voxel ids matching reconstruction sampling.

    Returns ``(representative, voxel_of_point, voxel_of_representative)`` where
    ``representative`` holds one source point index per voxel in acquisition
    order, ``voxel_of_point`` maps every source point to its voxel id, and
    ``voxel_of_representative`` maps each output position back to its voxel id.
    """
    size = validate_voxel_size(voxel_size)
    count = len(points)
    if size == 0:
        return np.arange(count), np.arange(count, dtype=np.int64), np.arange(count, dtype=np.int64)
    keys = voxel_keys(points, size)
    _, by_voxel, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    representative = np.sort(by_voxel)
    inverse = inverse.astype(np.int64)
    return representative, inverse, inverse[representative]


def colorize_session(session, trajectory=None, run=None, image_topic=None, lidar_topic=None,
                     voxel_size=DEFAULT_VOXEL_SIZE_M, max_time_delta=0.15, min_depth=0.0,
                     max_depth=20.0, occlusion_base_tolerance=0.03, occlusion_range_scale=0.0075,
                     validation_frames=0, chunk_points=1_000_000,
                     allow_unvalidated_calibration=False, progress=print, output_dir=None):
    """Run the full camera→LiDAR→master colored point cloud pipeline."""
    import cv2
    session = Path(session).resolve()
    config = ColorizationConfig(voxel_size=voxel_size, max_time_delta=max_time_delta,
                                min_depth=min_depth, max_depth=max_depth,
                                occlusion_base_tolerance=occlusion_base_tolerance,
                                occlusion_range_scale=occlusion_range_scale,
                                validation_frames=validation_frames,
                                chunk_points=chunk_points).validated()

    camera, intr, extrinsic, calibration = session_calibration(session, allow_unvalidated_calibration)
    offset = calibration['calibration_time_offset_sec']
    trajectory_path = resolve_trajectory(session, trajectory, run)
    trajectory = load_trajectory(trajectory_path)
    topic = resolve_lidar_topic(session, lidar_topic)
    bag = session/'raw_bag'
    if not (bag/'metadata.yaml').is_file():
        raise ValueError('Session raw RGB rosbag is missing')
    if image_topic is None:
        image_topic = camera['image_topic']

    progress('Interpolating trajectory and transforming raw measurements into world space')
    records = read_measurements(bag, trajectory, topic, progress)
    points = records['points']          # float64 [N,3] world
    timestamps = records['timestamps']  # float64 [N]
    intensity = records['intensity']    # float32 [N]
    origins = records['sensor_origins'] # float64 [N,3]

    progress(f'Voxel grouping {len(points):,} world-space measurements at {config.voxel_size:g} m')
    representative, voxel_of_point, voxel_of_representative = _voxel_grouping(points, config.voxel_size)
    n_voxels = len(representative)

    order = np.argsort(timestamps, kind='stable')
    sorted_timestamps = timestamps[order]
    trajectory_t0, trajectory_t1 = trajectory[0, 0], trajectory[-1, 0]

    stats = dict(input_point_count=int(len(points)), camera_frame_count=0, frames_projected=0,
                 frames_skipped=0, candidate_projections=0, behind_camera_count=0,
                 out_of_range_count=0, out_of_frame_count=0, valid_projections=0,
                 occlusion_rejected_count=0, colored_observation_count=0)

    obs_voxel, obs_rgb = [], []
    max_depth_range = config.max_depth

    def frames():
        for stamp, bag_ns, msg in bag_images(bag, image_topic):
            yield stamp, bag_ns, msg

    for stamp, bag_ns, msg in frames():
        stats['camera_frame_count'] += 1
        lidar_time = stamp + offset
        if not trajectory_t0 <= lidar_time <= trajectory_t1:
            stats['frames_skipped'] += 1
            continue
        lo = int(np.searchsorted(sorted_timestamps, lidar_time - config.max_time_delta, side='left'))
        hi = int(np.searchsorted(sorted_timestamps, lidar_time + config.max_time_delta, side='right'))
        if lo >= hi:
            stats['frames_skipped'] += 1
            continue
        selected = order[lo:hi]
        stats['frames_projected'] += 1
        stats['candidate_projections'] += len(selected)
        if (msg.width, msg.height) != (camera['width'], camera['height']):
            raise ValueError('Recorded image dimensions differ from session intrinsics')
        rotation, origin, _ = camera_pose(trajectory, stamp, offset, extrinsic)
        image = image_bgr(msg)
        width, height = msg.width, msg.height

        # Pass 1: project in chunks and build the per-frame depth buffer.
        buffers = []
        frame_valid = 0
        for start in range(0, len(selected), config.chunk_points):
            block = selected[start:start + config.chunk_points]
            _, pixels, depth, masks = project_to_pixels(points[block], rotation, origin,
                                                        intr['intrinsics'], width, height,
                                                        config.min_depth, config.max_depth)
            valid = masks['valid']
            stats['behind_camera_count'] += int((masks['finite'] & ~masks['in_front']).sum())
            stats['out_of_range_count'] += int((masks['finite'] & masks['in_front'] & ~masks['in_range']).sum())
            stats['out_of_frame_count'] += int((masks['finite'] & masks['in_front'] & masks['in_range'] & ~masks['in_bounds']).sum())
            stats['valid_projections'] += int(valid.sum())
            frame_valid += int(valid.sum())
            buffers.append((pixels, depth, valid))
        depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
        for pixels, depth, valid in buffers:
            if valid.any():
                u = np.rint(pixels[valid, 0]).astype(np.int64)
                v = np.rint(pixels[valid, 1]).astype(np.int64)
                np.minimum.at(depth_buffer, (v, u), depth[valid].astype(np.float32))

        # Pass 2: occlusion test, bilinear sampling and observation accumulation.
        for start in range(0, len(selected), config.chunk_points):
            stop = min(start + config.chunk_points, len(selected))
            block = selected[start:stop]
            _, pixels, depth, masks = project_to_pixels(points[block], rotation, origin,
                                                        intr['intrinsics'], width, height,
                                                        config.min_depth, config.max_depth)
            keep = occlusion_keep(pixels, depth, masks['valid'], depth_buffer,
                                  config.occlusion_base_tolerance, config.occlusion_range_scale)
            stats['occlusion_rejected_count'] += int(masks['valid'].sum() - keep.sum())
            if keep.any():
                idx = np.flatnonzero(keep)
                rgb = bilinear_sample_rgb(image, pixels[idx, 0], pixels[idx, 1])
                obs_voxel.append(voxel_of_point[block[idx]])
                obs_rgb.append(rgb)
                stats['colored_observation_count'] += int(len(idx))
        if stats['frames_projected'] % 50 == 0:
            progress(f'Frame {stats["frames_projected"]:,}: {frame_valid:,} projected, '
                     f'{stats["colored_observation_count"]:,} colored observations')

    progress('Fusing multi-view RGB observations')
    if obs_voxel:
        all_voxel = np.concatenate(obs_voxel)
        all_rgb = np.concatenate(obs_rgb)
    else:
        all_voxel = np.empty(0, dtype=np.int64)
        all_rgb = np.empty((0, 3), dtype=np.float32)
    fused, color_count = fuse_grouped_median(all_voxel, all_rgb, n_voxels)

    # Reorder voxel-indexed fusion results into acquisition order for output.
    fused = fused[voxel_of_representative]
    color_count = color_count[voxel_of_representative]

    final_points = points[representative].astype(np.float32)
    final_intensity = intensity[representative]
    final_timestamps = timestamps[representative]
    final_origins = origins[representative].astype(np.float32)

    colored = color_count > 0
    confidence = np.zeros(n_voxels, dtype=np.float32)
    confidence[colored] = np.clip(color_count[colored] / 3.0, 0.0, 1.0)
    cd = np.where(colored[:, None], fused, np.nan).astype(np.float32)
    rgb_u8 = np.empty((n_voxels, 3), dtype=np.uint8)
    rgb_u8[colored] = np.clip(np.rint(cd[colored] * 255.0), 0, 255)
    rgb_u8[~colored] = np.asarray(config.fallback_color, dtype=np.uint8)

    output = Path(output_dir) if output_dir is not None else session/'colorization'/('run_' + uuid.uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=True)
    (output/'output').mkdir(parents=True, exist_ok=True)
    (output/'validation').mkdir(parents=True, exist_ok=True)

    progress('Writing colored point cloud')
    write_colored_ply(output/'output/colored_points.ply', final_points, rgb_u8, final_intensity)
    write_colored_npz(output/'output/colored_points.npz', final_points, cd, confidence, color_count,
                      intensity=final_intensity, timestamps=final_timestamps, origins=final_origins)

    colored_point_count = int(colored.sum())
    if colored_point_count:
        mean_obs = float(color_count[colored].mean())
        median_obs = float(np.median(color_count[colored]))
    else:
        mean_obs = median_obs = 0.0

    stats.update(final_point_count=int(n_voxels), colored_point_count=colored_point_count,
                 uncolored_point_count=int(n_voxels - colored_point_count),
                 percentage_colored=100.0 * colored_point_count / n_voxels if n_voxels else 0.0,
                 mean_observation_count=mean_obs, median_observation_count=median_obs)

    if config.validation_frames:
        progress(f'Generating {config.validation_frames} projection validation overlays')
        eligible_indices = [i for i, (stamp, _, _) in enumerate(frames())
                            if trajectory_t0 <= stamp + offset <= trajectory_t1]
        if eligible_indices:
            picks = np.linspace(0, len(eligible_indices) - 1,
                                min(config.validation_frames, len(eligible_indices)), dtype=int)
            selected_frames = {eligible_indices[int(i)] for i in picks}
        else:
            selected_frames = set()
        written = 0
        for index, (stamp, bag_ns, msg) in enumerate(frames()):
            if index not in selected_frames:
                continue
            if (msg.width, msg.height) != (camera['width'], camera['height']):
                raise ValueError('Recorded image dimensions differ from session intrinsics')
            rotation, origin, _ = camera_pose(trajectory, stamp, offset, extrinsic)
            overlay = render_projection_overlay(image_bgr(msg), points, rotation, origin,
                                                intr['intrinsics'], msg.width, msg.height, max_depth_range)
            filename = f'frame_{index:06d}_overlay.jpg'
            if not cv2.imwrite(str(output/'validation'/filename), overlay,
                               [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise ValueError(f'Could not write {filename}')
            written += 1
        stats['validation_frames_written'] = written

    metadata = dict(calibration, image_topic=image_topic, lidar_topic=topic,
                    trajectory_path=str(trajectory_path), raw_bag=str(bag),
                    configuration=config.as_dict(), statistics=stats,
                    timestamp_source='recorded Image.header.stamp',
                    temporal_association=f'|point_t - (image_t + offset)| <= max_time_delta',
                    time_offset_sec=offset,
                    fusion='per-channel median',
                    output=dict(colored_ply=str(output/'output/colored_points.ply'),
                                colored_npz=str(output/'output/colored_points.npz'),
                                stats_json=str(output/'output/colorization_stats.json')))
    atomic_json(output/'metadata.json', metadata)
    atomic_json(output/'output/colorization_stats.json', stats)
    atomic_yaml(output/'calibration_snapshot.yaml',
                dict(intrinsics=intr, extrinsic=calibration, time_offset_sec=offset))
    progress(f'Colorized {colored_point_count:,}/{n_voxels:,} voxels ({stats["percentage_colored"]:.1f}%)')
    return output
