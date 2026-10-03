"""Raw bag → world-space paired observations → optional voxel selection.

No dependency on NKSR or Open3D; GLIM exports are never modified.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .storage import atomic_json

DEFAULT_VOXEL_SIZE_M = 0.01


def validate_voxel_size(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Voxel size must be finite and nonnegative, in meters (0 disables sampling)')
    return value


def voxel_indices(points, voxel_size_m=DEFAULT_VOXEL_SIZE_M):
    """First input measurement per world voxel, in original acquisition order.

    int64 coordinates avoid packed-key collisions. Chunked quantization bounds
    float temporaries; NumPy unique performs the global grouping across frames.
    """
    size = validate_voxel_size(voxel_size_m)
    if size == 0:
        return np.arange(len(points))
    keys = np.empty((len(points), 3), dtype=np.int64)
    for start in range(0, len(points), 250_000):
        block = np.floor(points[start:start+250_000].astype(np.float64) / size)
        if not np.isfinite(block).all() or np.any(block < -(2.**63)) or np.any(block >= 2.**63):
            raise ValueError('World coordinates exceed the supported voxel index range')
        keys[start:start+len(block)] = block
    _, indices = np.unique(keys, axis=0, return_index=True)
    indices.sort()
    return indices


def sample_records(records, voxel_size_m=DEFAULT_VOXEL_SIZE_M):
    count = len(records['points'])
    if any(len(values) != count for values in records.values()):
        raise ValueError('Measurement attribute arrays must have identical lengths')
    indices = voxel_indices(records['points'], voxel_size_m)
    return {name: values[indices] for name, values in records.items()}


def normalize_quaternion(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def slerp_batch(q0, q1, alpha):
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)

    dot = np.sum(q0 * q1, axis=1)

    # Take shortest quaternion path
    flip = dot < 0.0
    q1 = q1.copy()
    q1[flip] *= -1.0
    dot[flip] *= -1.0

    dot = np.clip(dot, -1.0, 1.0)

    result = np.empty_like(q0)

    # Nearly identical rotations -> linear interpolation
    linear = dot > 0.9995

    if np.any(linear):
        a = alpha[linear, None]

        result[linear] = (
            q0[linear] +
            a * (q1[linear] - q0[linear])
        )

        result[linear] = normalize_quaternion(result[linear])

    nonlinear = ~linear

    if np.any(nonlinear):
        theta0 = np.arccos(dot[nonlinear])
        sin_theta0 = np.sin(theta0)

        a = alpha[nonlinear]

        s0 = np.sin((1.0 - a) * theta0) / sin_theta0
        s1 = np.sin(a * theta0) / sin_theta0

        result[nonlinear] = (
            s0[:, None] * q0[nonlinear] +
            s1[:, None] * q1[nonlinear]
        )

    return normalize_quaternion(result)


def rotate_vectors(q, v):
    """
    Rotate Nx3 vectors with Nx4 quaternions.
    quaternion order = x,y,z,w
    """

    qv = q[:, :3]
    qw = q[:, 3:4]

    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)

    return v + 2.0 * (qw * uv + uuv)


def get_cloud_arrays(msg):

    fields = {f.name: f for f in msg.fields}

    required = [
        "x",
        "y",
        "z",
        "intensity",
        "timestamp",
    ]

    for name in required:
        if name not in fields:
            raise RuntimeError(f"Missing PointCloud2 field: {name}")

    endian = ">" if msg.is_bigendian else "<"

    dtype = np.dtype({
        "names": [
            "x",
            "y",
            "z",
            "intensity",
            "timestamp",
        ],
        "formats": [
            endian + "f4",
            endian + "f4",
            endian + "f4",
            endian + "f4",
            endian + "f8",
        ],
        "offsets": [
            fields["x"].offset,
            fields["y"].offset,
            fields["z"].offset,
            fields["intensity"].offset,
            fields["timestamp"].offset,
        ],
        "itemsize": msg.point_step,
    })

    arr = np.ndarray((msg.height, msg.width), buffer=msg.data, dtype=dtype,
                     strides=(msg.row_step, msg.point_step)).reshape(-1)

    xyz = np.column_stack((
        arr["x"],
        arr["y"],
        arr["z"],
    )).astype(np.float64)

    intensity = np.asarray(
        arr["intensity"],
        dtype=np.float32
    )

    timestamp_ns = np.asarray(
        arr["timestamp"],
        dtype=np.float64
    )

    # Livox timestamp field is absolute nanoseconds
    timestamp_s = timestamp_ns * 1e-9

    valid = (
        np.isfinite(xyz).all(axis=1) &
        np.isfinite(timestamp_s)
    )

    return (
        xyz[valid],
        intensity[valid],
        timestamp_s[valid]
    )


def transform_points(xyz_lidar, point_t, trajectory):
    traj_t = trajectory[:, 0]
    traj_xyz = trajectory[:, 1:4]
    traj_q = trajectory[:, 4:8]

    # Find trajectory pose immediately after each point
    upper = np.clip(np.searchsorted(traj_t, point_t, side="right"), 1, len(traj_t)-1)

    lower = upper - 1

    # Need two valid poses for interpolation
    valid = (
        (point_t >= traj_t[0]) &
        (point_t <= traj_t[-1])
    )

    xyz_lidar = xyz_lidar[valid]
    point_t = point_t[valid]
    lower = lower[valid]
    upper = upper[valid]

    if len(point_t) == 0:
        return (
            np.empty((0, 3)),
            np.empty((0, 3)),
            valid
        )

    t0 = traj_t[lower]
    t1 = traj_t[upper]

    alpha = (
        (point_t - t0) /
        (t1 - t0)
    )

    alpha = np.clip(alpha, 0.0, 1.0)

    # Interpolate translation
    p0 = traj_xyz[lower]
    p1 = traj_xyz[upper]

    sensor_xyz = (
        p0 +
        alpha[:, None] * (p1 - p0)
    )

    # Interpolate rotation
    q0 = traj_q[lower]
    q1 = traj_q[upper]

    q = slerp_batch(q0, q1, alpha)

    # Transform LiDAR-local points into world
    rotated = rotate_vectors(q, xyz_lidar)

    xyz_world = rotated + sensor_xyz

    return xyz_world, sensor_xyz, valid


def read_measurements(bag, trajectory, topic='/livox/lidar', progress=print):
    # Import ROS only when reading a bag; preparation never imports NKSR.
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    chunks = {k: [] for k in ('points', 'sensor_origins', 'intensity', 'timestamps')}
    frames = 0
    while reader.has_next():
        name, data, _ = reader.read_next()
        if name != topic:
            continue
        xyz, intensity, timestamps = get_cloud_arrays(deserialize_message(data, PointCloud2))
        points, origins, valid = transform_points(xyz, timestamps, trajectory)
        if len(points):
            for key, values in zip(chunks, (points, origins, intensity[valid], timestamps[valid])):
                chunks[key].append(values)
        frames += 1
        if frames % 100 == 0:
            progress(f'Transforming raw observations: {frames:,} LiDAR frames')
    if not chunks['points']:
        raise ValueError('No finite points within the valid trajectory range')
    return {key: np.concatenate(values) for key, values in chunks.items()}


def write_ply(path, records):
    # Binary blocks keep export overhead bounded for full-density debug exports.
    points, intensity = records['points'], records['intensity']
    with Path(path).open('wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\n'
                 f'element vertex {len(points)}\nproperty float x\nproperty float y\n'
                 'property float z\nproperty float intensity\nend_header\n').encode('ascii'))
        for start in range(0, len(points), 250_000):
            np.column_stack((points[start:start+250_000], intensity[start:start+250_000])).astype('<f4').tofile(f)


def save_inputs(records, output, voxel_size_m=DEFAULT_VOXEL_SIZE_M,
                save_full_density=False, progress=print):
    voxel_size_m = validate_voxel_size(voxel_size_m)
    output = Path(output)
    (output/'input').mkdir(parents=True, exist_ok=False)
    (output/'validation').mkdir(parents=True, exist_ok=False)
    before = len(records['points'])
    progress(f'Voxel sampling {before:,} world-space measurements at {voxel_size_m:g} m')
    selected = sample_records(records, voxel_size_m)
    # Select at float64 world precision before converting the NKSR geometry.
    for name in ('points', 'sensor_origins', 'intensity'):
        selected[name] = selected[name].astype(np.float32)
    after = len(selected['points'])
    metadata = dict(voxel_size_m=voxel_size_m, points_before_voxel=before,
                    points_after_voxel=after, voxel_reduction_ratio=after/before if before else 0.0,
                    representative_selection='first_measurement_in_acquisition_order',
                    validation_note='Compare spatial agreement with GLIM; different point counts are expected.')
    progress(f'Saving NKSR input: {before:,} → {after:,} measurements')
    np.savez_compressed(output/'input/nksr_input.npz', **selected)
    write_ply(output/'validation/reconstructed_from_bag.ply', selected)
    if save_full_density:
        progress('Saving optional full-density validation PLY')
        write_ply(output/'validation/reconstructed_from_bag_full.ply', records)
    atomic_json(output/'validation/comparison.json', metadata)
    return metadata


def prepare(bag, trajectory_path, output, voxel_size_m=DEFAULT_VOXEL_SIZE_M,
            save_full_density=False, topic='/livox/lidar', progress=print):
    validate_voxel_size(voxel_size_m)
    trajectory = np.loadtxt(trajectory_path, ndmin=2)
    if (trajectory.shape[1] != 8 or len(trajectory) < 2 or
            not np.isfinite(trajectory).all() or np.any(np.diff(trajectory[:, 0]) <= 0) or
            np.any(np.linalg.norm(trajectory[:, 4:8], axis=1) == 0)):
        raise ValueError('Trajectory requires increasing timestamps and finite XYZ / XYZW poses')
    progress('Interpolating trajectory and transforming raw measurements into world space')
    records = read_measurements(bag, trajectory, topic, progress)
    return save_inputs(records, output, voxel_size_m, save_full_density, progress)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bag', required=True, type=Path, help='Raw ROS 2 sqlite3 bag directory')
    p.add_argument('--trajectory', required=True, type=Path, help='GLIM traj_lidar.txt')
    p.add_argument('--output-dir', required=True, type=Path, help='New reconstruction/run_xxx directory')
    p.add_argument('--voxel-size', type=validate_voxel_size, default=DEFAULT_VOXEL_SIZE_M,
                   help='World-space voxel size in METERS (default: %(default)s = 1 cm; 0 disables sampling)')
    p.add_argument('--save-full-density', action='store_true', help='Also save full-density debug PLY')
    p.add_argument('--topic', default='/livox/lidar')
    return p


def main():
    args = parser().parse_args()
    # A fresh run prevents overwriting an earlier result or any GLIM export.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir/name).exists() for name in ('input', 'validation')):
        raise FileExistsError('Choose a new reconstruction output directory')
    def progress(message):
        print(message, flush=True)
        atomic_json(args.output_dir/'progress.json', {'message': message})
    result = prepare(args.bag, args.trajectory, args.output_dir, args.voxel_size,
                     args.save_full_density, args.topic, progress)
    progress('Preparation complete')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
