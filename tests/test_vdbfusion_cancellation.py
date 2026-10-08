"""Native cancellation: a real worker, a real bag, a real SIGTERM during extraction.

This is native verification, not a mocked test: the worker under test is the isolated
VDBFusion interpreter running the real native library over a real ROS 2 bag, and the
signal is delivered by the operating system to the worker's process group.

What is asserted: the worker exits within a bounded timeout, its process tree is gone,
the run is reported as ``CANCELLED`` (never as an engine failure), and the previous
reconstruction output is untouched.

What is *measured* rather than claimed: when the signal lands relative to the native
``extract_triangle_mesh`` call. A Python signal handler only runs when the interpreter
regains control, and the native call does not release it, so this test never asserts
interruption inside the C++ call. It records the stage the worker was in, the delay
between the signal and the exit, and the stage the cancellation was actually reported
from, so the deferral is visible instead of assumed.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ui/backend'))

# A dense 4 m x 4 m plane at 4 mm voxels yields roughly 4.6 million triangles and about
# three seconds of native extraction, which is what makes it possible to deliver the
# signal while the worker is inside the native extractor.
PLANE_SIZE_M = 4.0
PLANE_SPACING_M = 0.004
PLANE_Z_M = 3.0
SIGNAL_POLL_SECONDS = 0.02
STAGE_WAIT_SECONDS = 180.0
EXIT_WAIT_SECONDS = 90.0


def build_dense_plane_bag(root):
    """Write a real ROS 2 bag whose single frame observes a dense planar surface."""
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import PointCloud2, PointField

    axis = np.arange(-PLANE_SIZE_M/2, PLANE_SIZE_M/2 + PLANE_SPACING_M/2, PLANE_SPACING_M)
    grid = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
    points = np.column_stack((grid[:, 0], grid[:, 1], np.full(len(grid), PLANE_Z_M)))

    bag = root/'raw_bag'
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    writer.create_topic(rosbag2_py.TopicMetadata(name='/livox/lidar', type='sensor_msgs/msg/PointCloud2',
                                                 serialization_format='cdr'))
    cloud = np.zeros(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                         ('intensity', '<f4'), ('timestamp', '<f8')])
    cloud['x'], cloud['y'], cloud['z'] = points[:, 0], points[:, 1], points[:, 2]
    cloud['intensity'] = 1.0
    for stamp in (0.0, 1.0):
        cloud['timestamp'] = stamp*1e9
        message = PointCloud2()
        message.header.frame_id = 'livox_frame'
        message.height, message.width = 1, len(cloud)
        message.point_step = cloud.dtype.itemsize
        message.row_step = message.point_step*len(cloud)
        message.fields = [PointField(name=name, offset=int(cloud.dtype.fields[name][1]),
                                     datatype=PointField.FLOAT32, count=1)
                          for name in ('x', 'y', 'z', 'intensity')]
        message.fields.append(PointField(name='timestamp', offset=int(cloud.dtype.fields['timestamp'][1]),
                                        datatype=PointField.FLOAT64, count=1))
        message.data = cloud.tobytes()
        writer.write('/livox/lidar', serialize_message(message), int(stamp*1e9))
    del writer
    trajectory = root/'traj_lidar.txt'
    np.savetxt(trajectory, np.array([[0.0, 0, 0, 0, 0, 0, 0, 1], [1.0, 0, 0, 0, 0, 0, 0, 1]]))
    return bag, trajectory


def worker_environment():
    env = os.environ.copy()
    env.pop('PYTHONHOME', None)
    env['OMP_NUM_THREADS'] = env.get('OMP_NUM_THREADS', '4')
    return env


def group_members(pgid):
    """Every live process still in the worker's process group."""
    import psutil
    alive = []
    for process in psutil.process_iter(['pid', 'status']):
        try:
            if process.info['status'] == psutil.STATUS_ZOMBIE:
                continue
            if os.getpgid(process.pid) == pgid:
                alive.append(process.pid)
        except (ProcessLookupError, PermissionError, psutil.Error):
            continue
    return alive


def read_progress(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def wait_for_stage(progress_path, stage, timeout=STAGE_WAIT_SECONDS):
    """Poll the worker's own progress file until it reports ``stage``."""
    deadline = time.monotonic() + timeout
    seen = None
    while time.monotonic() < deadline:
        seen = read_progress(progress_path).get('stage', seen)
        if seen == stage:
            return time.monotonic()
        time.sleep(SIGNAL_POLL_SECONDS)
    return None


@pytest.mark.slow
def test_sigterm_during_real_native_extraction_cancels_cleanly(tmp_path, vdbfusion_python):
    bag, trajectory = build_dense_plane_bag(tmp_path)
    run = tmp_path/'run'
    output = run/'output'
    # The jobs layer archives the previous attempt before a retry; simulate that archived
    # previous output and prove the cancellation never touches it.
    previous = run/'attempts'/'previous_deadbeef'/'output'
    previous.mkdir(parents=True)
    (previous/'mesh.ply').write_bytes(b'previous valid reconstruction output')
    progress = run/'vdbfusion_progress.json'
    stderr_path = run/'worker_stderr.log'
    run.mkdir(parents=True, exist_ok=True)
    log = stderr_path.open('wb')
    process = subprocess.Popen(
        [str(vdbfusion_python), str(ROOT/'tools/vdbfusion_worker.py'),
         '--bag', str(bag), '--trajectory', str(trajectory), '--topic', '/livox/lidar',
         '--voxel-size', '0.004', '--output', str(output/'mesh.ply'),
         '--metadata', str(output/'vdbfusion_metadata.json'), '--progress', str(progress),
         '--memory-budget-gib', '20'],
        stdout=log, stderr=subprocess.STDOUT, env=worker_environment(), start_new_session=True)
    log.close()
    pgid = os.getpgid(process.pid)
    try:
        extracted_at = wait_for_stage(progress, 'EXTRACTING_MESH')
        stage_at_signal = read_progress(progress).get('stage')
        assert extracted_at is not None, (
            'the worker never reached EXTRACTING_MESH; evidence: ' + str(read_progress(progress)))
        signalled_at = time.monotonic()
        os.killpg(pgid, signal.SIGTERM)
        try:
            returncode = process.wait(timeout=EXIT_WAIT_SECONDS)
        except subprocess.TimeoutExpired:  # pragma: no cover - only on a real hang
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=30)
            pytest.fail('the worker did not exit within the bounded timeout after SIGTERM')
        exit_seconds = time.monotonic() - signalled_at
        failure = read_progress(progress)
        evidence = dict(stage_at_signal=stage_at_signal, exit_code=returncode,
                        seconds_from_signal_to_exit=round(exit_seconds, 3),
                        last_stage_reported=failure.get('stage'),
                        error_type=failure.get('error_type'),
                        worker_exited=process.poll() is not None)
        log_text = stderr_path.read_text(errors='replace')
        stages = [json.loads(line)['stage'] for line in log_text.splitlines()
                  if line.startswith('{') and '"stage"' in line]
        evidence['stages'] = stages[-6:]

        # 1. Clean, reported cancellation - a SIGTERM is a stop, never an engine failure.
        assert returncode == 2, evidence
        assert failure.get('error_type') == 'CANCELLED', evidence
        assert failure.get('stage') == 'EXTRACTING_MESH', evidence
        # 2. Bounded exit: the cooperative flag is honoured at the next checkpoint.
        assert exit_seconds < EXIT_WAIT_SECONDS, evidence
        # 3. Process-tree cleanup: nothing of the worker's group survives.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and group_members(pgid):
            time.sleep(0.1)
        assert group_members(pgid) == [], f'the process group leaked {group_members(pgid)}'
        # 4. Nothing was published, and the previous output is byte-identical.
        assert not (output/'mesh.ply').exists(), 'a cancelled run must not publish a mesh'
        assert not (output/'publish_manifest.json').exists()
        assert (previous/'mesh.ply').read_bytes() == b'previous valid reconstruction output'
        # 5. No partial staged mesh is left behind either.
        assert not (output/'staging').exists(), 'staging must be cleaned after a cancellation'
        print(f'native cancellation evidence: {json.dumps(evidence, sort_keys=True)}')
    finally:
        if process.poll() is None:  # pragma: no cover - defensive
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=30)


@pytest.mark.slow
def test_sigterm_after_extraction_completes_still_never_publishes(tmp_path, vdbfusion_python):
    """The same signal delivered later is still safe: the run stops before publishing.

    This is the control case for the deferral measurement above: if the native extraction
    finishes first, the cancellation is honoured at the pre-publish checkpoint instead, and
    the previous output is still untouched.
    """
    bag, trajectory = build_dense_plane_bag(tmp_path)
    run = tmp_path/'run'
    run.mkdir(parents=True)
    progress = run/'vdbfusion_progress.json'
    stderr_path = run/'worker_stderr.log'
    log = stderr_path.open('wb')
    process = subprocess.Popen(
        [str(vdbfusion_python), str(ROOT/'tools/vdbfusion_worker.py'),
         '--bag', str(bag), '--trajectory', str(trajectory), '--topic', '/livox/lidar',
         '--voxel-size', '0.004', '--output', str(run/'output'/'mesh.ply'),
         '--metadata', str(run/'output'/'vdbfusion_metadata.json'), '--progress', str(progress),
         '--memory-budget-gib', '20'],
        stdout=log, stderr=subprocess.STDOUT, env=worker_environment(), start_new_session=True)
    log.close()
    pgid = os.getpgid(process.pid)
    try:
        assert wait_for_stage(progress, 'VALIDATING_MESH') is not None, 'the worker never validated a mesh'
        os.killpg(pgid, signal.SIGTERM)
        returncode = process.wait(timeout=EXIT_WAIT_SECONDS)
        failure = read_progress(progress)
        assert returncode == 2 and failure.get('error_type') == 'CANCELLED', failure
        assert not (run/'output'/'mesh.ply').exists()
        assert not (run/'output'/'publish_manifest.json').exists()
    finally:
        if process.poll() is None:  # pragma: no cover - defensive
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=30)
