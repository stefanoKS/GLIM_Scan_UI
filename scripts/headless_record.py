"""Small process supervisor; no image subscription or image processing."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from datetime import datetime
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--countdown', type=int, default=10)
    parser.add_argument('--camera-serial')
    args = parser.parse_args()
    if args.countdown < 0:
        parser.error('--countdown must be nonnegative')
    import yaml
    import psutil
    from factory_mapping.commands import driver
    from factory_mapping.config import wired_connection

    root = Path(os.environ['FACTORY_MAPPING_ROOT'])
    state = root / '.state'
    state.mkdir(exist_ok=True)
    lock = (state / 'headless_record.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another headless recorder is running.')
    for process in psutil.process_iter(['pid', 'cmdline']):
        cmd = process.info['cmdline'] or []
        if process.pid == os.getpid():
            continue
        executables = {Path(token).name for token in cmd[:2]}
        if (('bag' in cmd and 'record' in cmd) or
                executables.intersection({'livox_ros_driver2_node', 'realsense2_camera_node'}) or
                'factory_mapping.realsense_camera' in cmd):
            raise RuntimeError(f'Conflicting recorder/sensor process: PID {process.pid}')
    for package in ('realsense2_camera', 'livox_ros_driver2', 'rosbag2_storage_default_plugins'):
        result = subprocess.run(['ros2', 'pkg', 'prefix', package], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f'Missing ROS package/workspace: {package}. '
                               'Install/build it for ROS2 Humble and source its workspace. '
                               'No Python camera fallback is used.')
    subprocess.run(['ros2', 'bag', 'record', '--help'], check=True, stdout=subprocess.DEVNULL)
    sensor = yaml.safe_load((root / 'config/livox/mid360.yaml').read_text())
    sensor['interface'], sensor['host_ip'] = wired_connection(sensor)
    if not sensor['host_ip']:
        raise RuntimeError('No unambiguous wired host address on the MID-360 subnet.')
    if not (root / 'config/livox/upstream_mid360.json').is_file():
        raise RuntimeError('Missing Livox upstream_mid360.json config.')
    os.environ['ROS_DOMAIN_ID'] = str(sensor['ros_domain_id'])
    base = root / 'data/sessions'
    base.mkdir(parents=True, exist_ok=True)
    system = yaml.safe_load((root / 'config/system.yaml').read_text())
    if shutil.disk_usage(base).free < max(5, system['storage']['minimum_free_gb']) * 1024**3:
        raise RuntimeError('Insufficient disk space: at least 5 GiB/configured minimum required.')

    # SDK enumeration only: no pipeline, stream, OpenCV, or frame processing.
    import pyrealsense2 as rs
    devices = [d for d in rs.context().query_devices()
               if 'D405' in d.get_info(rs.camera_info.name).split()]
    serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
    if args.camera_serial:
        devices = [d for d in devices if d.get_info(rs.camera_info.serial_number) == args.camera_serial]
    if len(devices) != 1:
        raise RuntimeError('Select exactly one D405 with --camera-serial SDK_SERIAL. '
                           f'Detected D405 SDK serials: {serials or "none"}')
    device = devices[0]
    camera = dict(model=device.get_info(rs.camera_info.name),
                  sdk_serial=device.get_info(rs.camera_info.serial_number),
                  usb_serial=None, usb=None, width=1280, height=720, fps=5)
    if device.supports(rs.camera_info.usb_type_descriptor):
        camera['usb'] = device.get_info(rs.camera_info.usb_type_descriptor)
    supported = set()
    for sdk_sensor in device.query_sensors():
        for profile in sdk_sensor.get_stream_profiles():
            if profile.stream_type() == rs.stream.color and profile.format() == rs.format.rgb8:
                video = profile.as_video_stream_profile()
                supported.add((video.width(), video.height(), profile.fps()))
    if (1280, 720, 5) not in supported:
        rates = sorted(fps for width, height, fps in supported if (width, height) == (1280, 720))
        raise RuntimeError(
            f'D405 SDK serial {camera["sdk_serial"]}, USB {camera["usb"] or "unknown"}: '
            f'1280x720@5 RGB8 is unavailable. Advertised 1280x720 RGB8 FPS: {rates}. '
            'Reconnect using a USB 3.x data cable and USB 3.x port; check any hub/adapter. '
            'Resolution and FPS will not be reduced.')
    # physical_port identifies a USB sysfs ancestor; never use video nodes for identity.
    if device.supports(rs.camera_info.physical_port):
        port = Path(device.get_info(rs.camera_info.physical_port))
        if port.is_absolute() and str(port).startswith('/sys/'):
            for parent in (port, *port.parents):
                try:
                    if (parent / 'idVendor').read_text().strip() == '8086':
                        camera['usb_serial'] = (parent / 'serial').read_text().strip()
                        break
                except OSError:
                    pass
    del device, devices
    calibration = {'status': 'UNKNOWN', 'sdk_serial': None}
    intrinsics = root / 'config/calibration/d405_intrinsics.yaml'
    if intrinsics.exists():
        try:
            metadata = yaml.safe_load(intrinsics.read_text()) or {}
            serial = metadata.get('factory_calibration', {}).get('serial_number')
            if serial:
                calibration = dict(sdk_serial=str(serial),
                                   status='MATCH' if str(serial) == camera['sdk_serial'] else 'MISMATCH')
        except (OSError, ValueError, yaml.YAMLError) as error:
            print(f'WARNING: calibration identity unavailable: {error}', flush=True)

    session = base / datetime.now().strftime('%Y%m%d_%H%M%S_headless')
    session.mkdir()
    (session / 'logs').mkdir()
    snapshot = session / 'config_snapshot'
    shutil.copytree(root / 'config', snapshot)
    camera_config = yaml.safe_load((root / 'config/camera/d405.yaml').read_text())
    camera_config.update(serial_number=camera['sdk_serial'], usb_serial_number=camera['usb_serial'])
    system['camera'].update(enabled=True, profile='d405')
    config = dict(sensor=sensor, camera=camera_config, system=system)
    (session / 'active_config.json').write_text(json.dumps(config, indent=2))
    (session / 'sensor_info.json').write_text(json.dumps(dict(
        camera=camera, lidar=dict(model='Livox MID-360', ip=sensor['lidar_ip']),
        calibration_identity=calibration,
        timestamps='Native realsense2_camera and Livox driver stamps; no restamping.',
        image_geometry='Raw distorted RGB; use recorded CameraInfo offline.'), indent=2))
    topics = [sensor['points_topic'], sensor['imu_topic'],
              camera_config['image_topic'], camera_config['camera_info_topic']]
    qos = {t: dict(reliability='best_effort', durability='volatile',
                   history='keep_last', depth=5) for t in topics}
    qos_path = snapshot / 'headless_qos.yaml'
    qos_path.write_text(yaml.safe_dump(qos))
    # D405 color belongs to the depth module; set both profile names for other wrapper versions.
    params = dict(serial_no='_' + camera['sdk_serial'], enable_color=True,
                  enable_depth=False, enable_infra=False, enable_infra1=False, enable_infra2=False,
                  enable_gyro=False, enable_accel=False, enable_sync=False, enable_rgbd=False,
                  publish_tf=False, color_qos='SENSOR_DATA', color_info_qos='SENSOR_DATA',
                  **{'rgb_camera.color_profile': '1280x720x5',
                     'depth_module.color_profile': '1280x720x5',
                     'rgb_camera.color_format': 'RGB8', 'depth_module.color_format': 'RGB8'})
    for name in ('pointcloud', 'align_depth', 'colorizer', 'decimation_filter',
                 'spatial_filter', 'temporal_filter', 'disparity_filter',
                 'hole_filling_filter', 'hdr_merge'):
        params[name + '.enable'] = False
    params_path = snapshot / 'headless_camera.yaml'
    params_path.write_text(yaml.safe_dump({'/**': {'ros__parameters': params}}))
    camera_cmd = ['ros2', 'run', 'realsense2_camera', 'realsense2_camera_node',
                  '--ros-args', '--params-file', str(params_path),
                  '-r', '__ns:=/camera', '-r', '__node:=camera',
                  '-r', '~/color/image_raw:=' + topics[2],
                  '-r', '~/color/camera_info:=' + topics[3]]
    children = []
    bag = None
    started = None
    stopping = False

    def stop_requested(*_):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop_requested)

    def launch(cmd, name):
        with (session / 'logs' / (name + '.log')).open('w') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        children.append(process)
        return process

    def check():
        for process in children:
            if process.poll() is not None:
                raise RuntimeError(f'Child process exited ({process.returncode}); see {session / "logs"}')
        if stopping:
            raise InterruptedError()

    def stop(process):
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            # Intentionally no timeout/kill: rosbag must finish before sensors stop.
            process.wait()

    try:
        launch(driver(root, config), 'livox')
        launch(camera_cmd, 'camera')
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node('headless_preflight')
        received = []
        node.create_subscription(CameraInfo, topics[3], received.append, qos_profile_sensor_data)
        try:
            deadline = time.monotonic() + 45
            while True:
                check()
                rclpy.spin_once(node, timeout_sec=0.2)
                if received and all(node.count_publishers(t) == 1 for t in topics):
                    if (received[-1].width, received[-1].height) != (1280, 720):
                        raise RuntimeError(f'D405 reported {received[-1].width}x{received[-1].height}, '
                                           f'expected 1280x720. See {session / "logs/camera.log"}')
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'Timed out waiting for unique publishers/CameraInfo: {topics}')
        finally:
            node.destroy_node()
            rclpy.shutdown()
        print(f'D405 detected:\n  Model: {camera["model"]}\n  SDK serial: {camera["sdk_serial"]}'
              f'\n  USB: {camera["usb"] or "unknown"}\nMID-360 ready.', flush=True)
        if calibration['status'] == 'MISMATCH':
            print('WARNING: connected D405 differs from calibrated camera.\nRaw recording will continue.')
        else:
            print('Calibration identity: ' + calibration['status'])
        print('Recording starts in:', flush=True)
        for remaining in range(args.countdown, 0, -1):
            check()
            print(remaining, flush=True)
            time.sleep(1)
        check()
        bag = launch(['ros2', 'bag', 'record', '--storage', 'sqlite3', '--output',
                      str(session / 'raw_bag'), '--qos-profile-overrides-path', str(qos_path),
                      *topics], 'rosbag')
        started = time.monotonic()
        deadline = started + 15
        while not list((session / 'raw_bag').glob('*.db3')):
            check()
            if time.monotonic() >= deadline:
                raise RuntimeError('Timed out opening rosbag storage; see logs/rosbag.log')
            time.sleep(0.1)
        check()
        print(f'RECORDING STARTED\nBag: {session / "raw_bag"}\nPress ENTER or Ctrl+C to stop.', flush=True)
        stdin_open = True
        while not stopping:
            check()
            if stdin_open and select.select([sys.stdin], [], [], 0.5)[0]:
                line = sys.stdin.readline()
                if line:
                    break
                stdin_open = False
            elif not stdin_open:
                time.sleep(0.5)
    except InterruptedError:
        pass
    finally:
        stop(bag)
        for process in reversed(children):
            if process is not bag:
                stop(process)
        if started is not None:
            bag_path = session / 'raw_bag'
            size = sum(p.stat().st_size for p in bag_path.rglob('*') if p.is_file())
            if bag.returncode != 0 or not (bag_path / 'metadata.yaml').is_file():
                raise RuntimeError(f'Bag did not finalize successfully; inspect {session / "logs/rosbag.log"}')
            print(f'Recording complete\nBag: {bag_path}\nDuration: {time.monotonic()-started:.1f} s'
                  f'\nSize: {size / 1024**3:.3f} GiB', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ImportError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
