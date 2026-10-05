"""D405 RGB-only acquisition, rectified using the device's factory calibration."""
import argparse
from array import array
import json
from pathlib import Path

from .calibration_data import replace_intrinsics, parse_intrinsics
from .camera_config import config_path


def resolve(camera):
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    config.disable_all_streams()
    config.enable_device(camera['serial_number'])
    config.enable_stream(rs.stream.color, camera['width'], camera['height'], rs.format.rgb8, int(camera['fps']))
    profile = config.resolve(rs.pipeline_wrapper(pipeline))
    if 'D405' not in profile.get_device().get_info(rs.camera_info.name):
        raise ValueError('Configured RealSense device must be a D405')
    return pipeline, config, profile


def calibration(camera, profile):
    import pyrealsense2 as rs
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    if intr.model not in (rs.distortion.none, rs.distortion.brown_conrady, rs.distortion.inverse_brown_conrady):
        raise ValueError('Unsupported RealSense color distortion model: '+str(intr.model))
    def matrix(rows, cols, values):
        return dict(rows=rows, cols=cols, data=values)
    # Images are rectified before publication, so downstream uses zero distortion.
    obj = dict(image_width=intr.width, image_height=intr.height, camera_name=camera['camera_name'],
               distortion_model='plumb_bob',
               camera_matrix=matrix(3, 3, [intr.fx, 0., intr.ppx, 0., intr.fy, intr.ppy, 0., 0., 1.]),
               distortion_coefficients=matrix(1, 5, [0.]*5),
               rectification_matrix=matrix(3, 3, [1., 0., 0., 0., 1., 0., 0., 0., 1.]),
               projection_matrix=matrix(3, 4, [intr.fx, 0., intr.ppx, 0., 0., intr.fy, intr.ppy, 0., 0., 0., 1., 0.]),
               factory_calibration=dict(serial_number=profile.get_device().get_info(rs.camera_info.serial_number),
                                        distortion_model=str(intr.model), coefficients=list(intr.coeffs),
                                        rectified=True, stream='color', format='rgb8'))
    parse_intrinsics(obj, camera)
    return intr, obj


def save_calibration(root, camera, obj):
    import yaml
    path = config_path(root, camera['intrinsics_file'])
    previous = yaml.safe_load(path.read_text()) if path.exists() else None
    if previous != obj:
        replace_intrinsics(root, camera, obj)


def rectification_maps(intr):
    """Use SDK projection for the exact device model (including inverse Brown)."""
    import numpy as np
    import pyrealsense2 as rs
    pixels = [rs.rs2_project_point_to_pixel(intr, [(x-intr.ppx)/intr.fx, (y-intr.ppy)/intr.fy, 1.])
              for y in range(intr.height) for x in range(intr.width)]
    maps = np.asarray(pixels, dtype=np.float32).reshape(intr.height, intr.width, 2)
    return maps[:, :, 0].copy(), maps[:, :, 1].copy()


def publish(camera, pipeline, config):
    import cv2
    import numpy as np
    import pyrealsense2 as rs
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    rclpy.init()
    node = Node('factory_mapping_d405')
    images = node.create_publisher(Image, camera['image_topic'], 10)
    infos = node.create_publisher(CameraInfo, camera['camera_info_topic'], 10)
    profile = pipeline.start(config)
    try:
        intr, obj = calibration(camera, profile)
        map_x, map_y = rectification_maps(intr)
        info = CameraInfo(width=intr.width, height=intr.height, distortion_model='plumb_bob')
        info.k = obj['camera_matrix']['data']; info.d = obj['distortion_coefficients']['data']
        info.r = obj['rectification_matrix']['data']; info.p = obj['projection_matrix']['data']
        while rclpy.ok():
            frame = pipeline.wait_for_frames(5000).get_color_frame()
            if not frame: continue
            # Host receipt time shares the ROS/LiDAR clock; never publish device uptime as epoch time.
            stamp = node.get_clock().now().to_msg()
            rgb = cv2.remap(np.asanyarray(frame.get_data()), map_x, map_y, cv2.INTER_LINEAR)
            msg = Image(width=intr.width, height=intr.height, encoding='rgb8', step=intr.width*3)
            msg.header.stamp = stamp; msg.header.frame_id = camera['frame_id']
            msg.data = array('B', rgb.tobytes()); info.header = msg.header
            images.publish(msg); infos.publish(info)
            rclpy.spin_once(node, timeout_sec=0)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop(); node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Trusted camera configuration JSON')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    camera = json.loads(args.config)
    pipeline, config, profile = resolve(camera)
    _, obj = calibration(camera, profile)
    if args.extract:
        save_calibration(args.root, camera, obj)
        print(json.dumps(obj))
    else:
        # Extraction is done before acquisition so session snapshots match the images.
        import yaml
        saved = yaml.safe_load(config_path(args.root, camera['intrinsics_file']).read_text())
        if saved != obj: raise ValueError('D405 factory intrinsics changed; restart camera to extract them before recording')
        publish(camera, pipeline, config)


if __name__ == '__main__':
    main()
