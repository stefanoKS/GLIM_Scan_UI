"""D405 RGB-only acquisition, rectified using the device's factory calibration."""
import argparse
from array import array
import json
import time
from importlib.metadata import version
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
    """Rectified destination pixel -> source pixel, matching librealsense 2.58.2.

    Inverse Brown uses the SDK's projection polynomial: tangential terms use
    radially scaled x/y but the original r². It is NOT OpenCV ordinary Brown.
    See src/rs.cpp rs2_project_point_to_pixel in the pinned SDK.
    """
    import numpy as np
    import pyrealsense2 as rs
    if intr.model == rs.distortion.none: return None
    if intr.model not in (rs.distortion.brown_conrady, rs.distortion.inverse_brown_conrady):
        raise ValueError('Unsupported rectification model: '+str(intr.model))
    y,x=np.indices((intr.height,intr.width),dtype=np.float32)
    x=(x-intr.ppx)/intr.fx; y=(y-intr.ppy)/intr.fy
    r2=x*x+y*y
    k1,k2,p1,p2,k3=intr.coeffs
    radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
    xf=x*radial; yf=y*radial
    if intr.model == rs.distortion.inverse_brown_conrady: x,y=xf,yf
    mx=(xf+2*p1*x*y+p2*(r2+2*x*x))*intr.fx+intr.ppx
    my=(yf+2*p2*x*y+p1*(r2+2*y*y))*intr.fy+intr.ppy
    return mx,my


def diagnostic(stage, started, **fields):
    print(json.dumps(dict(stage=stage,seconds=time.monotonic()-started,**fields)),flush=True)


def publish(camera, pipeline, config, root):
    import cv2
    import numpy as np
    import pyrealsense2 as rs
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    cv2.setNumThreads(2)
    rclpy.init()
    node = Node('factory_mapping_d405')
    images = node.create_publisher(Image, camera['image_topic'], 10)
    infos = node.create_publisher(CameraInfo, camera['camera_info_topic'], 10)
    pipeline_started=False
    try:
        started=time.monotonic()
        profile = pipeline.start(config)
        pipeline_started=True
        diagnostic("pipeline.start",started)
        started=time.monotonic()
        intr, obj = calibration(camera, profile)
        save_calibration(root,camera,obj)
        diagnostic('calibration',started)
        started=time.monotonic()
        maps = rectification_maps(intr)
        # Native fixed-point maps reduce per-frame remap cost and memory bandwidth.
        maps = cv2.convertMaps(*maps,cv2.CV_16SC2) if maps is not None else None
        diagnostic('rectification_maps',started,remap=maps is not None)
        first=True; waiting=time.monotonic()
        info = CameraInfo(width=intr.width, height=intr.height, distortion_model='plumb_bob')
        info.k = obj['camera_matrix']['data']; info.d = obj['distortion_coefficients']['data']
        info.r = obj['rectification_matrix']['data']; info.p = obj['projection_matrix']['data']
        while rclpy.ok():
            frame = pipeline.wait_for_frames(5000).get_color_frame()
            if not frame: continue
            # Host receipt time shares the ROS/LiDAR clock; never publish device uptime as epoch time.
            stamp = node.get_clock().now().to_msg()
            rgb = np.asanyarray(frame.get_data())
            if maps is not None: rgb = cv2.remap(rgb,*maps,cv2.INTER_LINEAR)
            msg = Image(width=intr.width, height=intr.height, encoding='rgb8', step=intr.width*3)
            msg.header.stamp = stamp; msg.header.frame_id = camera['frame_id']
            msg.data = array('B', rgb.tobytes()); info.header = msg.header
            images.publish(msg); infos.publish(info)
            if first:
                diagnostic('first_frame',waiting,width=intr.width,height=intr.height,total_startup_seconds=time.monotonic()-STARTED)
                first=False
            rclpy.spin_once(node, timeout_sec=0)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if pipeline_started: pipeline.stop()
        finally:
            node.destroy_node()
            if rclpy.ok(): rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Trusted camera configuration JSON')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    camera = json.loads(args.config)
    started=time.monotonic()
    pipeline, config, profile = resolve(camera)
    diagnostic("resolve",started,pyrealsense2=version("pyrealsense2"))
    if args.extract:
        _, obj = calibration(camera, profile)
        save_calibration(args.root, camera, obj)
        print(json.dumps(obj))
    else:
        publish(camera, pipeline, config, args.root)


STARTED=time.monotonic()

if __name__ == '__main__':
    main()
