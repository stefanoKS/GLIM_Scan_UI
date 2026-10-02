"""ROS is imported only in sensor subprocesses, never by the HTTP server."""
import os, time
import numpy as np
from .config import ROOT, load
from .health import Rates
from .storage import atomic_json
from .preview import encode

def run(kind):
    config=load(); sensor=config['sensor']; os.environ['ROS_DOMAIN_ID']=str(sensor['ros_domain_id'])
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2, Imu
    from sensor_msgs_py.point_cloud2 import read_points
    rclpy.init(); node=Node('factory_mapping_'+kind)
    rates={'lidar':Rates(),'imu':Rates()}; last=[0.0]
    def stamp(m): return m.header.stamp.sec+m.header.stamp.nanosec*1e-9
    def cloud(m):
        if kind=='monitor': rates['lidar'].add(stamp(m),m.width*m.height); return
        p=config['system']['preview']; t=time.monotonic()
        if t-last[0]<1/p['hz']: return
        last[0]=t
        fields={f.name for f in m.fields}; names=['x','y','z']+(['intensity'] if 'intensity' in fields else [])
        raw=read_points(m,field_names=names,skip_nans=True)
        if isinstance(raw,np.ndarray) and raw.dtype.names: arr=np.column_stack([raw[k] for k in names])
        else: arr=np.array(list(raw),dtype=np.float32).reshape(-1,len(names))
        if len(names)==3: arr=np.column_stack([arr,np.zeros(len(arr))])
        payload=encode(arr,stamp(m),p['max_points'],p['voxel_size'])
        out=ROOT/'.state/preview.bin'; tmp=out.with_suffix('.tmp'); tmp.write_bytes(payload); tmp.replace(out)
    node.create_subscription(PointCloud2,sensor['points_topic'],cloud,qos_profile_sensor_data)
    if kind=='monitor':
        node.create_subscription(Imu,sensor['imu_topic'],lambda m:rates['imu'].add(stamp(m)),qos_profile_sensor_data)
        def report():
            topics=dict(node.get_topic_names_and_types())
            obj={'updated_at':time.time(),'topics':topics,'lidar':rates['lidar'].view(),'imu':rates['imu'].view()}
            for key,topic,expected in [('lidar',sensor['points_topic'],sensor['publish_freq']),('imu',sensor['imu_topic'],sensor['expected_imu_hz'])]:
                r=obj[key]
                r['state']='topic_missing' if topic not in topics else ('no_messages' if r['hz']==0 else ('rate_abnormal' if not expected*.7<=r['hz']<=expected*1.3 else 'healthy'))
            atomic_json(ROOT/'.state/health.json',obj)
        node.create_timer(1.0,report)
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

def monitor(): run('monitor')
def preview(): run('preview')


def run_camera(kind):
    config=load();c=config['camera']
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data, QoSProfile
    from sensor_msgs.msg import Image,CameraInfo
    from .camera import CameraMetrics
    rclpy.init();node=Node('factory_mapping_'+kind);metrics=CameraMetrics(c);last=[0.0]
    if kind=='camera_preview':
        import cv2
        from cv_bridge import CvBridge
        bridge=CvBridge()
    def image(m):
        stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        if kind=='camera_monitor':metrics.image(stamp,m.width,m.height,m.header.frame_id);return
        t=time.monotonic()
        if t-last[0]<1/c['preview_hz']:return
        last[0]=t
        try:
            rgb=bridge.imgmsg_to_cv2(m,desired_encoding='bgr8')
            scale=min(1,c['preview_max_width']/rgb.shape[1]);rgb=cv2.resize(rgb,(max(1,int(rgb.shape[1]*scale)),max(1,int(rgb.shape[0]*scale))))
            ok,encoded=cv2.imencode('.jpg',rgb,[cv2.IMWRITE_JPEG_QUALITY,75])
            if not ok:raise ValueError('JPEG encoding failed')
            out=ROOT/'.state/camera_preview.jpg';tmp=out.with_suffix('.tmp');tmp.write_bytes(encoded.tobytes());tmp.replace(out)
        except Exception as e:node.get_logger().warning('Preview unavailable: '+str(e))
    node.create_subscription(Image,c['image_topic'],image,QoSProfile(depth=5) if kind=='camera_monitor' else qos_profile_sensor_data)
    if kind=='camera_monitor':
        node.create_subscription(CameraInfo,c['camera_info_topic'],lambda m:metrics.info(m.width,m.height,m.k,m.d,m.distortion_model,m.header.frame_id),QoSProfile(depth=5))
        node.create_timer(.5,lambda:atomic_json(ROOT/'.state/camera_health.json',dict(updated_at=time.time(),**metrics.view())))
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


def camera_monitor():run_camera('camera_monitor')
def camera_preview():run_camera('camera_preview')
