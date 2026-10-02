from pathlib import Path
import ipaddress, os, re, math
import yaml
from .camera_config import validate_camera, config_path

ROOT = Path(os.environ.get('FACTORY_MAPPING_ROOT', Path(__file__).resolve().parents[3])).resolve()
PRESETS = ('jetson_cpu', 'jetson_gpu', 'offline_quality')

def load(root=ROOT):
    system = yaml.safe_load((root/'config/system.yaml').read_text())
    sensor = yaml.safe_load((root/'config/livox/mid360.yaml').read_text())
    validate_sensor(sensor)
    if not 0 <= system.get('offline_ros_domain_id',230) <= 232 or system.get('offline_ros_domain_id',230)==sensor['ros_domain_id']: raise ValueError('Offline ROS domain must be valid and different from acquisition')
    if system['preset'] not in PRESETS: raise ValueError('Unknown GLIM preset')
    if system['loop_closure']['enabled'] not in (False, 'scan_context'): raise ValueError('Invalid loop closure mode')
    if type(system['camera']['enabled']) is not bool: raise ValueError('camera.enabled must be boolean')
    p = system['preview']
    if not 1 <= p['hz'] <= 5 or not 100 <= p['max_points'] <= 100000 or not 0.01 <= p['voxel_size'] <= 5:
        raise ValueError('Preview limits: 1–5 Hz, 100–100000 points, 0.01–5 m voxels')
    result={'system': system, 'sensor': sensor}
    camera_file=root/'config/camera/dfk33ux287.yaml'
    if camera_file.exists():
        camera=yaml.safe_load(camera_file.read_text())
        validate_camera(camera,sensor)
        for key in ('intrinsics_file','extrinsics_file'): config_path(root,camera[key])
        result['camera']=camera
    elif system['camera']['enabled']: raise ValueError('Camera configuration is missing')
    return result

def validate_sensor(s):
    for key in ('lidar_ip', 'host_ip'): ipaddress.IPv4Address(s[key])
    if not re.fullmatch(r'[a-zA-Z0-9_.:-]{1,32}', s['interface']): raise ValueError('Invalid interface')
    for key in ('points_topic', 'imu_topic'):
        if not re.fullmatch(r'/[A-Za-z_][A-Za-z0-9_/]*', s[key]): raise ValueError('Invalid ROS topic')
    if s['points_topic'] == s['imu_topic']: raise ValueError('Sensor topics must differ')
    if not 0 <= s['ros_domain_id'] <= 232: raise ValueError('ROS domain must be 0–232')
    if s['publish_freq'] not in (5, 10, 20, 50, 100): raise ValueError('Unsupported publish frequency')
    t=s['T_lidar_imu']
    if len(t)!=7 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in t) or abs(sum(v*v for v in t[3:])-1)>0.01: raise ValueError('T_lidar_imu must contain finite XYZ and a unit quaternion')
    if s['imu_frame_id'] != 'livox_frame': raise ValueError('Pinned official driver fixes the IMU frame to livox_frame')
