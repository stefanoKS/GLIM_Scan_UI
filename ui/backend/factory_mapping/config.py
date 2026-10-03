from pathlib import Path
import ipaddress, os, re, math, json, socket
import psutil
import yaml
from .camera_config import validate_camera, config_path

ROOT = Path(os.environ.get('FACTORY_MAPPING_ROOT', Path(__file__).resolve().parents[3])).resolve()
PRESETS = ('jetson_cpu', 'jetson_gpu', 'offline_quality', 'pc_dense')

def load(root=ROOT):
    system = yaml.safe_load((root/'config/system.yaml').read_text())
    preference=root/'.state/camera_enabled.json'
    if preference.is_file():
        enabled=json.loads(preference.read_text()).get('enabled')
        if type(enabled) is not bool: raise ValueError('Local camera preference must be boolean')
        system['camera']['enabled']=enabled
    sensor = yaml.safe_load((root/'config/livox/mid360.yaml').read_text())
    validate_sensor(sensor)
    sensor['interface_setting'] = sensor['interface']
    sensor['interface'], sensor['host_ip'] = wired_connection(sensor)
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

def wired_host_ip(sensor):
    lidar_ip=ipaddress.IPv4Address(sensor['lidar_ip'])
    addresses=psutil.net_if_addrs().get(sensor['interface'],[])
    matches=[address.address for address in addresses if address.family==socket.AF_INET and address.netmask and lidar_ip in ipaddress.IPv4Network(f'{address.address}/{address.netmask}',strict=False)]
    return matches[0] if len(matches)==1 else None

def wired_connection(sensor, sys_net=Path('/sys/class/net')):
    """Resolve one physical, up Ethernet adapter on the sensor subnet.

    Ambiguous networks fail closed; no Wi-Fi, bridges or OS route changes.
    An explicit interface remains available under Advanced Diagnostics.
    """
    setting=sensor.get('interface_setting',sensor['interface'])
    if setting!='auto': return setting,wired_host_ip({**sensor,'interface':setting})
    stats=psutil.net_if_stats();candidates=[]
    for name in psutil.net_if_addrs():
        node=sys_net/name
        try:
            ethernet=(node/'device').exists() and (node/'type').read_text().strip()=='1' and not (node/'wireless').exists()
        except OSError: ethernet=False
        if not ethernet or name not in stats or not stats[name].isup: continue
        address=wired_host_ip({**sensor,'interface':name})
        if address: candidates.append((name,address))
    return candidates[0] if len(candidates)==1 else ('auto',None)

def validate_sensor(s):
    ipaddress.IPv4Address(s['lidar_ip'])
    if not re.fullmatch(r'[a-zA-Z0-9_.:-]{1,32}', s['interface']): raise ValueError('Invalid interface')
    for key in ('points_topic', 'imu_topic'):
        if not re.fullmatch(r'/[A-Za-z_][A-Za-z0-9_/]*', s[key]): raise ValueError('Invalid ROS topic')
    if s['points_topic'] == s['imu_topic']: raise ValueError('Sensor topics must differ')
    if not 0 <= s['ros_domain_id'] <= 232: raise ValueError('ROS domain must be 0–232')
    if s['publish_freq'] not in (5, 10, 20, 50, 100): raise ValueError('Unsupported publish frequency')
    t=s['T_lidar_imu']
    if len(t)!=7 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in t) or abs(sum(v*v for v in t[3:])-1)>0.01: raise ValueError('T_lidar_imu must contain finite XYZ and a unit quaternion')
    if s['imu_frame_id'] != 'livox_frame': raise ValueError('Pinned official driver fixes the IMU frame to livox_frame')
