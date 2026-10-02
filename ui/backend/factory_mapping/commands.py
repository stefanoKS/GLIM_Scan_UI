"""Commands inspected against pinned official source; no user-supplied commands."""
import json, os, shutil, sys, uuid
from .storage import atomic_json
from .config import PRESETS

def ros_env(config, offline=False):
    env=os.environ.copy()
    env['ROS_DOMAIN_ID']=str(config['system'].get('offline_ros_domain_id',230) if offline else config['sensor']['ros_domain_id'])
    if offline: env['ROS_LOCALHOST_ONLY']='1'
    return env

def driver(root,config):
    sensor=config['sensor']; obj=json.loads((root/'config/livox/upstream_mid360.json').read_text())
    for k in obj['MID360']['host_net_info']:
        if k.endswith('_ip') and k!='log_data_ip': obj['MID360']['host_net_info'][k]=sensor['host_ip']
    obj['lidar_configs'][0]['ip']=sensor['lidar_ip']
    out=root/'.state/MID360.generated.json'; atomic_json(out,obj)
    args=['ros2','run','livox_ros_driver2','livox_ros_driver2_node','--ros-args']
    for key,value in dict(xfer_format=0,multi_topic=0,data_src=0,publish_freq=float(sensor['publish_freq']),output_data_type=0,frame_id=sensor['frame_id'],user_config_path=str(out)).items(): args+=['-p',f'{key}:={value}']
    args+=['-r',f"/livox/lidar:={sensor['points_topic']}",'-r',f"/livox/imu:={sensor['imu_topic']}"]
    return args

def record(session,config):
    qos={config['sensor'][key]:dict(reliability='best_effort',durability='volatile',history='keep_last',depth=1000) for key in ('points_topic','imu_topic')}
    import yaml
    path=session/'config_snapshot/record_qos.yaml'; path.write_text(yaml.safe_dump(qos))
    return ['ros2','bag','record','--storage','sqlite3','--output',str(session/'raw_bag'),'--qos-profile-overrides-path',str(path),*qos.keys()]

def preset_snapshot(root,session,preset,output):
    if preset not in PRESETS: raise ValueError('Unknown preset')
    if output.exists(): raise ValueError('Configuration output already exists')
    shutil.copytree(root/'config/glim'/preset,output)
    # Always take topic names/extrinsics from acquisition, including after network config edits.
    config=json.loads((session/'active_config.json').read_text()); sensor=config['sensor']
    p=output/'config_ros.json'; obj=json.loads(p.read_text()); obj['glim_ros'].update(points_topic=sensor['points_topic'],imu_topic=sensor['imu_topic']); atomic_json(p,obj)
    p=output/'config_sensors.json'; obj=json.loads(p.read_text()); obj['sensors']['T_lidar_imu']=sensor['T_lidar_imu']; atomic_json(p,obj)
    return output

def glim(config_path,dump,bag=None):
    args=['ros2','run','glim_ros','glim_rosbag' if bag else 'glim_rosnode']
    if bag: args.append(str(bag))
    args+=['--ros-args','-p',f'config_path:={config_path}','-p',f'dump_path:={dump}']
    if bag:
        args+=['-p','auto_quit:=true']
        # GLIM rosbag also spins live subscriptions. Remap only ROS subscriptions;
        # direct bag filtering still uses the original names in config_ros.json.
        config=json.loads((config_path/'config_ros.json').read_text())['glim_ros']
        namespace='/factory_mapping_offline_'+uuid.uuid4().hex
        for field in ('imu_topic','points_topic','image_topic'):
            if config.get(field): args+=['-r',f'{config[field]}:={namespace}/{field}']
    return args

def export(dump,target,config_path):
    # This upstream binary requires an OpenGL display even with --export_path.
    return ['ros2','run','glim_ros','offline_viewer',str(dump),'--export_path',str(target),'--config_path',str(config_path)]
