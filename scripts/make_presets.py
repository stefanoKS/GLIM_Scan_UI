#!/usr/bin/env python3
"""Copy pinned upstream configs and apply only documented integration changes."""
import json, pathlib, shutil
import json5, yaml
root=pathlib.Path(__file__).resolve().parents[1]
sensor=yaml.safe_load((root/'config/livox/mid360.yaml').read_text())
for preset in ['jetson_cpu','jetson_gpu','offline_quality']:
    dest=root/'config/glim'/preset
    dest.mkdir(parents=True,exist_ok=True)
    for src in (root/'external/glim/config').glob('*.json'):
        obj=json5.loads(src.read_text())
        if src.name=='config.json' and preset=='jetson_cpu':
            obj['global'].update(config_odometry='config_odometry_cpu.json',config_sub_mapping='config_sub_mapping_passthrough.json',config_global_mapping='config_global_mapping_pose_graph.json')
        if src.name=='config_ros.json':
            obj['glim_ros'].update(points_topic=sensor['points_topic'],imu_topic=sensor['imu_topic'],acc_scale=9.80665,extension_modules=[],publish_imu2lidar=False)
        if src.name=='config_sensors.json':
            obj['sensors']['T_lidar_imu']=sensor['T_lidar_imu']
            obj['sensors'].update(autoconf_perpoint_times=False,perpoint_relative_time=True,perpoint_time_scale=1e-9)
        (dest/src.name).write_text(json.dumps(obj,indent=2)+'\n')
