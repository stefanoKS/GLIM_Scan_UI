import json
import pytest
from factory_mapping.config import load
from factory_mapping.storage import Sessions,atomic_json
from factory_mapping.glim_tools import prepare,command
from factory_mapping import commands

def test_offline_isolation_keeps_bag_topic_names(root):
    config=load(root);s=Sessions(root);m=s.create('test','',config);p=s.get(m['id'])
    cfg=commands.preset_snapshot(root,p,'jetson_cpu',p/'processing/config')
    args=commands.glim(cfg,p/'glim_dump',p/'raw_bag')
    assert any(a.startswith('/livox/imu:=/factory_mapping_offline_') for a in args)
    assert any(a.startswith('/livox/lidar:=/factory_mapping_offline_') for a in args)
    assert json.loads((cfg/'config_ros.json').read_text())['glim_ros']['imu_topic']=='/livox/imu'
    env=commands.ros_env(config,True)
    assert env['ROS_DOMAIN_ID']!=str(config['sensor']['ros_domain_id']) and env['ROS_LOCALHOST_ONLY']=='1'

def test_sensor_timestamp_convention(root):
    c=json.loads((root/'config/glim/jetson_cpu/config_sensors.json').read_text())['sensors']
    assert c['perpoint_relative_time'] is False and c['perpoint_time_scale']==1e-9
    assert c['T_lidar_imu'][:3]==[.011,.02329,-.04412]

def test_edit_workspace_is_independent(root):
    s=Sessions(root);m=s.create('test','',load(root));p=s.get(m['id']);dump=p/'processing/run_001/glim_dump';dump.mkdir(parents=True)
    (dump/'graph.bin').write_bytes(b'test graph');(dump/'graph.txt').write_text('num_submaps: 1')
    src=dict(session=m['id'],run='run_001',dump=dump,config=root/'config/glim/jetson_cpu')
    workspace,meta=prepare(root,p,'run_001',[src],'offline_viewer')
    copy=workspace/'map_01';(copy/'graph.bin').write_bytes(b'edited')
    assert (dump/'graph.bin').read_bytes()==b'test graph'
    assert json.loads((copy/'config/config.json').read_text())['global']['config_global_mapping']=='config_global_mapping_cpu.json'
    assert command('map_editor',copy)==['ros2','run','glim_ros','map_editor',str(copy)]
    with pytest.raises(ValueError):prepare(root,p,'run_001',[src,src],'map_editor')
