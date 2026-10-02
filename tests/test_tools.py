import json, shutil, zipfile
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

def test_project_archive_preserves_bag_map_and_settings(root,tmp_path):
    sessions=Sessions(root);session=sessions.create('portable','',load(root));folder=sessions.get(session['id'])
    (folder/'raw_bag').mkdir();(folder/'raw_bag/raw_bag_0.db3').write_bytes(b'raw camera lidar and imu')
    output=folder/'processing/run_001/glim_dump';output.mkdir(parents=True)
    (output/'graph.bin').write_bytes(b'portable map graph');(folder/'exports/final.ply').write_bytes(b'ply result')
    archive=tmp_path/'project.zip';sessions.export_archive(session['id'],archive)
    target=tmp_path/'target';shutil.copytree(root/'config',target/'config')
    before=(target/'config/calibration/camera_intrinsics.yaml').read_bytes()
    imported=Sessions(target).import_archive(archive);restored=Sessions(target).get(imported['id'])
    assert imported['id']==session['id'] and imported['name']==session['name']
    assert (restored/'raw_bag/raw_bag_0.db3').read_bytes()==b'raw camera lidar and imu'
    assert (restored/'processing/run_001/glim_dump/graph.bin').read_bytes()==b'portable map graph'
    assert (restored/'exports/final.ply').read_bytes()==b'ply result'
    assert (restored/'config_snapshot/camera/dfk33ux287.yaml').read_bytes()==(folder/'config_snapshot/camera/dfk33ux287.yaml').read_bytes()
    assert (restored/'active_config.json').read_bytes()==(folder/'active_config.json').read_bytes()
    assert (target/'config/calibration/camera_intrinsics.yaml').read_bytes()==before
    with pytest.raises(ValueError,match='already exists'):Sessions(target).import_archive(archive)

def test_project_archive_rejects_traversal_tampering_and_symlinks(root,tmp_path):
    sessions=Sessions(root);session=sessions.create('untrusted','',load(root));folder=sessions.get(session['id'])
    archive=tmp_path/'good.zip';sessions.export_archive(session['id'],archive)
    with zipfile.ZipFile(archive) as original:
        manifest=json.loads(original.read('manifest.json'))
        entries={name:original.read(name) for name in original.namelist() if name!='manifest.json'}
    bad=tmp_path/'bad.zip'
    manifest['files'][0]['path']='../escape'
    with zipfile.ZipFile(bad,'w') as output:
        for name,data in entries.items():output.writestr(name,data)
        output.writestr('manifest.json',json.dumps(manifest))
    target=tmp_path/'target'
    with pytest.raises(ValueError,match='Unsafe project path'):Sessions(target).import_archive(bad)
    assert not (target/'data/escape').exists()
    manifest['files'][0]['path']=next(iter(entries)).removeprefix('session/')
    with zipfile.ZipFile(bad,'w') as output:
        for name,data in entries.items():output.writestr(name,(b'X'+data[1:]) if name==next(iter(entries)) else data)
        output.writestr('manifest.json',json.dumps(manifest))
    with pytest.raises(ValueError,match='checksum mismatch'):Sessions(target).import_archive(bad)
    assert not (target/'data/sessions'/session['id']).exists()
    (folder/'outside').symlink_to(tmp_path)
    with pytest.raises(ValueError,match='symlink'):sessions.export_archive(session['id'],tmp_path/'unsafe.zip')
