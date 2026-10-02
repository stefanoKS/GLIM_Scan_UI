"""Official GLIM tools, launched against derived working copies."""
import json, shutil, uuid
from pathlib import Path
from .storage import atomic_json, now, read_json

CATALOG = [
    dict(id='offline_viewer', label='Loop closure, merging & optimization', executable='offline_viewer', display=True,
         features=['Manual loop constraints', 'Multi-session alignment and merging', 'Plane bundle adjustment constraints', 'Find overlapping submaps', 'Graph recovery', 'Global optimization', 'Trajectory and factor inspection', 'PLY export']),
    dict(id='map_editor', label='Object segmentation & map cleanup', executable='map_editor', display=True,
         features=['MinCut object segmentation', 'Region-growing plane segmentation', 'Gizmo selection', 'Radius selection', 'Outlier removal', 'Point annotation and removal', 'Save edited submaps']),
    dict(id='validator',label='Upstream sensor data validator',executable='validator_node',display=False,
         features=['IMU and point timestamp validation', 'Sensor stream diagnostics']),
]

def validate_dump(path):
    if not (path/'graph.bin').is_file() or not (path/'graph.txt').is_file(): raise ValueError('A saved GLIM dump is required')
    if any(p.is_symlink() for p in path.rglob('*')): raise ValueError('Symlinks are not accepted inside an editable dump')

def prepare(root,primary_session,primary_run,sources,kind):
    if kind not in ('offline_viewer','map_editor'): raise ValueError('Unsupported map editing tool')
    if kind=='map_editor' and len(sources)!=1: raise ValueError('Merge sessions with the offline viewer before segmentation')
    for source in sources: validate_dump(source['dump'])
    workspace=primary_session/'edits'/('edit_'+uuid.uuid4().hex[:12]); workspace.mkdir(parents=True)
    copies=[]
    for i,source in enumerate(sources):
        copy=workspace/f'map_{i+1:02d}'; shutil.copytree(source['dump'],copy)
        cfg=copy/'config'
        if not cfg.exists(): shutil.copytree(source['config'],cfg)
        # offline_viewer loads the dump's own config. Enable the upstream CPU
        # global optimizer for portable desktop editing, even for GPU-origin maps.
        global_config=json.loads((cfg/'config.json').read_text())
        global_config['global']['config_global_mapping']='config_global_mapping_cpu.json'
        atomic_json(cfg/'config.json',global_config)
        shutil.copy2(root/'config/glim/jetson_cpu/config_global_mapping_cpu.json',cfg/'config_global_mapping_cpu.json')
        view=read_json(cfg/'config_viewer.json',{})
        for v in view.values():
            if isinstance(v,dict): v.update(viewer_width=1280,viewer_height=720)
        atomic_json(cfg/'config_viewer.json',view)
        copies.append(str(copy))
    metadata=dict(id=workspace.name,tool=kind,state='prepared',created_at=now(),source_session=primary_session.name,source_run=primary_run,maps=copies,sources=[{k:str(v) for k,v in s.items()} for s in sources],save_target=str(workspace/'saved_map'),note='Work on the copied maps. Native Save is required; closing a window does not save changes.')
    atomic_json(workspace/'workspace.json',metadata)
    return workspace,metadata

def command(kind,dump,config=None):
    if kind=='offline_viewer':return ['ros2','run','glim_ros','offline_viewer',str(dump),'--config_path',str(config or dump/'config')]
    if kind=='map_editor':return ['ros2','run','glim_ros','map_editor',str(dump)]
    raise ValueError('Unknown tool')

def capabilities(root):
    import os
    result=[]
    for entry in CATALOG:
        path=root/'ros2_ws/install/glim_ros/lib/glim_ros'/entry['executable']
        result.append({**entry,'installed':path.is_file(),'display_available':bool(os.environ.get('DISPLAY')) if entry['display'] else True,'runs_on':'Server desktop / Jetson local display; native editor is not streamed to a remote browser'})
    return result
