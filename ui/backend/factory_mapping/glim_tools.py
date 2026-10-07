"""Official GLIM tools, launched against derived working copies."""
import hashlib, json, os, shutil, uuid
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

def fingerprint(path):
    path = Path(path)
    if not path.is_dir() or path.is_symlink(): raise ValueError('Saved map is unavailable')
    digest = hashlib.sha256()
    for entry in sorted(path.rglob('*')):
        if entry.is_symlink(): raise ValueError('Symlinks are not accepted inside a saved map')
        if entry.is_file():
            digest.update(str(entry.relative_to(path)).encode('utf-8'))
            with entry.open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()

def file_fingerprint(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink(): raise ValueError('Saved map trajectory is unavailable')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare_export_dump(source, destination):
    source, destination = Path(source), Path(destination)
    validate_dump(source)
    shutil.copytree(source, destination, copy_function=os.link)
    graph = destination/'graph.txt'
    lines = graph.read_text().splitlines()
    try:
        if len(lines) < 3 or not lines[2].startswith('num_matching_cost_factors:'):
            raise ValueError
        count = int(lines[2].split(':', 1)[1])
        if count < 0 or len(lines) < 3 + count or any(not line.startswith('matching_cost ') for line in lines[3:3 + count]):
            raise ValueError
    except (ValueError, IndexError) as error:
        shutil.rmtree(destination)
        raise ValueError('Saved map has an invalid matching-cost factor manifest') from error
    lines[2] = 'num_matching_cost_factors: 0'
    staged = graph.with_name('.graph.txt.tmp')
    staged.write_text('\n'.join(lines[:3] + lines[3 + count:]) + '\n')
    staged.replace(graph)
    return destination

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
    metadata=dict(id=workspace.name,tool=kind,state='prepared',created_at=now(),source_session=primary_session.name,source_run=primary_run,maps=copies,sources=[{k:str(v) for k,v in s.items()} for s in sources],save_target=str(workspace/'saved_map'),pose_policy='map_editor_fixed_poses' if kind=='map_editor' else 'offline_viewer_may_optimize',note='Work on the copied maps. Native Save As to saved_map is required; closing a window does not save changes.')
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
