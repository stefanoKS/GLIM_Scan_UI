import asyncio
import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import numpy as np
import pytest
from factory_mapping import nksr_jobs as jobs
from factory_mapping.nksr_worker import (parser, reconstruction_kwargs, classify, WorkerError,
    load_input, execute)
from factory_mapping.nksr_mesh import write_mesh, inspect_mesh, merge_meshes
from factory_mapping.reconstruction_jobs import preparation_state
from factory_mapping.api import MeshRequest
from factory_mapping.storage import atomic_json, read_json


def test_prepared_never_completed(tmp_path):
    run=tmp_path/'run';(run/'input').mkdir(parents=True)
    assert preparation_state(run,{'state':'completed'})=='NOT_PREPARED'
    (run/'input/nksr_input.npz').touch()
    assert preparation_state(run,{'state':'completed'})=='PREPARED'
    assert jobs.mesh_state(run)['state']=='NOT_RECONSTRUCTED'
    atomic_json(run/'mesh_job.json',{'state':'RUNNING'})
    atomic_json(run/'nksr_progress.json',{'stage':'COMPLETED'})
    assert jobs.mesh_state(run)['stage']=='VALIDATING_MESH'


@pytest.mark.parametrize('mode',['full','chunked'])
def test_defaults_and_upstream_arguments(mode):
    settings=parser().parse_args([])
    settings.detail_level=.7
    fn=object();args=reconstruction_kwargs(settings,mode,5,fn)
    assert settings.normal_knn==64 and settings.normal_drop_angle_deg==85
    assert settings.mode=='auto' and settings.device=='auto'
    assert args['preprocess_fn'] is fn
    assert args['approx_kernel_grad'] and args['fused_mode']
    if mode=='full':
        assert args['detail_level'] is None and args['voxel_size']==.02
        assert args['solver_tol']==1e-4
    else:
        assert args['detail_level'] is None and args['chunk_size']==5
        assert 'voxel_size' not in args and 'solver_tol' not in args
    assert MeshRequest().preparation_voxel_size_m==.01


def test_input_validation(tmp_path):
    path=tmp_path/'input with spaces.npz'
    np.savez(path,points=np.zeros((3,3)),sensor_origins=np.ones((3,3)))
    assert load_input(path)[0].shape==(3,3)
    for points, sensors in [(np.zeros((0,3)),np.zeros((0,3))), (np.zeros((3,3)),np.zeros((2,3))),
                            (np.full((2,3),np.nan),np.zeros((2,3)))]:
        np.savez(path,points=points,sensor_origins=sensors)
        with pytest.raises(WorkerError,match='(Expected|nonfinite)') as error: load_input(path)
        assert error.value.code=='INPUT_INVALID'


def triangle(output):
    v=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=np.float32)
    f=np.array([[0,1,2]],dtype=np.int32)
    stats=write_mesh(output/'mesh.ply',v,f)
    atomic_json(output/'nksr_metadata.json',stats)
    return stats


def test_only_valid_triangle_mesh_completes(tmp_path):
    stats=triangle(tmp_path)
    assert jobs.validate_completed(tmp_path,0)['face_count']==1
    with pytest.raises(ValueError): jobs.validate_completed(tmp_path,1)
    (tmp_path/'mesh.ply').write_text('ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n')
    with pytest.raises((KeyError,ValueError)): jobs.validate_completed(tmp_path,0)
    with pytest.raises(ValueError):write_mesh(tmp_path/'bad.ply',np.zeros((3,3)),np.array([[0,1,9]]))


def test_streamed_mesh_merge_preserves_coordinates_and_offsets(tmp_path):
    from plyfile import PlyData
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=np.float32)
    faces=np.array([[0,1,2]],dtype=np.int32)
    sources=[tmp_path/'first.ply',tmp_path/'second.ply']
    for index,source in enumerate(sources): write_mesh(source,vertices+index*10,faces)
    output=tmp_path/'combined.ply'
    stats=merge_meshes(output,sources)
    assert stats['vertex_count']==6 and stats['face_count']==2
    assert stats['bounding_box_min']==[0,0,0] and stats['bounding_box_max']==[11,11,10]
    mesh=PlyData.read(str(output),known_list_len={'face':{'vertex_indices':3}})
    np.testing.assert_array_equal(mesh['face']['vertex_indices'],[[0,1,2],[3,4,5]])
    np.testing.assert_array_equal(mesh['vertex']['x'],[0,1,0,10,11,10])
    with pytest.raises(ValueError,match='already exists'): merge_meshes(output,sources)
    with pytest.raises(ValueError,match='No tile'): merge_meshes(tmp_path/'empty.ply',[])


def test_inspect_mesh_validates_across_blocks(tmp_path):
    vertices=np.zeros((65539,3),dtype=np.float32)
    vertices[-1]=[4,5,6]
    path=tmp_path/'large.ply'
    write_mesh(path,vertices,np.array([[65536,65537,65538]],dtype=np.int32))
    assert inspect_mesh(path)['bounding_box_max']==[4,5,6]


def test_tile_partition_is_spatial_paired_and_nonoverlapping(tmp_path):
    from factory_mapping.nksr_tiled import partition_input
    points=np.array([[-5,0,0],[-.01,0,0],[0,0,0],[4.99,0,0],[5,0,0]],dtype=np.float32)
    sensors=points+10
    source=tmp_path/'input.npz';np.savez_compressed(source,points=points,sensor_origins=sensors)
    tiles,stats=partition_input(source,tmp_path,5,lambda *args,**kwargs:None)
    assert [tile['cell'] for tile in tiles]==[[-1,0,0],[0,0,0],[1,0,0]]
    assert [tile['points'] for tile in tiles]==[2,2,1] and stats['input_points']==5
    recovered=np.concatenate([np.fromfile(tmp_path/(tile['id']+'.bin'),dtype='<f4').reshape(-1,6) for tile in tiles])
    np.testing.assert_array_equal(recovered[:,:3],points)
    np.testing.assert_array_equal(recovered[:,3:],sensors)


def test_low_ram_tiles_are_sequential_and_merge(tmp_path,monkeypatch):
    from factory_mapping import nksr_tiled as tiled
    points=np.array([[0,0,0],[1,0,0],[0,1,0],[10,0,0],[11,0,0],[10,1,0],[20,0,0]],dtype=np.float32)
    source=tmp_path/'input.npz';np.savez(source,points=points,sensor_origins=points+3)
    original=source.read_bytes();calls=[];events=[]
    def run(args):
        assert '--tile-worker' in args and args[args.index('--mode')+1]=='full'
        with np.load(args[args.index('--input')+1]) as data:
            vertices=data['points'];np.testing.assert_array_equal(data['sensor_origins'],vertices+3)
        output=Path(args[args.index('--output')+1])
        if calls: assert calls[-1].is_file()
        write_mesh(output,vertices,np.array([[0,1,2]],dtype=np.int32))
        atomic_json(Path(args[args.index('--metadata')+1]),{})
        calls.append(output)
        return 0
    monkeypatch.setattr(tiled,'run_tile',run)
    settings=parser().parse_args(['--input',str(source),'--output',str(tmp_path/'output/mesh.ply'),
                                  '--mode','low_ram','--tile-size','5','--normal-knn','3'])
    assert tiled.run_tiled(settings,lambda stage,**kw:events.append((stage,kw)))==0
    assert len(calls)==2 and inspect_mesh(settings.output)['face_count']==2
    metadata=read_json(settings.output.parent/'nksr_metadata.json')
    assert metadata['actual_mode']=='low_ram' and metadata['skipped_tiles']==1
    assert metadata['boundary_stitching'] is False and metadata['tile_size_m']==5
    assert events[-1][0]=='COMPLETED' and source.read_bytes()==original
    assert not list(settings.output.parent.glob('nksr-tiles-*'))


def test_low_ram_tile_failure_stops_before_merge(tmp_path,monkeypatch):
    from factory_mapping import nksr_tiled as tiled
    source=tmp_path/'input.npz'
    points=np.array([[0,0,0],[1,0,0],[0,1,0],[10,0,0],[11,0,0],[10,1,0]],dtype=np.float32)
    np.savez(source,points=points,sensor_origins=points+3)
    calls=[]
    def run(args): calls.append(args);return -9
    monkeypatch.setattr(tiled,'run_tile',run)
    settings=parser().parse_args(['--input',str(source),'--output',str(tmp_path/'out/mesh.ply'),
                                  '--mode','low_ram','--normal-knn','3'])
    with pytest.raises(WorkerError,match='smaller tile size'): tiled.run_tiled(settings,lambda *args,**kw:None)
    assert len(calls)==1 and not settings.output.exists()
    assert not list(settings.output.parent.glob('nksr-tiles-*'))
    assert read_json(settings.output.parent/'tiles.json')['tiles'][0]['state']=='FAILED'


def low_ram_tile_run(tmp_path,monkeypatch,output_mode,chunk_size='10'):
    """Run one Low RAM job with a stubbed per-tile worker and record what happened."""
    from factory_mapping import nksr_tiled as tiled
    points=np.array([[0,0,0],[1,0,0],[0,1,0],[10,0,0],[11,0,0],[10,1,0]],dtype=np.float32)
    source=tmp_path/'input.npz';np.savez(source,points=points,sensor_origins=points+3)
    calls=[];merges=[];events=[]
    def run(args):
        with np.load(args[args.index('--input')+1]) as data: vertices=data['points']
        output=Path(args[args.index('--output')+1])
        if calls: assert calls[-1].is_file()  # one isolated subprocess finishes before the next starts
        write_mesh(output,vertices,np.array([[0,1,2]],dtype=np.int32))
        atomic_json(Path(args[args.index('--metadata')+1]),{})
        calls.append(output);return 0
    original=tiled.merge_meshes
    def merge(output,sources):
        merges.append(list(sources))
        return original(output,sources)
    monkeypatch.setattr(tiled,'run_tile',run)
    monkeypatch.setattr(tiled,'merge_meshes',merge)
    settings=parser().parse_args(['--input',str(source),'--output',str(tmp_path/'output/mesh.ply'),
                                  '--mode','low_ram','--mesh-output-mode',output_mode,'--normal-knn','3']
                                 +([] if chunk_size is None else ['--chunk-size',chunk_size]))
    assert tiled.run_tiled(settings,lambda stage,**kw:events.append((stage,kw)))==0
    return settings,calls,merges,events


@pytest.mark.parametrize('output_mode,merged',[('merged',True),('chunks',False),('both',True)])
def test_low_ram_output_mode_controls_merging(output_mode,merged,tmp_path,monkeypatch):
    settings,calls,merges,events=low_ram_tile_run(tmp_path,monkeypatch,output_mode)
    metadata=read_json(settings.output.parent/'nksr_metadata.json')
    assert len(calls)==2 and metadata['completed_tiles']==2 and metadata['skipped_tiles']==0
    assert metadata['mesh_output_mode']==output_mode
    assert metadata['effective_chunk_size_m']==10.0 and metadata['tile_size_m']==10.0
    assert metadata['actual_mode']=='low_ram' and metadata['boundary_stitching'] is False
    # The independent tile meshes always exist in their existing directory.
    assert (settings.output.parent/'tiles/tile_000001/mesh.ply').is_file()
    assert (settings.output.parent/'tiles/tile_000002/mesh.ply').is_file()
    if merged:
        assert len(merges)==1 and settings.output.is_file()
        assert metadata['face_count']==2 and metadata['mesh_file_size']>0
    else:
        assert merges==[] and not settings.output.exists()
        assert metadata['face_count']==2 and metadata['mesh_file_size'] is None
        # Separate output must never be duplicated into the Full/Chunked chunk directory.
        assert not (settings.output.parent/'mesh_chunks').exists()
    assert metadata['total_faces']==2 and metadata['output_bytes']>0
    state=read_json(settings.output.parent/'tiles.json')
    assert state['mesh_output_mode']==output_mode and state['total_faces']==2
    assert state['effective_chunk_size_m']==10.0 and state['chunk_size_source']=='user'
    assert ('MERGING_MESHES' in [stage for stage,_ in events]) is merged
    assert events[-1][0]=='COMPLETED' and not list(settings.output.parent.glob('nksr-tiles-*'))


def test_low_ram_default_tile_edge_stays_five_metres(tmp_path,monkeypatch):
    settings,_,_,_=low_ram_tile_run(tmp_path,monkeypatch,'merged',chunk_size=None)
    metadata=read_json(settings.output.parent/'nksr_metadata.json')
    assert metadata['effective_chunk_size_m']==5.0 and metadata['chunk_size_source']=='default'
    assert metadata['requested_chunk_size_m'] is None
    assert len(read_json(settings.output.parent/'tiles.json')['tiles'])==2


def test_low_ram_chunks_skips_merging_but_keeps_every_tile(tmp_path,monkeypatch):
    settings,calls,merges,events=low_ram_tile_run(tmp_path,monkeypatch,'chunks')
    assert merges==[] and not settings.output.exists()
    assert [path.parent.name for path in calls]==['tile_000001','tile_000002']
    assert not list(settings.output.parent.glob('nksr-tiles-*'))
    assert read_json(settings.output.parent/'tiles.json')['tiles'][0]['state']=='COMPLETED'


def test_tile_worker_bounds_extraction_without_changing_original_modes():
    torch,nksr,reconstructor,_,_=fake_runtime(0)
    captured=[]
    field=SimpleNamespace(to_=lambda device:captured.append(device),
                          extract_dual_mesh=lambda **kwargs:(captured.append(kwargs) or
                              SimpleNamespace(v=np.zeros((3,3)),f=np.array([[0,1,2]]))))
    reconstructor.reconstruct=lambda *args,**kwargs:field
    for flag in ([],['--tile-worker']):
        settings=parser().parse_args(['--mode','full',*flag])
        execute(np.ones((70,3)),np.ones((70,3)),settings,lambda *args,**kwargs:None,
                torch,nksr,torch.device('cuda'),reconstructor)
    assert captured==[{'mise_iter':1},'cpu:0',{'mise_iter':1,'max_points':100000}]


def test_tile_subprocess_is_reaped_on_cancellation(monkeypatch):
    from factory_mapping import nksr_tiled as tiled
    events=[]
    class Process:
        def wait(self,timeout=None):
            events.append(('wait',timeout))
            if timeout is None: raise WorkerError('CANCELLED','cancelled')
        def poll(self): return None
        def terminate(self): events.append('terminate')
    monkeypatch.setattr(tiled.subprocess,'Popen',lambda args:Process())
    with pytest.raises(WorkerError,match='cancelled'): tiled.run_tile(['worker'])
    assert events==[('wait',None),'terminate',('wait',5)]


def test_low_ram_coordinator_does_not_load_model_or_whole_input(tmp_path,monkeypatch):
    from factory_mapping import nksr_worker as worker, nksr_tiled as tiled
    calls=[]
    monkeypatch.setattr(worker.sys,'argv',['worker','--mode','low_ram','--input','source.npz',
                                         '--output',str(tmp_path/'mesh.ply')])
    monkeypatch.setattr(worker.signal,'signal',lambda *args:None)
    def forbidden(*args): pytest.fail('Coordinator must not load the model or whole input')
    monkeypatch.setattr(worker,'runtime',forbidden);monkeypatch.setattr(worker,'load_input',forbidden)
    monkeypatch.setattr(tiled,'run_tiled',lambda settings,event:(calls.append(settings.mode) or 0))
    assert worker.main()==0 and calls==['low_ram']


def test_empty_tile_does_not_fail_worker(tmp_path,monkeypatch):
    from factory_mapping import nksr_worker as worker
    source=tmp_path/'input.npz';np.savez(source,points=np.ones((70,3)),sensor_origins=np.ones((70,3)))
    torch,nksr,reconstructor,_,_=fake_runtime(0)
    reconstructor.reconstruct=lambda *args,**kwargs:None
    monkeypatch.setattr(worker.sys,'argv',['worker','--mode','full','--tile-worker','--input',str(source),
                                         '--output',str(tmp_path/'mesh.ply')])
    monkeypatch.setattr(worker.signal,'signal',lambda *args:None)
    monkeypatch.setattr(worker,'runtime',lambda device:(torch,nksr,torch.device('cuda'),{}))
    monkeypatch.setattr(worker,'load_model',lambda *args:reconstructor)
    assert worker.main()==0
    assert read_json(tmp_path/'nksr_metadata.json')['status']=='EMPTY_TILE'
    assert not (tmp_path/'mesh.ply').exists()


@pytest.fixture
def prepared(root,monkeypatch):
    from factory_mapping.service import Service
    service=Service(root,True)
    item=service.sessions.create('NKSR','',service.config)
    session=service.sessions.get(item['id'])
    run=session/'reconstruction/run_0123456789ab';(run/'input').mkdir(parents=True)
    np.savez(run/'input/nksr_input.npz',points=np.zeros((3,3)),sensor_origins=np.ones((3,3)))
    atomic_json(run/'job.json',{'state':'completed','voxel_size_m':.01})
    python=root/'NKSR env with spaces/bin/python';python.parent.mkdir(parents=True);python.touch()
    monkeypatch.setenv('NKSR_PYTHON',str(python));service.mock=False
    return service,item['id'],run,python


@pytest.mark.parametrize('outcome',['success','vertex_only','nonzero','cancelled'])
def test_orchestration_and_isolation(prepared,monkeypatch,outcome):
    service,sid,run,python=prepared
    source=(run/'input/nksr_input.npz').read_bytes()
    calls=[]
    async def start(key,args,log,env,done): calls.append((key,args,log,env,done))
    monkeypatch.setattr(service.pm,'start',start)
    asyncio.run(jobs.reconstruct(service,sid,run.name,MeshRequest().model_dump()))
    key,args,log,env,done=calls[0]
    assert key=='nksr' and args[0]==str(python) and Path(args[1]).name=='nksr_worker.py'
    assert args[args.index('--input')+1]==str(run/'input/nksr_input.npz')
    assert '--voxel-size' not in args and 'PYTHONPATH' not in env
    assert read_json(run/'mesh_job.json')['state']=='RUNNING'
    if outcome=='success': triangle(run/'output')
    elif outcome=='vertex_only':
        (run/'output').mkdir();(run/'output/mesh.ply').write_text('ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n')
    elif outcome=='nonzero': atomic_json(run/'nksr_progress.json',{'error_type':'CUDA_OOM','message':'GPU memory exhausted'})
    asyncio.run(done({'state':'cancelled' if outcome=='cancelled' else 'failed' if outcome=='nonzero' else 'completed',
                      'returncode':1 if outcome=='nonzero' else 0}))
    result=read_json(run/'mesh_job.json')
    assert result['state']=={'success':'COMPLETED','cancelled':'CANCELLED'}.get(outcome,'FAILED')
    if outcome=='nonzero': assert result['error_type']=='CUDA_OOM'
    assert (run/'input/nksr_input.npz').read_bytes()==source


def test_stale_preparation_and_missing_runtime(prepared,monkeypatch):
    service,sid,run,python=prepared
    with pytest.raises(ValueError,match='stale'):
        asyncio.run(jobs.reconstruct(service,sid,run.name,MeshRequest(preparation_voxel_size_m=.02).model_dump()))
    python.unlink()
    assert jobs.health(service)['status']=='NKSR_NOT_INSTALLED'
    with pytest.raises(ValueError,match='NKSR_NOT_INSTALLED'):
        asyncio.run(jobs.reconstruct(service,sid,run.name,MeshRequest().model_dump()))
    assert preparation_state(run,read_json(run/'job.json'))=='PREPARED'


def test_low_ram_settings_reach_worker(prepared,monkeypatch):
    service,sid,run,_=prepared
    calls=[]
    async def start(key,args,*rest): calls.append(args)
    monkeypatch.setattr(service.pm,'start',start)
    # A legacy client sends tile_size; it must reach the worker as the shared chunk size.
    settings=MeshRequest(mode='low_ram',tile_size=2.5).model_dump()
    asyncio.run(jobs.reconstruct(service,sid,run.name,settings))
    assert calls[0][calls[0].index('--mode')+1]=='low_ram'
    assert calls[0][calls[0].index('--chunk-size')+1]=='2.5'
    assert calls[0][calls[0].index('--chunk-size-source')+1]=='legacy_tile_size'
    recorded=read_json(run/'mesh_job.json')['settings']
    assert recorded['chunk_size']==2.5 and recorded['chunk_size_source']=='legacy_tile_size'
    assert 'tile_size' not in recorded


def test_import_only_is_not_ready(prepared):
    service,_,_,_=prepared
    assert jobs.health(service)['status']=='UNVERIFIED'


def fake_runtime(failures, mesh=None):
    """Tiny tensor facade tests orchestration only; never used in production."""
    class Tensor:
        def __init__(self,value): self.value=value
        def float(self): return self
        def to(self,device): return self
        def __mul__(self,scale): return Tensor(self.value*scale)
    cuda=SimpleNamespace(mem_get_info=lambda device:(8*1024**3,8*1024**3),empty_cache=lambda:None,
                         max_memory_allocated=lambda device:0,reset_peak_memory_stats=lambda device:None)
    torch=SimpleNamespace(device=lambda name:SimpleNamespace(type=name.split(':')[0]),from_numpy=Tensor,
                          cuda=cuda,inference_mode=contextlib.nullcontext)
    calls=[];normal_calls=[]
    if mesh is None: mesh=SimpleNamespace(v=np.zeros((3,3)),f=np.array([[0,1,2]]))
    field=SimpleNamespace(extract_dual_mesh=lambda **kw:mesh,to_=lambda device:None)
    def reconstruct(xyz,**kwargs):
        calls.append(dict(kwargs,xyz=xyz))
        assert kwargs['sensor'].value.shape==xyz.value.shape
        if len(calls)<=failures:raise RuntimeError('CUDA out of memory')
        return field
    def normals(*args): normal_calls.append(args);return 'normal_fn'
    nksr=SimpleNamespace(get_estimate_normal_preprocess_fn=normals)
    reconstructor=SimpleNamespace(reconstruct=reconstruct,network=SimpleNamespace(to=lambda device:None))
    return torch,nksr,reconstructor,calls,normal_calls


@pytest.mark.parametrize('mode,failures,expected_calls',[('auto',1,2),('auto',3,2),('full',1,1),('chunked',1,1)])
def test_bounded_oom_retry_and_sensor_pairing(mode,failures,expected_calls):
    torch,nksr,reconstructor,calls,normal_calls=fake_runtime(failures)
    settings=parser().parse_args(['--mode',mode,'--chunk-size','10'])
    args=(np.ones((70,3)),np.ones((70,3))*5,settings,lambda *a,**kw:None,torch,nksr,torch.device('cuda'),reconstructor)
    if mode=='auto' and failures==1:
        _,_,meta=execute(*args)
        assert meta['actual_mode']=='chunked' and meta['detail_level'] is None
        assert meta['chunk_size']==5 and len(meta['attempts'])==1
    else:
        with pytest.raises(RuntimeError,match='out of memory'):execute(*args)
    assert len(calls)==expected_calls and normal_calls==[(64,85.)]
    assert calls[0]['preprocess_fn']=='normal_fn'


@pytest.mark.parametrize('mode,failures', [('chunked',0), ('full',0), ('auto',1)])
def test_metric_resolution_through_worker_output(tmp_path,monkeypatch,capsys,mode,failures):
    from factory_mapping import nksr_worker as worker, nksr_mesh
    from plyfile import PlyData
    chunked=mode!='full'
    scale=5.0 if chunked else 1.0
    points=np.array([[10,20,30],[11,20,30],[10,21,30]],dtype=np.float32)
    sensors=points+np.array([2,-3,4],dtype=np.float32)
    faces=np.array([[0,1,2]],dtype=np.int32)
    mesh=SimpleNamespace(v=points*scale,f=faces.copy())
    torch,nksr,reconstructor,calls,_=fake_runtime(failures,mesh)
    source=tmp_path/'existing.npz';output=tmp_path/'mesh.ply'
    np.savez(source,points=points,sensor_origins=sensors)
    original=source.read_bytes()
    monkeypatch.setattr(worker.sys,'argv',['nksr_worker','--input',str(source),'--output',str(output),
                                         '--mode',mode,'--chunk-size','5'])
    monkeypatch.setattr(worker.signal,'signal',lambda *args:None)
    monkeypatch.setattr(worker,'runtime',lambda device:(torch,nksr,torch.device('cuda'),{}))
    monkeypatch.setattr(worker,'load_model',lambda *args:reconstructor)
    validated=[]
    validate=nksr_mesh.validate_mesh
    def capture_validation(vertices,triangles):
        validated.append(vertices.copy())
        return validate(vertices,triangles)
    monkeypatch.setattr(nksr_mesh,'validate_mesh',capture_validation)
    assert worker.main()==0
    assert worker.NKSR_NATIVE_VOXEL_SIZE/worker.DEFAULT_NKSR_TARGET_VOXEL_M==5.0
    call=calls[-1]
    np.testing.assert_array_equal(call['xyz'].value,points*scale)
    np.testing.assert_array_equal(call['sensor'].value,sensors*scale)
    assert call['detail_level'] is None
    if chunked:
        assert call['chunk_size']==25.0
        assert 'voxel_size' not in call
    else:
        assert call['voxel_size']==.02 and 'chunk_size' not in call
    if failures:
        np.testing.assert_array_equal(calls[0]['xyz'].value,points)
        np.testing.assert_array_equal(calls[0]['sensor'].value,sensors)
        assert calls[0]['voxel_size']==.02
    np.testing.assert_array_equal(validated[0],points)
    ply=PlyData.read(output)
    np.testing.assert_array_equal(np.column_stack([ply['vertex'][axis] for axis in 'xyz']),points)
    np.testing.assert_array_equal(ply['face']['vertex_indices'].tolist(),faces)
    metadata=read_json(tmp_path/'nksr_metadata.json')
    assert metadata['mesh_bbox']==[points.min(axis=0).tolist(),points.max(axis=0).tolist()]
    assert metadata['bounding_box_min']==points.min(axis=0).tolist()
    assert metadata['bounding_box_max']==points.max(axis=0).tolist()
    assert metadata['validation_status']=='PASS' and metadata['bbox_difference']==[0.,0.,0.]
    assert metadata['coordinate_scale']==scale and metadata['target_voxel_m']==.02
    assert metadata['detail_level'] is None
    events=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    diagnostic=[event for event in events if 'coordinate_scale' in event][-1]
    assert diagnostic['coordinate_scale']==scale and diagnostic['target_voxel_m']==.02
    assert diagnostic['chunk_size']==(5.0 if chunked else None)
    assert diagnostic['nksr_chunk_size']==(25.0 if chunked else None)
    assert diagnostic['voxel_size']==(None if chunked else .02)
    assert source.read_bytes()==original


def test_failure_classification():
    assert classify(RuntimeError('CUDA out of memory'),'EXTRACTING_MESH')=='CUDA_OOM'
    assert classify(RuntimeError('network timeout'),'DOWNLOADING_MODEL')=='CHECKPOINT_DOWNLOAD_FAILED'
    assert classify(RuntimeError('bad mesh'),'SAVING_MESH')=='MESH_INVALID'


@pytest.mark.skipif(os.environ.get('RUN_NKSR_INTEGRATION')!='1',reason='Opt-in real NKSR model/inference test')
@pytest.mark.parametrize('mode',['full','low_ram'])
def test_real_nksr_integration(tmp_path,mode):
    root=Path(__file__).resolve().parents[1]
    python=jobs.interpreter(root)
    configured=os.environ.get('NKSR_TEST_INPUT')
    saved=root/'.state/nksr-validation-run.txt'
    if not configured and saved.exists(): configured=str(Path(saved.read_text())/'input/nksr_input.npz')
    if mode=='low_ram':
        rng=np.random.default_rng(42);directions=rng.normal(size=(2048,3)).astype(np.float32)
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        centers=[np.array([2.5,2.5,2.5]),np.array([7.5,2.5,2.5])]
        points=np.concatenate([directions+center for center in centers]).astype(np.float32)
        sensors=np.concatenate([directions*3+center for center in centers]).astype(np.float32)
    elif configured:
        with np.load(configured) as data:
            points=data['points'][:20000];sensors=data['sensor_origins'][:20000]
    else:
        rng=np.random.default_rng(42);points=rng.normal(size=(2048,3)).astype(np.float32)
        points/=np.linalg.norm(points,axis=1,keepdims=True);sensors=points*3
    np.savez(tmp_path/'prepared sample.npz',points=points,sensor_origins=sensors)
    result=subprocess.run([str(python),str(jobs.worker_path()),'--input',str(tmp_path/'prepared sample.npz'),
                           '--output',str(tmp_path/'mesh.ply'),'--mode',mode],
                          env=jobs.worker_environment(),capture_output=True,text=True,timeout=600)
    assert result.returncode==0,result.stdout+result.stderr
    assert inspect_mesh(tmp_path/'mesh.ply')['face_count']>0
    metadata=read_json(tmp_path/'nksr_metadata.json')
    assert metadata['checkpoint_loaded'] and metadata['vertex_count']>0 and metadata['face_count']>0
    if mode=='low_ram':
        assert metadata['tile_count']==2 and metadata['completed_tiles']==2
        assert metadata['actual_mode']=='low_ram' and metadata['extraction_max_points']==100000
        manifest=read_json(tmp_path/'tiles.json')
        assert all(tile['state']=='COMPLETED' for tile in manifest['tiles'])



def test_retry_preserves_previous_output_as_one_attempt(prepared,monkeypatch):
    service,sid,run,_=prepared
    triangle(run/'output');atomic_json(run/'mesh_job.json',{'state':'COMPLETED'});atomic_json(run/'nksr_progress.json',{'stage':'COMPLETED'})
    async def launch(*args): pass
    monkeypatch.setattr(service.pm,'start',launch)
    asyncio.run(jobs.reconstruct(service,sid,run.name,MeshRequest().model_dump()))
    attempts=list((run/'attempts').iterdir())
    assert len(attempts)==1
    assert inspect_mesh(attempts[0]/'output/mesh.ply')['face_count']==1
    assert read_json(attempts[0]/'mesh_job.json')['state']=='COMPLETED'


def test_api_reconstruction_validation_and_routes(root,monkeypatch):
    from fastapi.testclient import TestClient
    from factory_mapping.api import make_app
    calls=[]
    async def run(service,sid,rid,settings): calls.append(settings);return {'state':'RUNNING'}
    monkeypatch.setattr(jobs,'reconstruct',run)
    with TestClient(make_app(root,True)) as client:
        url='/api/sessions/session/reconstruction/run_0123456789ab/mesh'
        assert client.post(url,json={'mode':'chunked','chunk_size':5}).status_code==202
        assert calls[-1]['preparation_voxel_size_m']==.01 and calls[-1]['detail_level']==.5
        assert client.post(url,json={'mode':'low_ram','tile_size':2.5}).status_code==202
        assert calls[-1]['mode']=='low_ram' and calls[-1]['tile_size']==2.5
        for obj in ({'mode':'poisson'},{'chunk_size':-1},{'normal_knn':0},{'normal_drop_angle_deg':91},
                    {'mode':'low_ram','tile_size':0},{'tile_size':-1},{'tile_size':'NaN'}):
            assert client.post(url,json=obj).status_code==422


# --------------------------------------------------------------------------- #
# End-to-end worker output for Full and Chunked
# --------------------------------------------------------------------------- #

def world_surface():
    """Two unit cubes 25 m apart, in GLIM world metres."""
    local=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=np.float32)
    triangles=np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],
                        [1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]],dtype=np.int32)
    vertices,faces=[],[]
    for origin in ([0.,0.,0.],[25.,0.,0.]):
        base=len(vertices);vertices.extend((np.asarray(origin,dtype=np.float32)+local).tolist())
        faces.extend((triangles+base).tolist())
    return np.asarray(vertices,dtype=np.float32),np.asarray(faces,dtype=np.int32)


def tile_triangle_multiset(directory,manifest):
    from plyfile import PlyData
    triangles=[]
    for chunk in manifest['chunks']:
        mesh=PlyData.read(str(Path(directory)/chunk['file']),known_list_len={'face':{'vertex_indices':3}})
        local=np.round(np.column_stack([mesh['vertex'][axis] for axis in 'xyz']).astype(np.float64),6)
        indexed=np.asarray(mesh['face']['vertex_indices'])
        if indexed.dtype.kind=='O': indexed=np.stack(indexed)
        triangles.extend(sorted(tuple(local[index].tolist()) for index in face) for face in indexed)
    return sorted(triangles)


def run_worker(tmp_path,monkeypatch,mode,output_mode,mesh,chunk_size='10'):
    """Run the real worker main() against a stubbed runtime; returns (output, reconstruct calls)."""
    from factory_mapping import nksr_worker as worker
    source=tmp_path/'input.npz';np.savez(source,points=np.ones((70,3)),sensor_origins=np.ones((70,3))*5)
    output=tmp_path/'output/mesh.ply'
    torch,nksr,reconstructor,reconstruct_calls,_=fake_runtime(0,mesh=mesh)
    monkeypatch.setattr(worker.sys,'argv',['worker','--input',str(source),'--output',str(output),
        '--metadata',str(output.parent/'nksr_metadata.json'),'--mode',mode,
        '--mesh-output-mode',output_mode,'--chunk-size',chunk_size])
    monkeypatch.setattr(worker.signal,'signal',lambda *args:None)
    monkeypatch.setattr(worker,'runtime',lambda device:(torch,nksr,torch.device('cuda'),{}))
    monkeypatch.setattr(worker,'load_model',lambda *args:reconstructor)
    assert worker.main()==0
    return output,reconstruct_calls


@pytest.mark.parametrize('mode,scale',[('full',1.0),('chunked',5.0)])
def test_chunks_export_partitions_the_final_worker_mesh(mode,scale,tmp_path,monkeypatch):
    vertices,faces=world_surface()
    output,calls=run_worker(tmp_path,monkeypatch,mode,'chunks',
                            SimpleNamespace(v=np.asarray(vertices*scale,dtype=np.float32),f=faces))
    assert len(calls)==1  # the final surface is reconstructed and extracted once
    metadata=jobs.validate_completed(output.parent,0)
    assert metadata['actual_mode']==mode and metadata['mesh_output_mode']=='chunks'
    assert metadata['effective_chunk_size_m']==10.0 and metadata['chunk_size_source']=='user'
    assert metadata['chunk_count']==2 and metadata['chunk_faces_total']==len(faces)
    assert not output.exists() and metadata['export_strategy']=='spatial_split_of_final_mesh'
    manifest=read_json(output.parent/'mesh_chunks/chunks.json')
    assert manifest['total_faces']==len(faces) and manifest['source_faces']==len(faces)
    # Concatenated tiles reproduce the original triangles exactly, in world metres.
    assert tile_triangle_multiset(output.parent/'mesh_chunks',manifest)== \
        tile_triangle_multiset_for(vertices,faces)


def tile_triangle_multiset_for(vertices,faces):
    rounded=np.round(np.asarray(vertices,dtype=np.float64),6)
    return sorted(sorted(tuple(rounded[index].tolist()) for index in face) for face in faces)


@pytest.mark.parametrize('mode,scale',[('full',1.0),('chunked',5.0)])
def test_both_export_writes_merged_mesh_and_tiles(mode,scale,tmp_path,monkeypatch):
    vertices,faces=world_surface()
    output,calls=run_worker(tmp_path,monkeypatch,mode,'both',
                            SimpleNamespace(v=np.asarray(vertices*scale,dtype=np.float32),f=faces))
    assert len(calls)==1  # Both never reconstructs twice
    metadata=jobs.validate_completed(output.parent,0)
    assert metadata['mesh_output_mode']=='both' and metadata['vertex_count']==len(vertices)
    assert metadata['face_count']==len(faces) and metadata['chunk_count']==2
    assert output.is_file() and metadata['output_bytes']>metadata['mesh_file_size']
    merged=inspect_mesh(output)
    manifest=read_json(output.parent/'mesh_chunks/chunks.json')
    assert merged['face_count']==manifest['total_faces']==len(faces)


def test_chunked_both_uses_the_fused_field_once_not_native_fields(tmp_path,monkeypatch):
    """Chunked separate output must not fall back to independent native chunk fields."""
    from factory_mapping import nksr_worker as worker
    vertices,faces=world_surface()
    fields_seen=[]
    original=worker.partition_mesh
    def counted(vertices_in,faces_in,size,directory,**kwargs):
        fields_seen.append(kwargs['reconstruction_mode'])
        return original(vertices_in,faces_in,size,directory,**kwargs)
    monkeypatch.setattr(worker,'partition_mesh',counted)
    output,calls=run_worker(tmp_path,monkeypatch,'chunked','both',
                            SimpleNamespace(v=np.asarray(vertices*5,dtype=np.float32),f=faces))
    assert len(calls)==1 and fields_seen==['chunked']
    manifest=read_json(output.parent/'mesh_chunks/chunks.json')
    assert manifest['total_faces']==len(faces) and manifest['export_strategy']=='spatial_split_of_final_mesh'
    assert jobs.validate_completed(output.parent,0)['chunk_count']==2
