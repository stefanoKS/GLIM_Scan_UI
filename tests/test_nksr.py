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
from factory_mapping.nksr_mesh import write_mesh, inspect_mesh
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
def test_real_nksr_integration(tmp_path):
    root=Path(__file__).resolve().parents[1]
    python=jobs.interpreter(root)
    configured=os.environ.get('NKSR_TEST_INPUT')
    saved=root/'.state/nksr-validation-run.txt'
    if not configured and saved.exists(): configured=str(Path(saved.read_text())/'input/nksr_input.npz')
    if configured:
        with np.load(configured) as data:
            points=data['points'][:20000];sensors=data['sensor_origins'][:20000]
    else:
        rng=np.random.default_rng(42);points=rng.normal(size=(2048,3)).astype(np.float32)
        points/=np.linalg.norm(points,axis=1,keepdims=True);sensors=points*3
    np.savez(tmp_path/'prepared sample.npz',points=points,sensor_origins=sensors)
    result=subprocess.run([str(python),str(jobs.worker_path()),'--input',str(tmp_path/'prepared sample.npz'),
                           '--output',str(tmp_path/'mesh.ply'),'--mode','full'],
                          env=jobs.worker_environment(),capture_output=True,text=True,timeout=600)
    assert result.returncode==0,result.stdout+result.stderr
    assert inspect_mesh(tmp_path/'mesh.ply')['face_count']>0
    metadata=read_json(tmp_path/'nksr_metadata.json')
    assert metadata['checkpoint_loaded'] and metadata['vertex_count']>0 and metadata['face_count']>0



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
        for obj in ({'mode':'poisson'},{'chunk_size':-1},{'normal_knn':0},{'normal_drop_angle_deg':91}):
            assert client.post(url,json=obj).status_code==422
