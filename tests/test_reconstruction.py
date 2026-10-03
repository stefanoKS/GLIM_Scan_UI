import asyncio
import builtins
import json
from pathlib import Path
import numpy as np
import pytest
from factory_mapping.reconstruction import (DEFAULT_VOXEL_SIZE_M, parser, sample_records,
    save_inputs, transform_points, voxel_indices, prepare)
from factory_mapping.api import ReconstructionRequest


def records():
    points = np.array([[-.001,0,0], [.001,0,0], [.009,0,0], [.011,0,0], [.019,0,0], [.03,0,0]])
    n = len(points)
    return dict(points=points, sensor_origins=np.arange(n*3).reshape(n,3),
                intensity=np.arange(n)+100, timestamps=np.arange(n)+1000.,
                tag=np.arange(n)+10, line=np.arange(n)+20)


def test_default_units_and_density():
    args = parser().parse_args(['--bag','bag','--trajectory','traj','--output-dir','out'])
    assert args.voxel_size == DEFAULT_VOXEL_SIZE_M == .01
    assert args.voxel_size * 100 == 1.0
    assert ReconstructionRequest(trajectory='x').voxel_size_m == .01
    data = records()
    assert len(sample_records(data)['points']) == 4
    assert len(sample_records(data,.02)['points']) == 3
    np.testing.assert_array_equal(voxel_indices(data['points']), [0,1,3,5])


@pytest.mark.parametrize('size', [0,.01,.02])
def test_pairing_order_and_determinism(size):
    data = records()
    a, b = sample_records(data,size), sample_records(data,size)
    indices = (a['intensity']-100).astype(int)
    assert np.all(np.diff(indices)>0)
    for name in data:
        np.testing.assert_array_equal(a[name],data[name][indices])
        np.testing.assert_array_equal(a[name],b[name])
        if size == 0: np.testing.assert_array_equal(a[name],data[name])


@pytest.mark.parametrize('size', [-1,float('nan'),float('inf')])
def test_invalid_size(size):
    with pytest.raises(ValueError): sample_records(records(),size)
    with pytest.raises(ValueError): ReconstructionRequest(trajectory='x',voxel_size_m=size)


def test_world_space_and_trajectory_range(tmp_path, monkeypatch):
    trajectory = np.array([[10,0,0,0,0,0,0,1],[11,1,0,0,0,0,0,1]])
    # Identical raw coordinates become different world voxels as the sensor moves.
    xyz = np.zeros((5,3)); times = np.array([9,10,10.5,11,12])
    world, origins, valid = transform_points(xyz,times,trajectory)
    np.testing.assert_array_equal(valid,[False,True,True,True,False])
    np.testing.assert_allclose(world,origins)
    assert len(voxel_indices(world)) == 3
    assert len(voxel_indices(xyz)) == 1


def test_outputs_metadata_and_export_isolation(tmp_path, monkeypatch):
    export = tmp_path/'exports/run_001.ply';export.parent.mkdir();export.write_bytes(b'untouched GLIM')
    original_import = builtins.__import__
    def missing_nksr(name,*args,**kwargs):
        if name in ('nksr','open3d'): raise ImportError(name)
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',missing_nksr)
    data=records();out=tmp_path/'reconstruction/run_test'
    meta=save_inputs(data,out)
    assert meta['points_before_voxel']==6 and meta['points_after_voxel']==4
    assert meta['voxel_reduction_ratio']==4/6
    assert json.loads((out/'validation/comparison.json').read_text())==meta
    with np.load(out/'input/nksr_input.npz') as npz:
        selected=sample_records(data)
        for name in data: np.testing.assert_allclose(npz[name],selected[name])
        with (out/'validation/reconstructed_from_bag.ply').open('rb') as f:
            while f.readline()!=b'end_header\n': pass
            vertices=np.frombuffer(f.read(),dtype='<f4').reshape(-1,4)
        np.testing.assert_array_equal(vertices[:,:3],npz['points'])
        np.testing.assert_array_equal(vertices[:,3],npz['intensity'])
    assert not (out/'validation/reconstructed_from_bag_full.ply').exists()
    save_inputs(data,tmp_path/'debug',0,True)
    assert (tmp_path/'debug/validation/reconstructed_from_bag_full.ply').exists()
    assert export.read_bytes()==b'untouched GLIM'
    from factory_mapping.commands import export as export_command
    assert export_command('dump',export,'config') == ['ros2','run','glim_ros','offline_viewer','dump','--export_path',str(export),'--config_path','config']


def test_job_passes_parameter_and_preserves_exports(root,monkeypatch):
    from factory_mapping.service import Service
    from factory_mapping.reconstruction_jobs import start, view
    service=Service(root,mock=True)
    m=service.sessions.create('test','',service.config);session=service.sessions.get(m['id'])
    trajectory=session/'processing/run_001/glim_dump/traj_lidar.txt';trajectory.parent.mkdir(parents=True);trajectory.write_text('test')
    (session/'raw_bag').mkdir();(session/'raw_bag/metadata.yaml').write_text('test')
    export=session/'exports/run_001.ply';export.write_bytes(b'GLIM')
    calls=[]
    async def launch(*args): calls.append(args)
    monkeypatch.setattr(service.pm,'start',launch);service.mock=False
    job=asyncio.run(start(service,m['id'],str(trajectory.relative_to(session)),.025))
    argv=calls[0][1]
    assert argv[argv.index('--voxel-size')+1]=='0.025'
    assert export.read_bytes()==b'GLIM'
    assert len(view(service,m['id'])['trajectories'])==1
    with pytest.raises(ValueError): asyncio.run(start(service,m['id'],'../traj_lidar.txt',.01))


def test_ui_units_and_advanced_scope():
    root=Path(__file__).resolve().parents[1]
    html=(root/'ui/frontend/index.html').read_text()
    section=html.split('id="surface-reconstruction"')[1].split('</section>')[0]
    advanced=section.split('<details>')[1].split('</details>')[0]
    assert 'id="voxel-size-cm"' in advanced and 'value="1.0"' in advanced
    assert 'min="0.2"' in advanced and 'max="20"' in advanced
    assert 'voxel_size_m:cm/100.0' in (root/'ui/frontend/app.js').read_text()


def test_api_preparation_default_and_overrides(root, monkeypatch):
    from fastapi.testclient import TestClient
    from factory_mapping.api import make_app
    from factory_mapping import reconstruction_jobs
    calls=[]
    async def start(service,sid,trajectory,size,save_full):
        calls.append((sid,trajectory,size,save_full))
        return {'state':'running'}
    monkeypatch.setattr(reconstruction_jobs,'start',start)
    with TestClient(make_app(root,True)) as client:
        for params, expected in [({},.01),({'voxel_size_m':.025},.025),({'voxel_size_m':0},0)]:
            response=client.post('/api/sessions/test/reconstruction',json={'trajectory':'traj_lidar.txt',**params})
            assert response.status_code==202
            assert calls[-1][2]==expected
        assert client.post('/api/sessions/test/reconstruction',json={'trajectory':'x','voxel_size_m':-1}).status_code==422


def test_glim_export_without_nksr(root, monkeypatch):
    from factory_mapping.service import Service
    from factory_mapping.storage import atomic_json
    service=Service(root,True)
    session=service.sessions.create('export','',service.config)
    path=service.sessions.get(session['id'])/'processing/run_001'
    path.mkdir(parents=True);atomic_json(path/'job.json',{'state':'completed'})
    monkeypatch.setenv('DISPLAY',':0')
    service.mock=False
    original_import=builtins.__import__
    def without_nksr(name,*args,**kwargs):
        if name=='nksr': raise ImportError('NKSR unavailable')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',without_nksr)
    async def launch(key,args,log,**kwargs):
        assert key=='export' and 'offline_viewer' in args
        assert not any('voxel' in arg for arg in args)
        target=Path(args[args.index('--export_path')+1]);target.write_bytes(b'GLIM optimized')
        await kwargs['done']({'returncode':0})
        return target
    monkeypatch.setattr(service,'start_process',launch)
    output=asyncio.run(service.export(session['id'],'run_001'))
    assert output.parent.name=='exports'
    assert output.read_bytes()==b'GLIM optimized'
