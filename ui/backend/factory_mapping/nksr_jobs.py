"""Subprocess-only NKSR orchestration. The web/ROS process never imports torch."""
import asyncio
import os
from pathlib import Path
import re
import uuid
import time
from .storage import atomic_json, read_json, now
from .nksr_mesh import inspect_mesh


def interpreter(root):
    configured=os.environ.get('NKSR_PYTHON')
    saved=root/'.state/nksr_python.txt'
    if not configured and saved.is_file(): configured=saved.read_text().strip()
    return Path(configured or Path.home()/'.cache/factory-mapping/nksr-env/bin/python').expanduser().absolute()


def worker_environment():
    env=os.environ.copy()
    # Keep ROS's library/Python paths out of the independent environment.
    for name in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH'): env.pop(name,None)
    env['OMP_NUM_THREADS']='4'
    return env


def health(service):
    if service.config['system']['deployment_mode'] == 'record_only':
        return dict(status='RECORD_ONLY', message='Export recordings to the workstation for surfacing')
    python=interpreter(service.root)
    if not python.is_file(): return dict(status='NKSR_NOT_INSTALLED',message='Run scripts/setup_nksr.sh',python=str(python))
    result=read_json(service.root/'.state/nksr_health.json',{})
    if result.get('python') != str(python) or not result or time.time()-result.get('checked_at',0)>86400:
        result=dict(status='UNVERIFIED',message='Run Check NKSR to load the model and test real inference',python=str(python))
    if service.pm.active('nksr_check'): result={**result,'status':'CHECKING'}
    return result


def worker_path(): return Path(__file__).resolve().parents[3]/'tools/nksr_worker.py'


async def check(service, device='auto'):
    service.require_processing()
    python=interpreter(service.root)
    if not python.is_file(): raise ValueError('NKSR_NOT_INSTALLED: run scripts/setup_nksr.sh')
    if service.capture.busy or service.active or any(service.pm.active(k) for k in ('nksr','nksr_check','reconstruction','offline','glim','tool')):
        raise ValueError('Wait for active capture or processing before checking NKSR')
    target=service.root/'.state/nksr_health.json'
    atomic_json(target,dict(status='CHECKING',python=str(python),checked_at=time.time()))
    async def done(item):
        if item['state'] != 'completed' and read_json(target,{}).get('status') in ('READY','CHECKING'):
            atomic_json(target,dict(status='FAILED',python=str(python),message='NKSR check failed; inspect nksr_check.log'))
    return await service.pm.start('nksr_check',[str(python),str(worker_path()),'--check','--device',device,
                                  '--health-output',str(target)],service.root/'.state/nksr_check.log',worker_environment(),done)


def get_run(service,sid,rid):
    if not re.fullmatch(r'run_[a-f0-9]{12}',rid): raise ValueError('Invalid reconstruction run')
    session=service.sessions.get(sid);run=session/'reconstruction'/rid
    if not run.is_dir() or any(p.is_symlink() for p in (run,run.parent)):
        raise ValueError('Reconstruction run not found')
    return run


def mesh_state(run):
    result=read_json(run/'mesh_job.json',{'state':'NOT_RECONSTRUCTED'})
    progress=read_json(run/'nksr_progress.json',{})
    if result['state']=='RUNNING':
        result['stage']=progress.get('stage','LOADING_MODEL')
        if result['stage']=='COMPLETED': result['stage']='VALIDATING_MESH'
    result['progress']=progress
    return result


def validate_completed(output, returncode):
    if returncode != 0: raise ValueError('Worker exited unsuccessfully')
    stats=inspect_mesh(output/'mesh.ply')
    metadata=read_json(output/'nksr_metadata.json',{})
    if stats['vertex_count'] != metadata.get('vertex_count') or stats['face_count'] != metadata.get('face_count'):
        raise ValueError('Mesh counts do not match worker metadata')
    return metadata


async def reconstruct(service,sid,rid,settings):
    service.require_processing()
    from .reconstruction_jobs import preparation_state
    run=get_run(service,sid,rid)
    job=read_json(run/'job.json',{})
    if preparation_state(run,job)!='PREPARED': raise ValueError('Prepare point input first')
    if abs(settings['preparation_voxel_size_m']-job.get('voxel_size_m',.01))>1e-12:
        raise ValueError('Prepared input is stale; prepare again with the selected voxel size')
    if service.mock: raise ValueError('Real NKSR requires real prepared input')
    if service.capture.busy or service.active or any(service.pm.active(k) for k in
            ('nksr','nksr_check','reconstruction','offline','tool','glim','recording')):
        raise ValueError('Finish capture and active processing first')
    python=interpreter(service.root)
    if not python.is_file(): raise ValueError('NKSR_NOT_INSTALLED: run scripts/setup_nksr.sh')
    input_path=run/'input/nksr_input.npz'
    if input_path.is_symlink() or input_path.parent.is_symlink(): raise ValueError('Invalid input path')
    # Preserve successful and failed prior attempts. Never overwrite prepared data or a mesh.
    archive=run/'attempts'/('previous_'+uuid.uuid4().hex[:12])
    for name in ('output','mesh_job.json','nksr_progress.json'):
        path=run/name
        if path.exists():
            archive.mkdir(parents=True,exist_ok=True)
            path.rename(archive/name)
    output=run/'output'
    data=dict(state='RUNNING',started_at=now(),settings=settings,python=str(python))
    atomic_json(run/'mesh_job.json',data)
    args=[str(python),str(worker_path()),'--input',str(input_path),'--output',str(output/'mesh.ply'),
          '--metadata',str(output/'nksr_metadata.json'),'--progress',str(run/'nksr_progress.json')]
    for key in ('device','mode','detail_level','chunk_size','normal_knn','normal_drop_angle_deg','mise_iter','overlap_ratio'):
        if settings.get(key) is not None: args.extend(['--'+key.replace('_','-'),str(settings[key])])
    async def done(item):
        progress=read_json(run/'nksr_progress.json',{})
        data.update(ended_at=now(),returncode=item['returncode'])
        if item['state']=='cancelled': data.update(state='CANCELLED',error_type='CANCELLED',message='Reconstruction cancelled')
        elif item['state']=='completed':
            try:
                data.update(metadata=await asyncio.to_thread(validate_completed,output,item['returncode']),state='COMPLETED')
            except Exception as error: data.update(state='FAILED',error_type='MESH_INVALID',message=str(error))
        else:
            data.update(state='FAILED',error_type=progress.get('error_type','NKSR_RECONSTRUCTION_FAILED'),
                        message=progress.get('message','NKSR worker failed; inspect job.log'))
        atomic_json(run/'mesh_job.json',data)
    try: await service.pm.start('nksr',args,run/'job.log',worker_environment(),done)
    except Exception as error:
        data.update(state='FAILED',error_type='NKSR_NOT_INSTALLED',message=str(error));atomic_json(run/'mesh_job.json',data);raise
    return data


async def cancel(service,sid,rid):
    run=get_run(service,sid,rid)
    item=service.pm.items.get('nksr',{})
    if item and Path(item['log']).parent!=run: raise ValueError('Another reconstruction owns the worker')
    await service.pm.stop('nksr',20,cancel=True)
    return mesh_state(run)
