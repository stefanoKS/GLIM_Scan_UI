import asyncio, contextlib, fcntl, json, os, shutil, tempfile, time, zipfile
from pathlib import Path
from typing import Literal
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field, ConfigDict
from .config import ROOT, validate_sensor
from .service import Service
from .glim_tools import capabilities
from .storage import atomic_json, read_json
from .preview import encode
from .quality import quality, ply_stats, ply_to_pcd

class CaptureAction(BaseModel):
    model_config=ConfigDict(extra='forbid')
    action: Literal['start_scan','stop_scan','start_camera_recording','stop_camera_recording']

class CaptureSettings(BaseModel):
    model_config=ConfigDict(extra='forbid')
    live_glim: bool=Field(strict=True)
    auto_process: bool=Field(strict=True)
    mapping_preset: Literal['auto','jetson_cpu','pc_dense','jetson_gpu','offline_quality'] | None=None

class SessionEdit(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name: str=Field(min_length=1,max_length=100)
    notes: str=Field(default='',max_length=4000)

class Create(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name: str=Field(default='',max_length=100)
    notes: str=Field(default='',max_length=4000)
class Action(BaseModel):
    model_config=ConfigDict(extra='forbid')
    action: str
    session: str | None=None
    preset: str='jetson_cpu'
    run: str | None=None
class ToolSource(BaseModel):
    model_config=ConfigDict(extra='forbid')
    session: str
    run: str
class ToolRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    session: str
    run: str
    tool: str
    additional: list[ToolSource]=Field(default_factory=list,max_length=10)
class CameraEnabled(BaseModel):
    model_config=ConfigDict(extra="forbid")
    enabled: bool=Field(strict=True)

class IntrinsicsImport(BaseModel):
    model_config=ConfigDict(extra='forbid')
    yaml_text: str=Field(min_length=1,max_length=65536)
class CalibrationAction(BaseModel):
    model_config=ConfigDict(extra='forbid')
    action: str
    notes: str=Field(default='',max_length=8000)
class Network(BaseModel):
    model_config=ConfigDict(extra='forbid')
    lidar_ip: str
    interface: str
    points_topic: str
    imu_topic: str
    publish_freq: float
    ros_domain_id: int

def make_app(root=ROOT,mock=None):
    mock=bool(os.environ.get('FACTORY_MAPPING_MOCK')=='1') if mock is None else mock
    state=root/'.state'; state.mkdir(parents=True,exist_ok=True)
    @asynccontextmanager
    async def lifespan(app):
        guard=(state/'backend.lock').open('w')
        try: fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: guard.close(); raise RuntimeError('Another mapping backend is already running')
        service=Service(root,mock); app.state.service=service
        await service.start_preview_sensors()
        bg=asyncio.create_task(service.background())
        try: yield
        finally:
            bg.cancel()
            with contextlib.suppress(asyncio.CancelledError): await bg
            await service.close(); guard.close()
    app=FastAPI(title='Factory Mapping',lifespan=lifespan)
    @app.middleware('http')
    async def same_origin_guard(request,call_next):
        origin=request.headers.get('origin')
        if request.url.path.startswith('/api/') and origin and origin!=str(request.base_url).rstrip('/'):
            return JSONResponse({'detail':'Cross-origin control is disabled'},status_code=403)
        return await call_next(request)
    @app.exception_handler(ValueError)
    async def bad_request(request,e): return JSONResponse({'detail':str(e)},status_code=409)
    @app.get('/api/status')
    async def status(): return await asyncio.to_thread(app.state.service.status)
    @app.get('/api/capture')
    async def capture_status(): return app.state.service.capture.view()

    @app.post('/api/capture/action',status_code=202)
    async def capture_action(body:CaptureAction):
        return await getattr(app.state.service.capture,body.action)()

    @app.put('/api/capture/settings')
    async def capture_settings(body:CaptureSettings):
        s=app.state.service
        async with s.lock:
            if s.capture.busy or s.active: raise ValueError('Finish capture before changing its settings')
            if body.mapping_preset and body.mapping_preset != 'auto': s.check_preset(body.mapping_preset)
            settings={**s.capture.capabilities()['settings'], **body.model_dump(exclude_none=True)}
            atomic_json(state/'capture_settings.json',settings)
            return s.capture.capabilities()

    @app.patch('/api/sessions/{sid}')
    async def edit_session(sid:str,body:SessionEdit):
        s=app.state.service
        async with s.lock:
            p=s.sessions.get(sid)
            if s.active==p or (s.capture.busy and s.capture.data.get('session')==sid): raise ValueError('Finish capture and processing before renaming the scan')
            if not body.name.strip(): raise ValueError('Scan name cannot be blank')
            return s.sessions.update(p,name=body.name.strip(),notes=body.notes)

    @app.get('/api/preview/orientation')
    async def view_orientation():
        return read_json(state/'view_orientation.json',{'quaternion':[0,0,0,1]})

    @app.post('/api/preview/orientation')
    async def orient_preview():
        from .orientation import MAX_IMU_AGE_SECONDS, leveling_quaternion
        s=app.state.service
        async with s.lock:
            if s.capture.busy or s.active: raise ValueError('Finish recording before orienting the view')
            s.calibrations.idle()
            if s.mock: raise ValueError('Orientation requires real IMU measurements')
            if not s.pm.active('driver'): await s.start_driver()
            sample=s.health().get('gravity',{})
            if not sample.get('stable') or not 0<=time.time()-sample.get('last',0)<=MAX_IMU_AGE_SECONDS:
                raise ValueError('Fresh, stable IMU data is unavailable; keep the LiDAR still with the sensor running and try again')
            q=leveling_quaternion(sample['acceleration'],s.config['sensor']['T_lidar_imu'][3:])
            result={'quaternion':q,'saved_at':time.time(),'scope':'live_preview_only'}
            atomic_json(state/'view_orientation.json',result)
            return result

    @app.delete('/api/preview/orientation')
    async def reset_orientation():
        result={'quaternion':[0,0,0,1],'scope':'live_preview_only'}
        async with app.state.service.lock: atomic_json(state/'view_orientation.json',result)
        return result

    @app.post('/api/calibration/prepare')
    async def prepare_intrinsic_camera():
        s=app.state.service
        async with s.lock:
            if s.capture.busy: raise ValueError('Finish capture first')
            s.calibrations.idle()
            await s.prepare_camera()
            return {'ready':True}

    @app.post('/api/calibrations/{cid}/wizard')
    async def calibration_wizard(cid:str,body:CalibrationAction):
        s=app.state.service
        async with s.lock:
            if s.capture.busy: raise ValueError('Finish capture first')
            cal=s.calibrations
            if body.action=='capture':
                cal.idle()
                if cal.detail(cid)['state'] not in ('CREATED','CAPTURED'): raise ValueError('Create a new alignment for additional captures')
                await s.prepare_recording()
                return await cal.capture_start(cid)
            if body.action=='stop': return await cal.capture_stop(cid)
            if body.action=='advance':
                current=cal.detail(cid)['state']
                stage={'CAPTURED':'preprocess','PREPROCESSED':'initial_guess_manual','INITIALIZED':'calibrate'}.get(current)
                if stage: return await cal.run(cid,stage)
                if current=='CALIBRATED':
                    cal.idle()
                    await s.stop_camera();await s.stop_driver()
                    return await asyncio.to_thread(cal.import_result,cid)
                if current=='IMPORTED': return await asyncio.to_thread(cal.validate,cid,body.notes)
                raise ValueError('Complete the current calibration step before continuing')
            raise ValueError('Unknown calibration wizard action')

    @app.post('/api/camera/enabled')
    async def camera_enabled(body:CameraEnabled):
        s=app.state.service
        async with s.lock:
            if s.capture.busy: raise ValueError('Finish capture before changing RGB recording')
            s.calibrations.idle()
            if body.enabled and 'camera' not in s.config: raise ValueError('Camera configuration is missing')
            if not body.enabled: await s.stop_camera()
            # Machine preference is separate from the portable default-disabled config.
            atomic_json(root/'.state/camera_enabled.json',{'enabled':body.enabled})
            s.config['system']['camera']['enabled']=body.enabled
            return {'enabled':body.enabled}
    @app.get('/api/camera/preview')
    async def camera_preview():
        s=app.state.service
        if not s.pm.active('camera'): raise HTTPException(404,'Camera preview is unavailable')
        p=Path(__file__).with_name('mock_camera.jpg') if s.mock else root/'.state/camera_preview.jpg'
        if not s.mock and (not s.pm.active('camera_preview') or not p.is_file() or time.time()-p.stat().st_mtime>max(3,2/s.config['camera']['preview_hz'])): raise HTTPException(404,'Camera preview is stale or unavailable; recording is independent')
        return Response(await asyncio.to_thread(p.read_bytes),media_type='image/jpeg',headers={'Cache-Control':'no-store'})
    @app.get('/api/sessions')
    async def sessions(): return await asyncio.to_thread(app.state.service.sessions.list)
    def project_transfer_ready(service):
        if service.capture.busy or service.active or any(service.pm.active(key) for key in ('recording','glim','offline','export','tool','calibration_record','calibration_tool')):
            raise ValueError('Stop recording and processing before transferring a project')
    @app.get('/api/sessions/{sid}/project')
    async def export_project(sid:str):
        s=app.state.service
        async with s.lock:
            project_transfer_ready(s)
            session=s.sessions.get(sid)
            from .storage import size
            if await asyncio.to_thread(size,session)>shutil.disk_usage(state).free-s.config['system']['storage']['minimum_free_gb']*1e9: raise ValueError('Insufficient free space to stage the project export')
            descriptor,path=tempfile.mkstemp(prefix='project-export-',suffix='.zip',dir=state)
            os.close(descriptor)
            try:await asyncio.to_thread(s.sessions.export_archive,sid,path)
            except Exception:
                Path(path).unlink(missing_ok=True)
                raise
        return FileResponse(path,filename=sid+'.fmproject.zip',media_type='application/zip',background=BackgroundTask(Path(path).unlink,missing_ok=True))
    @app.post('/api/projects/import')
    async def import_project(request:Request):
        if request.headers.get('content-type','').split(';')[0].strip()!='application/zip':raise ValueError('Select a project ZIP archive')
        project_transfer_ready(app.state.service)
        descriptor,path=tempfile.mkstemp(prefix='project-import-',suffix='.zip',dir=state)
        try:
            with os.fdopen(descriptor,'wb') as output:
                async for chunk in request.stream():
                    project_transfer_ready(app.state.service)
                    if len(chunk)>shutil.disk_usage(state).free-1024**3:raise ValueError('Insufficient disk space for project upload')
                    await asyncio.to_thread(output.write,chunk)
            s=app.state.service
            async with s.lock:
                project_transfer_ready(s)
                return await asyncio.to_thread(s.sessions.import_archive,path)
        except (zipfile.BadZipFile,zipfile.LargeZipFile) as error:
            raise ValueError('Invalid project ZIP archive') from error
        finally:Path(path).unlink(missing_ok=True)
    @app.post('/api/sessions')
    async def create(body:Create):
        s=app.state.service
        async with s.lock: return s.sessions.create(body.name,body.notes,s.config)
    @app.post('/api/action')
    async def action(body:Action):
        s=app.state.service
        if s.capture.busy:
            if body.action in ('session_stop','record_stop'):
                return await (s.capture.stop_scan() if s.capture.data['mode']=='scan' else s.capture.stop_camera_recording())
            if body.action not in ('diagnose','glim_stop','cancel'): raise ValueError('Use Stop Scan / Stop Camera before advanced controls')
        async with s.lock:
            a=body.action
            if a=='driver_start': return await s.start_driver()
            if a=='driver_stop': await s.stop_driver()
            elif a=='camera_start': return await s.start_camera()
            elif a=='camera_stop': await s.stop_camera()
            elif a=='record_start': return await s.start_recording(body.session or '')
            elif a=='record_stop': await s.stop_recording()
            elif a=='glim_start': return await s.start_glim(body.session or '',body.preset)
            elif a=='glim_viewer_start': return await s.start_glim(body.session or '',body.preset,viewer=True)
            elif a=='glim_stop': await s.stop_glim()
            elif a in ('session_start','session_record_start'):
                s.sessions.get(body.session or '')
                if a=='session_start': s.check_preset(body.preset)
                elif s.pm.active('glim'): raise ValueError('Stop live GLIM before starting a record-only session')
                await s.start_recording(body.session or '')
                if a=='session_start':
                    try: await s.start_glim(body.session or '',body.preset)
                    except Exception as error:
                        s.errors.append('Live GLIM failed to start; recording continues: '+str(error))
                        return {'ok':True,'warning':str(error),'recording_continues':True}
            elif a=='session_stop': await s.stop_session()
            elif a=='process': return await s.offline(body.session or '',body.preset)
            elif a=='validator_start': return await s.start_validator()
            elif a=='validator_stop': await s.pm.stop('validator',20)
            elif a=='tool_stop': await s.pm.stop('tool',20,cancel=True)
            elif a=='export_edit': return await s.export_edit(body.session or '',body.run or '')
            elif a=='cancel': await s.pm.stop('offline',180,cancel=True)
            elif a=='export': return await s.export(body.session or '',body.run or '')
            elif a=='delete_derived': s.delete_run(body.session or '',body.run or '')
            elif a=='diagnose':
                s.net={'state':'mock'} if s.mock else await __import__('factory_mapping.health',fromlist=['network']).network(s.config['sensor'])
                return s.status()
            elif a not in ('driver_stop',): raise ValueError('Unknown controlled action')
            return {'ok':True}
    @app.put('/api/network')
    async def update_network(body:Network):
        s=app.state.service
        async with s.lock:
            if any(s.pm.active(k) for k in s.pm.items) or s.active: raise ValueError('Stop all processes and the active session before changing network settings')
            sensor={**s.config['sensor'],**body.model_dump()}; validate_sensor(sensor)
            from .config import wired_connection
            sensor['interface_setting']=body.interface
            sensor['interface'],sensor['host_ip']=wired_connection(sensor)
            if s.config.get('camera'):
                from .camera_config import validate_camera
                validate_camera(s.config['camera'],sensor)
            if sensor['ros_domain_id']==s.config['system'].get('offline_ros_domain_id',230): raise ValueError('Acquisition and offline ROS domains must differ')
            import yaml
            p=root/'config/livox/mid360.yaml'; tmp=p.with_suffix('.tmp'); tmp.write_text(yaml.safe_dump({**{key:value for key,value in sensor.items() if key not in ('host_ip','interface_setting')},'interface':body.interface},sort_keys=False)); tmp.replace(p); s.config['sensor']=sensor
            s.event('network_config_updated'); return {'ok':True}
    @app.post('/api/camera/intrinsics')
    async def import_intrinsics(body:IntrinsicsImport):
        import yaml
        from .calibration_data import parse_intrinsics,replace_intrinsics
        s=app.state.service
        async with s.lock:
            if s.capture.busy or s.active or any(s.pm.active(k) for k in s.pm.items): raise ValueError('Stop active processes before replacing intrinsics')
            if not s.config.get('camera'):raise ValueError('Camera configuration is missing')
            try:obj=yaml.safe_load(body.yaml_text);parse_intrinsics(obj,s.config['camera'])
            except (yaml.YAMLError,TypeError,AttributeError) as e:raise ValueError('Invalid ROS intrinsic calibration YAML') from e
            replace_intrinsics(root,s.config['camera'],obj)
            return s.camera_calibration()

    @app.get('/api/camera/intrinsics/board')
    async def intrinsic_board():
        from .camera_intrinsics import printable_board
        image=await asyncio.to_thread(printable_board)
        return Response(image,media_type='image/png',headers={'Content-Disposition':'attachment; filename="charuco_24x16.png"'})

    @app.get('/api/camera/intrinsics/views')
    async def intrinsic_views():
        from .camera_intrinsics import MIN_VIEWS
        s=app.state.service
        return dict(views=len(s.intrinsic_samples),required=MIN_VIEWS,intrinsics=s.camera_calibration()['intrinsics'])

    @app.delete('/api/camera/intrinsics/views')
    async def clear_intrinsic_views():
        s=app.state.service
        async with s.lock:
            s.intrinsic_samples.clear()
            return {'views':0}

    @app.post('/api/camera/intrinsics/views')
    async def capture_intrinsic_view():
        import cv2
        from .camera_intrinsics import add_view
        s=app.state.service
        async with s.lock:
            if not s.pm.active('camera') or not s.camera_health()['healthy'] or not s.pm.active('camera_preview'):
                raise ValueError('Start the camera and wait for healthy image frames')
            path=root/'.state/camera_calibration_frame.jpg'
            if not path.is_file() or time.time()-path.stat().st_mtime>2 or path.stat().st_size>16_000_000:
                raise ValueError('Full-resolution camera frame is unavailable or stale')
            frame_id=path.stat().st_mtime_ns
            image=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise ValueError('Could not decode camera frame')
            if (image.shape[1],image.shape[0]) != (s.config['camera']['width'],s.config['camera']['height']):
                raise ValueError('Camera resolution differs from configured calibration geometry')
            return await asyncio.to_thread(add_view,s.intrinsic_samples,image,frame_id)

    async def apply_intrinsics(service,obj):
        from .calibration_data import replace_intrinsics
        running=service.pm.active('camera')
        if running:await service.stop_camera()
        replace_intrinsics(root,service.config['camera'],obj)
        service.intrinsic_samples.clear()
        if running:await service.start_camera()
        return service.camera_calibration()

    def intrinsic_change_allowed(service):
        if service.capture.busy or service.active or any(service.pm.active(key) for key in ('recording','glim','calibration_record','calibration_tool','offline','export')):
            raise ValueError('Stop recording and calibration jobs before changing camera intrinsics')

    @app.post('/api/camera/intrinsics/calibrate')
    async def calibrate_intrinsics():
        from .camera_intrinsics import calibrate
        s=app.state.service
        async with s.lock:
            intrinsic_change_allowed(s)
            obj,quality=await asyncio.to_thread(calibrate,s.intrinsic_samples,s.config['camera'])
            return dict(**(await apply_intrinsics(s,obj)),quality=quality)

    @app.delete('/api/camera/intrinsics')
    async def delete_intrinsics():
        s=app.state.service
        async with s.lock:
            intrinsic_change_allowed(s)
            if s.camera_calibration()['intrinsics']['status']=='MISSING':raise ValueError('No camera intrinsics to delete')
            return await apply_intrinsics(s,{'calibrated':False})
    @app.get('/api/calibrations')
    async def calibrations():return await asyncio.to_thread(app.state.service.calibrations.list)
    @app.post('/api/calibrations')
    async def create_calibration(body:Create):
        s=app.state.service
        async with s.lock:return await asyncio.to_thread(s.calibrations.create,body.name)
    @app.get('/api/calibrations/{cid}')
    async def calibration_detail(cid:str):return await asyncio.to_thread(app.state.service.calibrations.detail,cid)
    @app.post('/api/calibrations/{cid}/action')
    async def calibration_action(cid:str,body:CalibrationAction):
        s=app.state.service
        async with s.lock:
            cal=s.calibrations
            if body.action=='capture_start':return await cal.capture_start(cid)
            if body.action=='capture_stop':return await cal.capture_stop(cid)
            if body.action in ('preprocess','initial_guess_manual','calibrate'):return await cal.run(cid,body.action)
            if body.action=='import':return await asyncio.to_thread(cal.import_result,cid)
            if body.action=='validate':return await asyncio.to_thread(cal.validate,cid,body.notes)
            if body.action=='cancel':
                p=cal.get(cid)
                item=s.pm.items.get('calibration_tool',{})
                if item and not Path(item['log']).is_relative_to(p):raise ValueError('Another calibration owns the active tool')
                await s.pm.stop('calibration_tool',20,cancel=True);return cal.detail(cid)
            raise ValueError('Unknown fixed calibration action')
    @app.get('/api/calibrations/{cid}/logs/{jid}')
    async def calibration_log(cid:str,jid:str):
        import re
        p=app.state.service.calibrations.get(cid)
        if not re.fullmatch(r'job_[a-f0-9]{12}',jid):raise ValueError('Invalid calibration job')
        log=p/'jobs'/jid/'tool.log'
        if not log.is_file() or log.is_symlink():raise ValueError('Calibration log not found')
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-60000));return Response(f.read(),media_type='text/plain')
    @app.get('/api/tools')
    async def tools_catalog(): return capabilities(root)
    @app.post('/api/tools/open')
    async def open_tool(body:ToolRequest):
        s=app.state.service
        async with s.lock: return await s.open_tool(body.session,body.run,body.tool,[x.model_dump() for x in body.additional])
    @app.get('/api/sessions/{sid}/edits')
    async def edits(sid:str):
        p=app.state.service.sessions.get(sid)
        result=[]
        for f in sorted((p/'edits').glob('*/workspace.json')):
            meta=read_json(f,{})
            # Derived workspace paths are local to this imported session.
            meta['maps']=[str(d) for d in sorted(f.parent.glob('map_[0-9][0-9]')) if d.is_dir()]
            meta['save_target']=str(f.parent/'saved_map')
            result.append(meta)
        return result
    def artifact(sid,path):
        p=app.state.service.sessions.get(sid); f=(p/path).resolve()
        if p.resolve() not in f.parents or not f.is_file() or any(x.is_symlink() for x in [p/path,*(p/path).parents] if x!=p.parent): raise ValueError('Invalid artifact path')
        if f.suffix not in ('.log','.json','.jsonl','.txt','.ply','.pcd','.yaml'): raise ValueError('Artifact type not exposed')
        if 'raw_bag' in f.relative_to(p).parts: raise ValueError('Raw bag downloads are not served by this endpoint; copy the session directory')
        return f
    @app.get('/api/sessions/{sid}/artifact')
    async def download(sid:str,path:str): return FileResponse(artifact(sid,path),filename=Path(path).name)
    @app.get('/api/sessions/{sid}/quality/{rid}')
    async def stats(sid:str,rid:str):
        s=app.state.service; return await asyncio.to_thread(quality,s.get_run(s.sessions.get(sid),rid))
    @app.get('/api/sessions/{sid}/cloud')
    async def cloud(sid:str,path:str):
        f=artifact(sid,path)
        if f.suffix!='.ply': raise ValueError('Select an exported PLY')
        return Response(await asyncio.to_thread(ply_stats,f,True),media_type='application/octet-stream')
    @app.get('/api/sessions/{sid}/cloud_stats')
    async def cloud_stats(sid:str,path:str):
        f=artifact(sid,path)
        if f.suffix!='.ply': raise ValueError('Select a PLY')
        return await asyncio.to_thread(ply_stats,f)
    @app.post('/api/sessions/{sid}/pcd')
    async def pcd(sid:str,path:str):
        f=artifact(sid,path)
        if f.suffix!='.ply': raise ValueError('Select a PLY')
        await asyncio.to_thread(ply_to_pcd,f,f.with_suffix('.pcd')); return {'ok':True}
    async def ws_origin_guard(ws):
        origin=ws.headers.get('origin'); host=ws.headers.get('host')
        if origin and origin not in (f'http://{host}',f'https://{host}'): await ws.close(code=1008); return False
        await ws.accept(); return True
    @app.websocket('/ws/preview')
    async def preview(ws:WebSocket):
        if not await ws_origin_guard(ws): return
        try:
            while True:
                s=app.state.service; payload=None; p=root/'.state/preview.bin'
                if s.mock and s.pm.active('driver'):
                    rng=np.random.default_rng(7); pts=rng.uniform(-5,5,(6000,4)); pts[:,2]*=.2; pts[:,3]=rng.uniform(0,255,6000); payload=encode(pts,time.time())
                elif not s.mock and s.pm.active('preview') and p.exists() and time.time()-p.stat().st_mtime<2: payload=p.read_bytes()
                if payload: await asyncio.wait_for(ws.send_bytes(payload),5)
                else: await ws.send_json({'state':'waiting_for_sensor','mock':s.mock})
                await asyncio.sleep(1/s.config['system']['preview']['hz'])
        except (WebSocketDisconnect,asyncio.TimeoutError,RuntimeError): pass
    @app.websocket('/ws/logs/{sid}/{rid}')
    async def logs(ws:WebSocket,sid:str,rid:str):
        if not await ws_origin_guard(ws): return
        try:
            s=app.state.service; p=s.get_run(s.sessions.get(sid),rid)/'job.log'; offset=0
            while True:
                if p.exists():
                    with p.open('rb') as f:
                        f.seek(offset); data=f.read(32768); offset=f.tell()
                    if data: await asyncio.wait_for(ws.send_text(data.decode(errors='replace')),5)
                await ws.send_json({'job':read_json(p.parent/'job.json',{})}); await asyncio.sleep(.5)
        except (WebSocketDisconnect,asyncio.TimeoutError,RuntimeError,ValueError): pass
    app.mount('/',StaticFiles(directory=root/'ui/frontend',html=True),name='frontend')
    return app

app=make_app()
