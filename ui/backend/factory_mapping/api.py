import asyncio, contextlib, fcntl, json, os, time
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from .config import ROOT, validate_sensor
from .service import Service
from .glim_tools import capabilities
from .storage import atomic_json, read_json
from .preview import encode
from .quality import quality, ply_stats, ply_to_pcd

class Create(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name: str=Field(min_length=1,max_length=100)
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
class Network(BaseModel):
    model_config=ConfigDict(extra='forbid')
    lidar_ip: str
    host_ip: str
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
    @app.get('/api/sessions')
    async def sessions(): return await asyncio.to_thread(app.state.service.sessions.list)
    @app.post('/api/sessions')
    async def create(body:Create):
        s=app.state.service
        async with s.lock: return s.sessions.create(body.name,body.notes,s.config)
    @app.post('/api/action')
    async def action(body:Action):
        s=app.state.service
        async with s.lock:
            a=body.action
            if a=='driver_start': return await s.start_driver()
            if a=='driver_stop': await s.stop_driver()
            elif a=='record_start': return await s.start_recording(body.session or '')
            elif a=='record_stop': await s.stop_recording()
            elif a=='glim_start': return await s.start_glim(body.session or '',body.preset)
            elif a=='glim_viewer_start': return await s.start_glim(body.session or '',body.preset,viewer=True)
            elif a=='glim_stop': await s.stop_glim()
            elif a=='session_start':
                if not s.pm.active('driver'): await s.start_driver()
                for _ in range(15):
                    try: s.require_health(); break
                    except ValueError: await asyncio.sleep(1)
                await s.start_recording(body.session or '')
                try: await s.start_glim(body.session or '',body.preset)
                except Exception: await s.stop_session(); raise
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
            if sensor['ros_domain_id']==s.config['system'].get('offline_ros_domain_id',230): raise ValueError('Acquisition and offline ROS domains must differ')
            import yaml
            p=root/'config/livox/mid360.yaml'; tmp=p.with_suffix('.tmp'); tmp.write_text(yaml.safe_dump(sensor,sort_keys=False)); tmp.replace(p); s.config['sensor']=sensor
            s.event('network_config_updated'); return {'ok':True}
    @app.get('/api/tools')
    async def tools_catalog(): return capabilities(root)
    @app.post('/api/tools/open')
    async def open_tool(body:ToolRequest):
        s=app.state.service
        async with s.lock: return await s.open_tool(body.session,body.run,body.tool,[x.model_dump() for x in body.additional])
    @app.get('/api/sessions/{sid}/edits')
    async def edits(sid:str):
        p=app.state.service.sessions.get(sid)
        return [read_json(f) for f in sorted((p/'edits').glob('*/workspace.json'))]
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
