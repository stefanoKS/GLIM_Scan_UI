import asyncio, json, os, shutil, sys, time, uuid
from datetime import datetime
from pathlib import Path
from . import commands
from .config import ROOT, PRESETS, load, validate_sensor
from .storage import Sessions, atomic_json, read_json, now, size
from .processes import ProcessManager
from .health import network, system_status

class Service:
    def __init__(self, root=ROOT, mock=False):
        self.root=root; self.mock=mock; self.config=load(root); self.sessions=Sessions(root,mock)
        self.pm=ProcessManager(root/'.state'); self.active=None; self.lock=asyncio.Lock(); self.errors=list(self.pm.recovery); self.net={}; self.closed=False
        self.record_started=None; self.rate_baseline={}
        for m in self.sessions.list():
            p=self.sessions.get(m['id'])
            if m['state'] in ('recording','mapping','stopping'):
                self.sessions.update(p,state='interrupted',recovery_note='Backend stopped unexpectedly. Inspect bag metadata before processing; raw data was preserved.')
            for f in (p/'processing').glob('*/job.json'):
                job=read_json(f,{})
                if job.get('state')=='running': job.update(state='interrupted'); atomic_json(f,job)
    def event(self,kind,**fields):
        entry=dict(time=now(),event=kind,**fields)
        path=self.root/'.state/application.jsonl'
        with path.open('a') as f: f.write(json.dumps(entry)+'\n')
        if self.active:
            with (self.active/'logs/application.jsonl').open('a') as f: f.write(json.dumps(entry)+'\n')
    def health(self):
        if self.mock:
            active=self.pm.active('driver')
            return {k:dict(state='mock' if active else 'stopped',hz=(10 if k=='lidar' else 200) if active else 0,stamp=time.time(),count=0) for k in ('lidar','imu')}
        obj=read_json(self.root/'.state/health.json',{})
        if time.time()-obj.get('updated_at',0)>3:
            return {k:dict(state='monitor_stale',hz=0) for k in ('lidar','imu')}
        return obj
    def status(self):
        processes=self.pm.view(); h=self.health()
        return dict(mock=self.mock,system=system_status(self.root),config=self.config,network=self.net,health=h,processes=processes,active_session=self.active.name if self.active else None,recording_elapsed=time.time()-self.record_started if self.record_started else 0,bag_size_bytes=size(self.active/'raw_bag') if self.active else 0,loop_detection='UNAVAILABLE' if self.config['system']['loop_closure']['enabled'] else 'OFF',errors=self.errors[-15:])
    async def start_process(self,key,args,log,out=None,done=None):
        if self.mock:
            args=[sys.executable,'-m','factory_mapping.mock_worker',key,str(out or self.root/'.state/mock')]
        result=await self.pm.start(key,args,log,commands.ros_env(self.config),done)
        self.event('process_start',role=key,pid=result['pid']); return result
    async def start_driver(self):
        if not self.mock:
            self.net=await network(self.config['sensor'])
            if self.net['state']!='reachable': raise ValueError(self.net['action'])
        result=await self.start_process('driver',commands.driver(self.root,self.config),self.root/'.state/driver.log')
        try:
            if not self.mock:
                for role in ('monitor','preview'):
                    await self.start_process(role,[sys.executable,'-c',f'from factory_mapping.ros_nodes import {role}; {role}()'],self.root/f'.state/{role}.log')
        except Exception:
            for role in ('preview','monitor','driver'): await self.pm.stop(role)
            raise
        return result
    async def stop_driver(self):
        if any(self.pm.active(k) for k in ('recording','glim')): raise ValueError('Stop recording and GLIM before stopping the sensor')
        for k in ('preview','monitor','driver'): await self.pm.stop(k,20)
    def require_health(self):
        if not self.pm.active('driver'): raise ValueError('Start Mid-360 first')
        if not self.mock and not all(self.health().get(k,{}).get('state')=='healthy' for k in ('lidar','imu')): raise ValueError('LiDAR and IMU must both have healthy message rates; run diagnostics')
        if shutil.disk_usage(self.root).free<self.config['system']['storage']['minimum_free_gb']*1e9: raise ValueError('Insufficient free disk space')
    async def start_recording(self,sid):
        self.require_health(); p=self.sessions.get(sid)
        if self.active and self.active!=p: raise ValueError('Another session is active')
        if (p/'raw_bag').exists(): raise ValueError('This session already contains a raw bag; create another session')
        # Re-snapshot at acquisition time, not merely when a name is reserved.
        shutil.rmtree(p/'config_snapshot'); shutil.copytree(self.root/'config',p/'config_snapshot'); atomic_json(p/'active_config.json',self.config)
        self.active=p; self.rate_baseline={k:self.health().get(k,{}).get('count',0) for k in ('lidar','imu')}
        async def done(item):
            if item['state']=='failed': self.errors.append('Recorder exited unexpectedly; inspect recording.log and bag metadata'); self.sessions.update(p,state='failed')
        try: result=await self.start_process('recording',commands.record(p,self.config),p/'logs/recording.log',p/'raw_bag',done)
        except Exception: self.active=None; raise
        self.record_started=time.time(); self.sessions.update(p,state='recording',start_time=now(),topics={k:self.config['sensor'][k] for k in ('points_topic','imu_topic')},lidar_ip=self.config['sensor']['lidar_ip'])
        return result
    async def stop_recording(self):
        if not self.active or self.record_started is None: return
        p=self.active; duration=time.time()-self.record_started if self.record_started else 0
        item=await self.pm.stop('recording',self.config['system']['shutdown']['recording_timeout'])
        finalized=self.mock or (p/'raw_bag/metadata.yaml').is_file()
        ok=bool(item and item['returncode']==0 and not item['forced'] and finalized)
        h=self.health(); rates={k:(h.get(k,{}).get('count',0)-self.rate_baseline.get(k,0))/duration if duration and not self.mock else None for k in ('lidar','imu')}
        if finalized and not self.mock:
            import yaml
            info=yaml.safe_load((p/'raw_bag/metadata.yaml').read_text())['rosbag2_bagfile_information']
            seconds=info['duration']['nanoseconds']/1e9
            for entry in info['topics_with_message_count']:
                for kind,key in [('lidar','points_topic'),('imu','imu_topic')]:
                    if entry['topic_metadata']['name']==self.config['sensor'][key]: rates[kind]=entry['message_count']/seconds if seconds else None
        self.sessions.update(p,state='recorded' if ok else 'failed',end_time=now(),duration=duration,average_lidar_hz=rates['lidar'],average_imu_hz=rates['imu'],disk_usage_bytes=size(p),bag_finalized=finalized)
        self.record_started=None
        if not ok: self.errors.append('Recording did not finalize cleanly; preserve raw_bag and inspect recording.log')
    def check_preset(self,preset):
        if preset not in PRESETS: raise ValueError('Unknown GLIM preset')
        if not self.mock and preset!='jetson_cpu' and not (self.root/'ros2_ws/install/glim/lib/libodometry_estimation_gpu.so').exists():
            raise ValueError('This installation has no CUDA GLIM module; select the CPU preset or build CUDA support')
    async def start_glim(self,sid,preset):
        self.check_preset(preset)
        self.require_health()
        if self.config['system']['loop_closure']['enabled']: raise ValueError('ScanContext is unavailable until separately validated')
        if self.pm.active('offline') or self.pm.active('export'): raise ValueError('Wait for offline processing/export to finish')
        p=self.sessions.get(sid)
        if self.active and self.active!=p: raise ValueError('Another session is active')
        if self.pm.active('glim'): raise ValueError('GLIM is already active')
        out=p/'glim_dump'
        if any(out.iterdir()): raise ValueError('Live output exists; use a new session or offline reprocessing')
        cfg=p/'config_snapshot'/('live_'+uuid.uuid4().hex[:8]); commands.preset_snapshot(self.root,p,preset,cfg)
        self.active=p
        self.sessions.update(p,glim_live=True,live_preset=preset)
        async def done(item):
            valid=self.mock or (out/'graph.bin').exists()
            self.sessions.update(p,live_result='completed' if item['returncode']==0 and valid and not item['forced'] else 'failed')
        return await self.start_process('glim',commands.glim(cfg,out),p/'logs/glim.log',out,done)
    async def stop_glim(self): await self.pm.stop('glim',self.config['system']['shutdown']['glim_timeout'])
    async def stop_session(self):
        p=self.active
        if not p: return
        if self.record_started is not None or self.pm.active('recording'): await self.stop_recording()
        await self.stop_glim()
        for role in ('driver','monitor','preview'):
            src=self.root/f'.state/{role}.log'
            if src.exists(): await asyncio.to_thread(shutil.copy2,src,p/'logs'/f'{role}.log')
        self.sessions.update(p,finalized_at=now(),disk_usage_bytes=await asyncio.to_thread(size,p)); self.event('session_stopped',session=p.name); self.active=None
    async def offline(self,sid,preset):
        self.check_preset(preset)
        if self.config['system']['loop_closure']['enabled']: raise ValueError('ScanContext is unavailable until validated')
        if any(self.pm.active(k) for k in ('glim','offline','export')): raise ValueError('GLIM or export is already active')
        p=self.sessions.get(sid)
        if self.active==p or read_json(p/'metadata.json',{})['state'] in ('created','recording','interrupted','failed'): raise ValueError('Session needs a finalized recording before processing')
        if not self.mock and not (p/'raw_bag/metadata.yaml').exists(): raise ValueError('No finalized ROS bag found')
        runs=p/'processing'; i=1
        while (runs/f'run_{i:03d}').exists(): i+=1
        run=runs/f'run_{i:03d}'; run.mkdir(); cfg=commands.preset_snapshot(self.root,p,preset,run/'config'); dump=run/'glim_dump'
        job=dict(id=run.name,session=sid,preset=preset,state='running',started_at=now(),mock=self.mock); atomic_json(run/'job.json',job)
        async def done(item):
            valid=self.mock or (dump/'graph.bin').exists()
            job.update(state=item['state'] if item['state']!='completed' or valid else 'failed',returncode=item['returncode'],forced=item['forced'],ended_at=now(),result=str(dump.relative_to(p)) if valid else None)
            atomic_json(run/'job.json',job)
            if job['state']=='failed': self.errors.append(f'GLIM processing failed: {sid}/{run.name}; inspect job.log')
        try: await self.start_process('offline',commands.glim(cfg,dump,p/'raw_bag'),run/'job.log',dump,done)
        except Exception as e: job.update(state='failed',error=str(e)); atomic_json(run/'job.json',job); raise
        return job
    async def export(self,sid,run_id):
        p=self.sessions.get(sid); run=self.get_run(p,run_id); job=read_json(run/'job.json',{})
        if job.get('state')!='completed': raise ValueError('Process this session successfully before export')
        if self.mock: raise ValueError('Mock results cannot be exported as maps')
        if any(self.pm.active(k) for k in ('glim','offline','export')): raise ValueError('GLIM or export is active')
        if not os.environ.get('DISPLAY'): raise ValueError('Official GLIM exporter needs an OpenGL display; run on a desktop workstation or configured Xvfb')
        output=p/'exports'/f'{run_id}_{uuid.uuid4().hex[:8]}.ply'
        async def done(item):
            if item['returncode']!=0 or not output.is_file(): self.errors.append('PLY export failed; inspect export.log')
        return await self.start_process('export',commands.export(run/'glim_dump',output,run/'config'),run/'export.log',done=done)
    def get_run(self,p,rid):
        import re
        if not re.fullmatch(r'run_\d{3,}',rid): raise ValueError('Invalid processing run')
        run=p/'processing'/rid
        if not run.is_dir() or run.is_symlink(): raise ValueError('Processing run not found')
        return run
    def delete_run(self,sid,rid):
        if any(self.pm.active(k) for k in ('glim','offline','export')): raise ValueError('Stop processing before deleting derived data')
        run=self.get_run(self.sessions.get(sid),rid); shutil.rmtree(run)
    async def background(self):
        while not self.closed:
            try:
                self.net={'state':'mock'} if self.mock else await network(self.config['sensor'])
                if self.pm.active('recording') and shutil.disk_usage(self.root).free<self.config['system']['storage']['minimum_free_gb']*1e9:
                    async with self.lock:
                        self.errors.append('Low disk threshold reached; stopping acquisition gracefully'); await self.stop_session()
            except Exception as e: self.errors.append(str(e))
            await asyncio.sleep(2)
    async def close(self):
        self.closed=True
        await self.stop_session()
        for k in ('offline','export','preview','monitor','driver'):
            if self.pm.items.get(k,{}).get('state')!='orphaned': await self.pm.stop(k,self.config['system']['shutdown']['glim_timeout'],cancel=True)
