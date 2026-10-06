"""Semantic capture orchestration above the existing session/process service.

Only this controller owns high-level state. ROS, rosbag and GLIM still run through
Service/ProcessManager. A failed estimator never rolls back a running recorder.
"""
import asyncio
import platform
from datetime import datetime
from .storage import atomic_json, read_json, now

BUSY = {'PREFLIGHT', 'SCANNING', 'FINALIZING', 'PROCESSING'}


class CaptureController:
    def __init__(self, service):
        self.s = service
        self.path = service.root / '.state/capture.json'
        self.data = read_json(self.path, {'state': 'READY', 'session': None, 'mode': None, 'warnings': [], 'error': None})
        self.task = None
        self.stop_requested = False
        if self.data.get('state') in BUSY:
            self.data.update(state='COMPLETE', outcome='interrupted', error='The application restarted during capture or processing. Raw data is preserved; inspect the scan in Library.')
            atomic_json(self.path, self.data)

    @property
    def busy(self):
        return self.data['state'] in BUSY

    def capabilities(self):
        available = self.s.glim_available()
        gpu = available and read_json(self.s.root / '.state/build_capabilities.json', {}).get('cuda', False) and (self.s.root / 'ros2_ws/install/glim/lib/libodometry_estimation_gpu.so').is_file()
        jetson = platform.machine() == 'aarch64'
        defaults = dict(live_glim=False, auto_process=True, mapping_preset='auto')
        settings = {**defaults, **read_json(self.s.root / '.state/capture_settings.json', {})}
        if self.s.config['system']['deployment_mode'] == 'record_only':
            settings.update(live_glim=False, auto_process=False)
        preset = settings['mapping_preset']
        if preset == 'auto': preset = 'jetson_gpu' if gpu else 'jetson_cpu'
        return dict(processing=available, gpu=bool(gpu), cpu=available, camera='camera' in self.s.config,
                    profile='Jetson' if jetson else 'Workstation', record_only=not available,
                    preset=preset, settings=settings)

    def view(self):
        return {**self.data, 'busy': self.busy, 'capabilities': self.capabilities(),
                'recovering': self.s.recovering,
                'can_start': not self.s.recovering and not self.busy and not self.s.active and not any(item['state']=='orphaned' for item in self.s.pm.items.values()) and not any(self.s.pm.active(k) for k in ('recording','glim','offline','export','tool','calibration_record','calibration_tool'))}

    def transition(self, state, **fields):
        self.data.update(state=state, updated_at=now(), **fields)
        atomic_json(self.path, self.data)
        if self.data.get('session'):
            self.s.sessions.update(self.s.sessions.get(self.data['session']), capture=dict(self.data))

    def warn(self, message):
        if message not in self.data['warnings']:
            self.data['warnings'].append(message)
            self.transition(self.data['state'])

    async def start_scan(self):
        return await self._begin('scan')

    async def start_camera_recording(self):
        return await self._begin('camera')

    async def _begin(self, mode):
        async with self.s.lock:
            if not self.view()['can_start']:
                raise ValueError('Finish the current capture, processing or calibration job first')
            if mode == 'camera' and 'camera' not in self.s.config:
                raise ValueError('Camera support is not configured on this system')
            self.stop_requested = False
            name = ('Scan ' if mode == 'scan' else 'Camera ') + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            session = self.s.sessions.create(name, '', self.s.config)
            self.s.sessions.update(self.s.sessions.get(session['id']), kind=mode)
            self.data = dict(state='READY', session=session['id'], mode=mode, warnings=[], error=None, outcome=None, run=None, live_glim_started=False)
            if mode == 'scan' and not self.s.config['system']['camera']['enabled']:
                self.data['warnings'].append('RGB recording is disabled; this scan records LiDAR and IMU only, without camera images for colorization.')
            self.transition('PREFLIGHT')
            self.task = asyncio.create_task(self._start())
            return self.view()

    async def _start(self):
        async with self.s.lock:
            try:
                await self.s.start_recording(self.data['session'], camera_only=self.data['mode']=='camera')
                if self.stop_requested or self.s.closed:
                    await self._finish(process=False)
                    return
                self.transition('SCANNING')
                caps = self.capabilities()
                if self.data['mode']=='scan' and caps['processing'] and caps['settings']['live_glim']:
                    try:
                        await self.s.start_glim(self.data['session'], caps['preset'])
                        self.transition('SCANNING', live_glim_started=True)
                    except Exception as error:
                        self.warn('Live mapping could not start; raw recording continues. '+str(error))
            except Exception as error:
                if self.s.active:
                    try: await self.s.stop_session()
                    except Exception as cleanup: self.s.errors.append(str(cleanup))
                p = self.s.sessions.get(self.data['session'])
                if not (p/'raw_bag').exists(): self.s.sessions.update(p, state='failed', failure_stage='preflight')
                self.transition('COMPLETE', outcome='failed', error=str(error))

    async def stop_scan(self):
        return await self._request_stop('scan')

    async def stop_camera_recording(self):
        return await self._request_stop('camera')

    async def _request_stop(self, mode):
        # A stop during the bounded preflight is remembered without waiting on
        # its service lock; no live estimator is launched afterwards.
        if self.data['mode'] != mode and self.busy: raise ValueError('Use the stop action for the active capture')
        if self.data['state']=='PREFLIGHT':
            self.stop_requested = True
            return self.view()
        async with self.s.lock:
            if self.data['state'] in ('READY','COMPLETE','FINALIZING','PROCESSING'): return self.view()
            self.transition('FINALIZING')
            self.task = asyncio.create_task(self._stop_task())
            return self.view()

    async def _stop_task(self):
        async with self.s.lock:
            try: await self._finish()
            except Exception as error: self.transition('COMPLETE', outcome='failed', error=str(error))

    async def _finish(self, process=True):
        self.transition('FINALIZING')
        await self.s.stop_session()
        meta = read_json(self.s.sessions.get(self.data['session'])/'metadata.json', {})
        if meta.get('state')!='recorded' or not meta.get('bag_finalized'):
            self.transition('COMPLETE', outcome='failed', error='Recording did not finalize cleanly. Raw files remain in Library; inspect diagnostics.')
            return
        caps = self.capabilities()
        if process and not self.s.closed and self.data['mode']=='scan' and caps['processing'] and caps['settings']['auto_process']:
            try:
                job = await self.s.offline(self.data['session'], caps['preset'])
                self.transition('PROCESSING', run=job['id'])
                return
            except Exception as error:
                self.warn('Scan saved; automatic processing could not start. '+str(error))
        self.transition('COMPLETE', outcome='recorded', error=None)

    async def reconcile(self):
        """Called by Service background while holding its mutation lock."""
        if self.data['state']=='SCANNING':
            if not self.s.pm.active('recording') or not self.s.active:
                self.warn('Recording ended outside the normal Stop action; inspect the saved scan.')
                try: await self._finish(process=False)
                except Exception as error: self.transition('COMPLETE', outcome='failed', error=str(error))
            elif self.data.get('live_glim_started') and self.s.pm.items.get('glim', {}).get('state')=='failed':
                self.warn('Live mapping failed; raw recording is still running. Offline processing remains available after stopping.')
        elif self.data['state']=='PROCESSING' and not self.s.pm.active('offline'):
            p = self.s.sessions.get(self.data['session'])/'processing'/self.data['run']/'job.json'
            job = read_json(p, {})
            ok = job.get('state')=='completed'
            self.transition('COMPLETE', outcome='processed' if ok else 'processing_failed',
                            error=None if ok else 'Scan is saved, but processing did not complete. Retry from Library.')

    async def close(self):
        self.stop_requested = True
        if self.task and not self.task.done(): await self.task
        if self.data['state']=='SCANNING':
            async with self.s.lock: await self._finish(process=False)
        elif self.data['state']=='PROCESSING':
            self.transition('COMPLETE', outcome='processing_interrupted', error='Processing stopped when the application closed; raw recording is preserved.')
