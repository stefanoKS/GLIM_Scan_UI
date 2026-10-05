"""Camera preferences resolve to a measured stream; callers hold Service.lock."""
import asyncio
import copy
import time
from .config import CAMERA_PROFILES, load_camera_profile
from .camera import detect_camera, check_camera_dependencies


class CameraSelection:
    startup_timeout = 6.0
    dependency_timeout = 6.0

    def __init__(self, service):
        self.s = service
        self.active_profile = None
        self.fallback_reason = None
        self.fallback_used = False
        self.candidates = {}
        self.refresh_detection()

    def refresh_detection(self):
        for profile in CAMERA_PROFILES:
            try: c = load_camera_profile(self.s.root, profile, self.s.config['sensor'])
            except (OSError, ValueError) as error:
                self.candidates[profile]=dict(detected=False,dependency_ok=False,stream_healthy=False,reason=str(error))
                continue
            detection = detect_camera(c) if not self.s.mock else dict(detected=False, devices=[], basis='MOCK')
            state=self.candidates.setdefault(profile,dict(dependency_ok=None,stream_healthy=None,reason=None))
            state.update(detection)

    def view(self):
        self.refresh_detection()
        candidates = copy.deepcopy(self.candidates)
        for profile,candidate in candidates.items():
            if profile!=self.active_profile and candidate['stream_healthy']: candidate['stream_healthy']=False
        if self.active_profile:
            candidates[self.active_profile]['stream_healthy'] = self.usable(self.s.camera_health())
        return dict(preferred_profile=self.s.config['system']['camera']['profile'],
                    active_profile=self.active_profile, fallback_used=self.fallback_used,
                    fallback_reason=self.fallback_reason, candidates=candidates)

    def publisher_alive(self):
        item=self.s.pm.items.get('camera',{})
        process=item.get('process')
        return item.get('state')=='running' and (process is None or process.returncode is None)

    def usable(self, h):
        c = self.s.config['camera']
        return bool(self.publisher_alive() and h.get('healthy') and h.get('camera_info_seen')
                    and h.get('camera_info_age') is not None and h['camera_info_age'] < 3
                    and h.get('image_age') is not None and h['image_age'] < 2
                    and (h.get('width'),h.get('height')) == (c['width'],c['height'])
                    and c['expected_hz']*.7 <= h.get('image_hz',0) <= c['expected_hz']*1.3)

    def reason(self, profile, h):
        c=self.s.config['camera']
        if not self.publisher_alive(): return f'{profile} publisher exited (pipeline failed to start); inspect camera.log'
        if not h.get('image_hz'): return f'{profile} no fresh image messages'
        if (h.get('width'),h.get('height')) != (c['width'],c['height']):
            return f"{profile} received {h.get('width')}x{h.get('height')} but expected {c['width']}x{c['height']}"
        if not h.get('camera_info_seen') or h.get('camera_info_age') is None or h['camera_info_age']>=3: return f'{profile} CameraInfo missing'
        return f"{profile} stream unhealthy: {h.get('image_hz',0):.1f} FPS, expected {c['expected_hz']}; {h.get('state')}"

    async def resolve(self, required=True, profile=None):
        s=self.s
        if s.active or s.pm.active('recording') or s.pm.active('calibration_record') or s.calibrations.active:
            if self.active_profile and (profile is None or profile==self.active_profile) and self.usable(s.camera_health()):
                return True
            raise ValueError('Camera is locked during acquisition; stop capture before resolving another camera')
        preferred=s.config['system']['camera']['profile']
        order=[profile] if profile else (list(CAMERA_PROFILES) if preferred=='auto' else [preferred,*[p for p in CAMERA_PROFILES if p!=preferred]])
        self.refresh_detection()
        failures=[]
        for candidate in order:
            state=self.candidates[candidate]
            if candidate==self.active_profile and self.usable(s.camera_health()):
                self.fallback_used=bool(failures) or (preferred!='auto' and candidate!=preferred)
                self.fallback_reason='; '.join(failures) or (self.fallback_reason if self.fallback_used else None)
                return True
            # Never publish a new candidate until the previous process group has exited.
            await s._stop_camera_candidate()
            self.active_profile=None
            state.update(dependency_ok=None,stream_healthy=None,reason=None)
            if not s.mock and not state['detected']:
                state['reason']=f'{candidate} not detected';failures.append(state['reason']);continue
            s.intrinsic_samples=[]
            s.config['camera']=load_camera_profile(s.root,candidate,s.config['sensor'])
            try:
                if not s.mock:
                    try:
                        state['dependency']=await asyncio.wait_for(check_camera_dependencies(s.config['camera'],s.root),self.dependency_timeout)
                        state['dependency_ok']=True
                    except (ValueError, asyncio.TimeoutError) as error:
                        state['dependency_ok']=False
                        raise ValueError(f'{candidate} dependency unavailable: {error or "diagnostic timeout"}') from error
                self.active_profile=candidate
                started=time.monotonic()
                await s._start_camera_candidate()
                deadline=started+self.startup_timeout
                while True:
                    h=s.camera_health()
                    if self.usable(h): break
                    if not self.publisher_alive() or time.monotonic()>=deadline:
                        raise ValueError(self.reason(candidate,h))
                    await asyncio.sleep(.1)
                state.update(stream_healthy=True,reason=None,startup_seconds=time.monotonic()-started)
                self.fallback_used=bool(failures)
                self.fallback_reason='; '.join(failures) or None
                s.event('camera_selected',**self.view())
                return True
            except (ValueError, RuntimeError, OSError) as error:
                state.update(stream_healthy=False,reason=str(error));failures.append(str(error))
                s.event('camera_candidate_failed',profile=candidate,reason=str(error))
                await s._stop_camera_candidate()
                self.active_profile=None
        self.fallback_used=True
        self.fallback_reason='; '.join(failures)
        if required: raise ValueError('RGB unavailable: '+self.fallback_reason)
        return False

    def acquisition(self, requested, enabled):
        config=copy.deepcopy(self.s.config)
        config['system']['camera']['enabled']=enabled
        selected=self.view()
        selected.update(rgb_requested=requested, rgb_recorded=False,
                        active_profile=self.active_profile if enabled else None,
                        stream=self.s.camera_health() if enabled else None)
        if not requested: selected.update(fallback_used=False,fallback_reason=None)
        config['camera_selection']=selected
        if enabled: config['system']['camera']['profile']=self.active_profile
        return config
