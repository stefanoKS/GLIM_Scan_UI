"""Optional camera health and dependency checks; no ROS imports in the server."""
import asyncio
import math
import shutil
import time
from collections import deque
from .health import Rates
from .calibration_data import finite_vector


class CameraMetrics:
    def __init__(self,config):
        self.config=config;self.rate=Rates();self.width=0;self.height=0;self.frame_id=None
        self.info_seen=False;self.info_valid=False;self.info_time=None;self.deltas=deque(maxlen=150);self.previous=None;self.rewinds=0
    def image(self,stamp,width,height,frame_id):
        if not math.isfinite(stamp): return
        if self.previous is not None:
            dt=stamp-self.previous
            if dt<=0:self.rewinds+=1
            else:self.deltas.append(dt)
        self.previous=stamp;self.rate.add(stamp);self.width=width;self.height=height;self.frame_id=frame_id
    def info(self,width,height,k,distortion,model,frame):
        self.info_seen=True;self.info_time=time.monotonic();self.info_valid=False
        try:
            k=finite_vector(list(k),9);finite_vector(list(distortion))
            self.info_valid=(width==self.config['width'] and height==self.config['height'] and frame==self.config['frame_id'] and k[0]>0 and k[4]>0 and abs(k[8]-1)<1e-6 and model in ('plumb_bob','equidistant','fisheye','omnidir') and len(distortion)==(5 if model=='plumb_bob' else 4))
        except ValueError:pass
    def view(self):
        r=self.rate.view();expected=self.config['expected_hz']
        healthy=expected*.7<=r['hz']<=expected*1.3 and (self.width,self.height)==(self.config['width'],self.config['height']) and self.frame_id==self.config['frame_id']
        mean=sum(self.deltas)/len(self.deltas) if self.deltas else None
        jitter=(sum((x-mean)**2 for x in self.deltas)/len(self.deltas))**.5 if mean is not None else None
        return dict(r,state='healthy' if healthy else ('no_messages' if not r['hz'] else 'rate_or_geometry_abnormal'),healthy=healthy,image_hz=r['hz'],last_image_timestamp=r['stamp'],image_age=r['age'],width=self.width,height=self.height,frame_id=self.frame_id,camera_info_seen=self.info_seen,camera_info_valid=self.info_valid and self.info_time is not None and time.monotonic()-self.info_time<3,timestamp_age_sec=time.time()-r['stamp'] if r['stamp'] else None,timestamp_jitter_sec=jitter,timestamp_rewinds=self.rewinds)


async def check_camera_dependencies(config):
    async def run(args):
        if not shutil.which(args[0]): raise ValueError(args[0]+' unavailable; run scripts/install_camera.sh and source scripts/env.sh')
        p=await asyncio.create_subprocess_exec(*args,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
        try:out,_=await asyncio.wait_for(p.communicate(),10)
        except asyncio.TimeoutError:
            p.kill();await p.communicate();raise ValueError('Camera diagnostic timed out: '+args[0])
        return p.returncode,out.decode(errors='replace')
    rc,text=await run(['ros2','pkg','executables','gscam2'])
    if rc or 'gscam_main' not in text: raise ValueError('gscam2/gscam_main unavailable; install optional camera support')
    rc,text=await run(['gst-inspect-1.0','tcambin'])
    if rc:raise ValueError('tiscamera tcambin plugin unavailable; check tiscamera installation and GST_PLUGIN_PATH')
    rc,text=await run(['tcam-ctrl','--list'])
    serial=config.get('serial_number')
    if rc or not text.strip() or (serial and serial not in text):raise ValueError('Configured camera not found by tcam-ctrl; check USB3, serial and device permissions')
    return {'devices':text[-4000:],'plugin':'tcambin','publisher':'gscam_main'}
