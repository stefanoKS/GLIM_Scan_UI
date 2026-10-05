"""Optional camera health and dependency checks; no ROS imports in the server."""
import asyncio
import math
import shutil
import re
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
            k=finite_vector([float(value) for value in k],9);finite_vector([float(value) for value in distortion])
            self.info_valid=(width==self.config['width'] and height==self.config['height'] and frame==self.config['frame_id'] and k[0]>0 and k[4]>0 and abs(k[8]-1)<1e-6 and model in ('plumb_bob','equidistant','fisheye','omnidir') and len(distortion)==(5 if model=='plumb_bob' else 4))
        except (TypeError,ValueError):pass
    def view(self):
        r=self.rate.view();expected=self.config['expected_hz']
        healthy=expected*.7<=r['hz']<=expected*1.3 and (self.width,self.height)==(self.config['width'],self.config['height']) and self.frame_id==self.config['frame_id']
        mean=sum(self.deltas)/len(self.deltas) if self.deltas else None
        jitter=(sum((x-mean)**2 for x in self.deltas)/len(self.deltas))**.5 if mean is not None else None
        return dict(r,state='healthy' if healthy else ('no_messages' if not r['hz'] else 'rate_or_geometry_abnormal'),healthy=healthy,image_hz=r['hz'],last_image_timestamp=r['stamp'],image_age=r['age'],width=self.width,height=self.height,frame_id=self.frame_id,camera_info_seen=self.info_seen,camera_info_valid=self.info_valid and self.info_time is not None and time.monotonic()-self.info_time<3,timestamp_age_sec=time.time()-r['stamp'] if r['stamp'] else None,timestamp_jitter_sec=jitter,timestamp_rewinds=self.rewinds)


async def check_camera_dependencies(config, root=None):
    async def run(args):
        if not shutil.which(args[0]): raise ValueError(args[0]+' unavailable; run scripts/install_camera.sh and source scripts/env.sh')
        p=await asyncio.create_subprocess_exec(*args,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
        try:out,_=await asyncio.wait_for(p.communicate(),10)
        except asyncio.TimeoutError:
            p.kill();await p.communicate();raise ValueError('Camera diagnostic timed out: '+args[0])
        return p.returncode,out.decode(errors='replace')
    if config['source']=='realsense':
        import sys,json
        from .config import ROOT
        rc,text=await run([sys.executable,'-m','factory_mapping.realsense_camera','--root',str(root or ROOT),'--config',json.dumps(config),'--extract'])
        if rc: raise ValueError('D405 unavailable; install scripts/install_d405.sh and check USB/serial: '+text[-2000:])
        return {'devices':text,'publisher':'librealsense RGB only'}
    rc,text=await run(['ros2','pkg','executables','gscam2'])
    if rc or 'gscam_main' not in text: raise ValueError('gscam2/gscam_main unavailable; install optional camera support')
    rc,text=await run(['gst-inspect-1.0','tcambin'])
    if rc:raise ValueError('tiscamera tcambin plugin unavailable; check tiscamera installation and GST_PLUGIN_PATH')
    rc,text=await run(['tcam-ctrl','--list'])
    serial=config.get('serial_number')
    devices=re.findall(r'Model: (.*?) Serial: (\S+) Type: (\S+)',text)
    if rc or not devices or (serial and serial not in [d[1] for d in devices]):raise ValueError('Configured camera not found by tcam-ctrl; check USB3, serial and device permissions')
    return {'devices':text[-4000:],'plugin':'tcambin','publisher':'gscam_main'}


def detect_camera(config, sysfs=None):
    """USB presence only; stream readiness is reported separately."""
    from pathlib import Path
    devices=[]
    for device in (Path(sysfs) if sysfs else Path('/sys/bus/usb/devices')).glob('*'):
        try:
            if (device/'idVendor').read_text().strip().lower()!=('8086' if config.get('source')=='realsense' else '199e'): continue
            model=(device/'product').read_text().strip()
            serial=(device/'serial').read_text().strip()
            if ('405' if config.get('source')=='realsense' else '33UX287') not in model.replace(' ','').upper(): continue
            # D405 USB descriptors identify the ASIC, not the SDK device serial.
            expected=config.get('usb_serial_number') if config.get('source')=='realsense' else config.get('serial_number')
            if expected and expected!=serial: continue
            devices.append(dict(model=model,serial=serial,usb_speed_mbps=(device/'speed').read_text().strip()))
        except OSError: continue
    return dict(detected=bool(devices),basis='USB enumeration; does not prove frame delivery',devices=devices)
