#!/usr/bin/env python3
"""Read real camera frames through the configured vendor Python/GStreamer path.
Stop the dashboard camera process first. Outputs stay under the project .state.
"""
import argparse, json, time
from factory_mapping.config import ROOT, load

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seconds',type=int,default=10,choices=range(2,61),metavar='2..60')
args=parser.parse_args()
import gi
gi.require_version('Gst','1.0')
gi.require_version('Tcam','1.0')
from gi.repository import Gst, Tcam
import cv2
import numpy as np

c=load()['camera']
if not c.get('gstreamer_pipeline'):raise SystemExit('Configure the trusted camera pipeline first')
Gst.init(None)
pipeline=Gst.parse_launch(c['gstreamer_pipeline']+' ! video/x-raw,format=BGR ! appsink name=check_sink sync=false max-buffers=2 drop=true')
sink=pipeline.get_by_name('check_sink'); stamps=[];deadline=time.monotonic()+args.seconds
try:
    pipeline.set_state(Gst.State.PLAYING)
    while time.monotonic()<deadline:
        sample=sink.emit('try-pull-sample',5*Gst.SECOND)
        if sample is None:
            error=pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
            raise RuntimeError(str(error.parse_error()) if error else 'No frames: stop any other camera process and check USB permissions')
        caps=sample.get_caps().get_structure(0);width=caps.get_value('width');height=caps.get_value('height')
        if (width,height)!=(c['width'],c['height']):raise RuntimeError('Unexpected camera geometry')
        buf=sample.get_buffer();stamps.append(buf.pts)
    ok,data=buf.map(Gst.MapFlags.READ)
    if not ok:raise RuntimeError('Cannot map image buffer')
    try:
        # BGR rows may be padded to 4-byte boundaries.
        stride=((width*3+3)//4)*4
        pixels=np.frombuffer(data.data,np.uint8).reshape(height,stride)[:,:width*3].reshape(height,width,3)
        cv2.imwrite(str(ROOT/'.state/camera_check.jpg'),pixels)
        mean=float(pixels.mean())
    finally:buf.unmap(data)
    hz=(len(stamps)-1)*1e9/(stamps[-1]-stamps[0]) if len(stamps)>1 and stamps[-1]>stamps[0] else 0
    result=dict(serial=c.get('serial_number'),width=width,height=height,frames=len(stamps),fps=hz,mean_pixel_value=mean,monotonic_timestamps=all(b>a for a,b in zip(stamps,stamps[1:])))
    (ROOT/'.state/camera_stream_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not c['expected_hz']*.9<=hz<=c['expected_hz']*1.1 or not result['monotonic_timestamps']:raise SystemExit('Camera rate/timestamp check failed')
finally:
    pipeline.set_state(Gst.State.NULL)
