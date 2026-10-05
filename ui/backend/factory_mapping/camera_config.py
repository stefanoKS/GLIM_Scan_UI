"""Camera configuration is local, trusted input; no pipeline-writing HTTP endpoint."""
import math
import re
from pathlib import Path


def topic(value):
    return isinstance(value,str) and re.fullmatch(r'/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*',value) is not None


def config_path(root, value):
    p=Path(value)
    if p.is_absolute() or not p.parts or p.parts[0]!='config' or '..' in p.parts:
        raise ValueError('Calibration files must be project-relative paths under config/')
    result=root/p
    if (root/'config').resolve() not in result.resolve().parents or any(x.is_symlink() for x in [result,*result.parents] if x!=root.parent):
        raise ValueError('Calibration paths cannot escape config/ or use symlinks')
    return result


def validate_camera(c, sensor=None):
    if (c.get('driver'),c.get('source')) not in (('gscam2','tiscamera'),('librealsense','realsense')): raise ValueError('Unsupported camera driver/source')
    if c.get('source')=='realsense' and (not c.get('serial_number') or c.get('fps') not in (5,15,30)):
        raise ValueError('D405 requires a serial number and 5, 15 or 30 FPS')
    for key in ('image_topic','camera_info_topic'):
        if not topic(c.get(key)): raise ValueError('Invalid camera ROS topic: '+key)
    topics=[c['image_topic'],c['camera_info_topic']]
    if sensor: topics += [sensor['points_topic'],sensor['imu_topic']]
    if len(set(topics))!=len(topics): raise ValueError('Acquisition topics must be distinct')
    for key in ('frame_id','camera_name'):
        if not isinstance(c.get(key),str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_/]*',c[key]): raise ValueError('Invalid camera '+key)
    for key in ('width','height'):
        if type(c.get(key)) is not int or not 1<=c[key]<=16384: raise ValueError('Camera dimensions must be positive integers <=16384')
    for key,low,high in [('fps',.1,240),('expected_hz',.1,240),('preview_hz',.2,2),('preview_max_width',16,1280),('time_offset_sec',-3600,3600)]:
        value=c.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high: raise ValueError('Invalid camera '+key)
    for key in ('required_for_mapping','use_gst_timestamps','pipeline_validated'):
        if type(c.get(key)) is not bool: raise ValueError('Camera '+key+' must be boolean')
    serial=c.get('serial_number')
    if serial is not None and (not isinstance(serial,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',serial)): raise ValueError('Invalid camera serial number')
    pipeline=c.get('gstreamer_pipeline')
    if pipeline is not None and (not isinstance(pipeline,str) or len(pipeline)>8192 or '\x00' in pipeline): raise ValueError('Invalid local camera pipeline')
    for key in ('intrinsics_file','extrinsics_file'):
        value=c.get(key)
        if not isinstance(value,str) or not value.startswith('config/') or '..' in Path(value).parts: raise ValueError('Invalid '+key)
