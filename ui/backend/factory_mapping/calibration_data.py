"""Calibration values and conventions shared by acquisition and calibration jobs."""
import hashlib
import json
import math
import uuid
from pathlib import Path
import yaml
from .camera_config import config_path


def finite_vector(value,length=None):
    if not isinstance(value,(list,tuple)) or (length is not None and len(value)!=length) or not all(type(x) in (int,float) and math.isfinite(x) for x in value):
        raise ValueError('Expected a finite numeric vector'+(f' of length {length}' if length else ''))
    return list(value)


def transform(value):
    value=finite_vector(value,7)
    if abs(sum(x*x for x in value[3:])-1)>0.01: raise ValueError('T_lidar_camera quaternion must have unit length')
    return value


def camera_model(model):
    if model in ('equidistant','fisheye'): return 'fisheye'
    if model not in ('plumb_bob','omnidir'): raise ValueError('Unsupported camera distortion model')
    return model


def parse_intrinsics(obj,camera):
    if not isinstance(obj,dict) or obj.get('calibrated') is False: raise ValueError('No measured intrinsic calibration')
    w,h=obj.get('image_width'),obj.get('image_height')
    if (w,h)!=(camera['width'],camera['height']): raise ValueError('Intrinsic resolution differs from configured camera geometry')
    def matrix(key,rows,cols):
        v=obj.get(key,{})
        if v.get('rows')!=rows or v.get('cols')!=cols: raise ValueError('Invalid '+key+' shape')
        return finite_vector(v.get('data'),rows*cols)
    k=matrix('camera_matrix',3,3);matrix('rectification_matrix',3,3);matrix('projection_matrix',3,4)
    if k[0]<=0 or k[4]<=0 or abs(k[8]-1)>1e-6 or not 0<=k[2]<w or not 0<=k[5]<h: raise ValueError('Invalid intrinsic focal length or principal point')
    model=camera_model(obj.get('distortion_model'))
    d=obj.get('distortion_coefficients',{});dist=finite_vector(d.get('data'))
    expected=5 if model=='plumb_bob' else 4
    if len(dist)!=expected or d.get('rows')!=1 or d.get('cols')!=expected: raise ValueError('Unexpected distortion coefficient dimensions')
    intr=[k[0],k[4],k[2],k[5]]
    if model=='omnidir':
        xi=obj.get('xi')
        if type(xi) not in (int,float) or not math.isfinite(xi): raise ValueError('Omnidir requires measured xi in addition to ROS matrices')
        intr.append(xi)
    return dict(model=model,ros_model=obj['distortion_model'],width=w,height=h,intrinsics=intr,distortion=dist,K=k)


def intrinsics_status(root,camera):
    if not camera: return {'status':'MISSING'}
    path=config_path(root,camera['intrinsics_file'])
    if not path.is_file(): return {'status':'MISSING','path':camera['intrinsics_file']}
    try:
        obj=yaml.safe_load(path.read_text())
        if isinstance(obj,dict) and obj.get('calibrated') is False: return {'status':'MISSING','path':camera['intrinsics_file']}
        values=parse_intrinsics(obj,camera)
        return dict(status='VALID',path=camera['intrinsics_file'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**values)
    except (ValueError,TypeError,KeyError,AttributeError,yaml.YAMLError) as e: return {'status':'INVALID','path':camera['intrinsics_file'],'error':str(e)}


def require_intrinsics(root,camera):
    info=intrinsics_status(root,camera)
    if info['status']!='VALID': raise ValueError('Valid camera intrinsics are required: '+info.get('error',info['status']))
    return info


def parse_result(path,expected):
    if not path.is_file(): raise ValueError('calib.json result file is missing')
    try:
        obj=json.loads(path.read_text())
        pose=transform(obj['results']['T_lidar_camera'])
        cam=obj['camera']
        if camera_model(cam['camera_model'])!=expected['model']: raise ValueError('Result camera model differs from intrinsic snapshot')
        for field,key in [('intrinsics','intrinsics'),('distortion_coeffs','distortion')]:
            value=finite_vector(cam[field],len(expected[key]))
            if any(not math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-8) for a,b in zip(value,expected[key])): raise ValueError('Result intrinsics/distortion differ from snapshot')
        return pose
    except (KeyError,TypeError,json.JSONDecodeError) as e: raise ValueError('Malformed calib.json result') from e


def camera_metadata(root,config):
    enabled=config['system'].get('camera',{}).get('enabled',False);c=config.get('camera',{})
    result=dict(enabled=enabled,**{k:c.get(k) for k in ('camera_name','model','serial_number','image_topic','camera_info_topic','frame_id','fps','time_offset_sec','intrinsics_file','extrinsics_file')})
    result['measured_image_hz']=None
    for key in ('intrinsics_file','extrinsics_file'):
        if c.get(key):
            p=config_path(root,c[key]);result[key+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return result


def atomic_yaml(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(yaml.safe_dump(obj,sort_keys=False));tmp.replace(path)
