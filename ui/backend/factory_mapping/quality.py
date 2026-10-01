"""Descriptive map statistics only; these do not measure absolute accuracy."""
from pathlib import Path
import numpy as np
from .storage import size
from .preview import encode

def quality(run):
    result={'size_bytes':size(run),'trajectory_duration':None,'trajectory_length':None,'start_end_displacement':None}
    p=run/'glim_dump/traj_lidar.txt'
    if p.exists() and p.stat().st_size:
        x=np.atleast_2d(np.loadtxt(p)); valid=x[np.isfinite(x).all(axis=1)]
        if len(valid):
            result.update(trajectory_duration=float(valid[-1,0]-valid[0,0]),trajectory_length=float(np.linalg.norm(np.diff(valid[:,1:4],axis=0),axis=1).sum()),start_end_displacement=float(np.linalg.norm(valid[-1,1:4]-valid[0,1:4])))
    return result

def ply_stats(path, preview=False):
    from plyfile import PlyData
    v=PlyData.read(str(path),mmap='r')['vertex'].data
    count=len(v); low=np.full(3,np.inf); high=np.full(3,-np.inf)
    for offset in range(0,count,100000):
        chunk=v[offset:offset+100000]; xyz=np.column_stack([chunk[k] for k in ('x','y','z')]); xyz=xyz[np.isfinite(xyz).all(axis=1)]
        if len(xyz): low=np.minimum(low,xyz.min(axis=0)); high=np.maximum(high,xyz.max(axis=0))
    if not preview: return {'number_of_points':count,'bounding_box':[low.tolist(),high.tolist()] if np.isfinite(low).all() else None,'intensity_available':'intensity' in v.dtype.names}
    v=v[::max(1,(count+49999)//50000)]; xyzi=np.column_stack([v[k] for k in ('x','y','z')]+[v['intensity'] if 'intensity' in v.dtype.names else np.zeros(len(v))]); return encode(xyzi)

def ply_to_pcd(source,target):
    from plyfile import PlyData
    v=PlyData.read(str(source),mmap='r')['vertex'].data
    names=['x','y','z']+(['intensity'] if 'intensity' in v.dtype.names else [])
    n=len(names)
    with target.open('xb') as f:
        header=f'# .PCD v0.7\nVERSION 0.7\nFIELDS {" ".join(names)}\nSIZE {" ".join(["4"]*n)}\nTYPE {" ".join(["F"]*n)}\nCOUNT {" ".join(["1"]*n)}\nWIDTH {len(v)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(v)}\nDATA binary\n'
        f.write(header.encode())
        for i in range(0,len(v),100000): f.write(np.column_stack([v[k][i:i+100000] for k in names]).astype('<f4').tobytes())
