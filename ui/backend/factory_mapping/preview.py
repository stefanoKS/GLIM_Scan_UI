"""FMPC v1: 16-byte little-endian header (magic,count,stamp_sec), then float32 XYZI."""
import struct
import numpy as np
HEADER=struct.Struct('<4sId')
def encode(points,stamp=0.0,max_points=50000,voxel=0.1):
    points=np.asarray(points,dtype=np.float32).reshape(-1,4)
    points=points[np.isfinite(points).all(axis=1)]
    if len(points):
        # Bound voxel work and output even for enormous input clouds.
        points=points[::max(1,(len(points)+max_points*4-1)//(max_points*4))]
        _,indices=np.unique(np.floor(points[:,:3]/voxel).astype(np.int64),axis=0,return_index=True)
        points=points[np.sort(indices)]
        points=points[::max(1,(len(points)+max_points-1)//max_points)][:max_points]
    return HEADER.pack(b'FMPC',len(points),float(stamp))+points.astype('<f4').tobytes()

def decode(payload):
    magic,count,stamp=HEADER.unpack_from(payload)
    if magic!=b'FMPC' or len(payload)!=16+16*count: raise ValueError('Invalid preview packet')
    return np.frombuffer(payload, dtype='<f4',offset=16).reshape(count,4),stamp
