"""Triangle mesh I/O and validation; no torch, NKSR, or ROS imports."""
from pathlib import Path
import numpy as np


def validate_mesh(vertices, faces):
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('Mesh requires finite [V,3] vertices, V > 0')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in 'iu':
        raise ValueError('Mesh requires integer [F,3] triangle faces, F > 0')
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Mesh face indices are out of range')
    return dict(vertex_count=len(vertices), face_count=len(faces),
                bounding_box_min=vertices.min(axis=0).tolist(), bounding_box_max=vertices.max(axis=0).tolist())


def write_mesh(path, vertices, faces):
    from plyfile import PlyData, PlyElement
    vertices=np.asarray(vertices,dtype=np.float32)
    stats = validate_mesh(vertices, faces)
    if len(vertices)>np.iinfo(np.int32).max: raise ValueError('Mesh exceeds int32 PLY index capacity')
    vertex = np.empty(len(vertices), dtype=[('x','<f4'),('y','<f4'),('z','<f4')])
    for i, axis in enumerate('xyz'): vertex[axis] = vertices[:,i]
    face = np.empty(len(faces), dtype=[('vertex_indices','<i4',(3,))])
    face['vertex_indices'] = faces
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial.ply')
    PlyData([PlyElement.describe(vertex,'vertex'), PlyElement.describe(face,'face')],
            text=False, byte_order='<').write(str(temp))
    temp.replace(path)
    stats['mesh_file_size'] = path.stat().st_size
    return stats


def inspect_mesh(path):
    from plyfile import PlyData
    try:
        mesh = PlyData.read(str(path),known_list_len={'face':{'vertex_indices':3}})
    except Exception as error:
        raise ValueError(f'Not a readable binary triangle PLY: {error}') from error
    vertex_count=len(mesh['vertex'].data);face_count=len(mesh['face'].data)
    if not vertex_count or not face_count: raise ValueError('Mesh requires vertices and triangle faces')
    lower=np.full(3,np.inf);upper=np.full(3,-np.inf)
    for start in range(0,vertex_count,65536):
        vertices=np.column_stack([mesh['vertex'][axis][start:start+65536] for axis in 'xyz'])
        if not np.isfinite(vertices).all(): raise ValueError('Mesh requires finite vertices')
        lower=np.minimum(lower,vertices.min(axis=0));upper=np.maximum(upper,vertices.max(axis=0))
    for start in range(0,face_count,65536):
        faces=np.asarray(mesh['face']['vertex_indices'][start:start+65536])
        if faces.dtype.kind=='O': faces=np.stack(faces)
        if faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu':
            raise ValueError('Mesh requires integer triangle faces')
        if faces.min()<0 or faces.max()>=vertex_count: raise ValueError('Mesh face indices are out of range')
    return dict(vertex_count=vertex_count,face_count=face_count,
                bounding_box_min=lower.tolist(),bounding_box_max=upper.tolist())


def merge_meshes(path, sources):
    from plyfile import PlyData
    path=Path(path)
    if path.exists(): raise ValueError('Output mesh already exists')
    sources=list(sources)
    if not sources: raise ValueError('No tile meshes to merge')
    stats=[inspect_mesh(source) for source in sources]
    vertex_count=sum(item['vertex_count'] for item in stats)
    face_count=sum(item['face_count'] for item in stats)
    if vertex_count>np.iinfo(np.int32).max: raise ValueError('Mesh exceeds int32 PLY index capacity')
    header=(f'ply\nformat binary_little_endian 1.0\nelement vertex {vertex_count}\n'
            f'property float x\nproperty float y\nproperty float z\nelement face {face_count}\n'
            'property list uchar int vertex_indices\nend_header\n').encode('ascii')
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix('.partial.ply')
    face_dtype=np.dtype([('count','u1'),('indices','<i4',(3,))])
    try:
        with partial.open('wb') as output:
            output.write(header)
            for source in sources:
                mesh=PlyData.read(str(source),known_list_len={'face':{'vertex_indices':3}})
                for start in range(0,len(mesh['vertex'].data),65536):
                    vertices=np.column_stack([mesh['vertex'][axis][start:start+65536] for axis in 'xyz'])
                    output.write(vertices.astype('<f4',copy=False).tobytes())
                del mesh
            offset=0
            for source,item in zip(sources,stats):
                mesh=PlyData.read(str(source),known_list_len={'face':{'vertex_indices':3}})
                for start in range(0,item['face_count'],65536):
                    faces=np.asarray(mesh['face']['vertex_indices'][start:start+65536])
                    if faces.dtype.kind=='O': faces=np.stack(faces)
                    records=np.empty(len(faces),dtype=face_dtype)
                    records['count']=3;records['indices']=faces.astype(np.int64)+offset
                    output.write(records.tobytes())
                offset+=item['vertex_count']
                del mesh
        result=inspect_mesh(partial)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    result['mesh_file_size']=path.stat().st_size
    return result
