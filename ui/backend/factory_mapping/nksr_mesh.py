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
    mesh = PlyData.read(str(path),known_list_len={'face':{'vertex_indices':3}})
    vertices = np.column_stack([mesh['vertex'][axis] for axis in 'xyz'])
    faces = np.asarray(mesh['face']['vertex_indices'])
    if faces.dtype.kind=='O': faces=np.stack(faces)
    return validate_mesh(vertices, faces)
