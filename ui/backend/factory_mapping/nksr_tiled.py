"""Disk-backed spatial partitioning and sequential, process-isolated NKSR tiles."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import numpy as np
from .nksr_mesh import inspect_mesh, merge_meshes
from .nksr_worker import WorkerError, atomic_json, DEFAULT_NKSR_TARGET_VOXEL_M


def partition_input(source, scratch, tile_size, event):
    scratch=Path(scratch)
    arrays=[]
    try:
        with zipfile.ZipFile(source) as archive:
            for name in ('points','sensor_origins'):
                target=scratch/(name+'.npy')
                with archive.open(name+'.npy') as original, target.open('wb') as output:
                    shutil.copyfileobj(original,output,length=1024*1024)
                arrays.append(np.load(target,mmap_mode='r',allow_pickle=False))
        points,sensors=arrays
        if points.ndim!=2 or points.shape[1]!=3 or not len(points) or sensors.shape!=points.shape:
            raise ValueError('Expected nonempty paired points and sensor_origins with shape [N,3]')
        tiles={};lower=np.full(3,np.inf);upper=np.full(3,-np.inf)
        for start in range(0,len(points),65536):
            block=np.asarray(points[start:start+65536],dtype=np.float32)
            origins=np.asarray(sensors[start:start+65536],dtype=np.float32)
            if not np.isfinite(block).all() or not np.isfinite(origins).all():
                raise ValueError('Input contains nonfinite values')
            lower=np.minimum(lower,block.min(axis=0));upper=np.maximum(upper,block.max(axis=0))
            cells=np.floor(block.astype(np.float64)/tile_size)
            if not np.isfinite(cells).all() or np.any(np.abs(cells)>=2**63):
                raise ValueError('Tile coordinates exceed int64 capacity')
            keys,inverse=np.unique(cells.astype(np.int64),axis=0,return_inverse=True)
            order=np.argsort(inverse,kind='stable')
            boundaries=np.r_[0,np.cumsum(np.bincount(inverse))]
            for index,key in enumerate(keys):
                cell=tuple(int(value) for value in key)
                if cell not in tiles:
                    tiles[cell]=dict(id=f'tile_{len(tiles)+1:06d}',cell=list(cell),points=0)
                tile=tiles[cell]
                selected=order[boundaries[index]:boundaries[index+1]]
                records=np.column_stack((block[selected],origins[selected])).astype('<f4',copy=False)
                with (scratch/(tile['id']+'.bin')).open('ab') as output: output.write(records.tobytes())
                tile['points']+=len(selected)
            event('PARTITIONING',points_partitioned=min(start+65536,len(points)),input_points=len(points),
                  tile_count=len(tiles),message=f'Partitioning {min(start+65536,len(points)):,} / {len(points):,} points')
        return list(tiles.values()),dict(input_points=len(points),input_bbox=[lower.tolist(),upper.tolist()])
    except (ValueError,KeyError,zipfile.BadZipFile) as error:
        raise WorkerError('INPUT_INVALID',str(error)) from error


def run_tile(args):
    process=subprocess.Popen(args)
    try:
        return process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()


def run_tiled(settings, event):
    started=time.monotonic()
    output=settings.output
    if output.exists(): raise WorkerError('MESH_INVALID','Output exists; choose a new output path')
    output.parent.mkdir(parents=True,exist_ok=True)
    tile_root=output.parent/'tiles'
    if tile_root.exists(): raise WorkerError('MESH_INVALID','Tile output exists; choose a new output directory')
    tile_root.mkdir()
    manifest=output.parent/'tiles.json'
    worker=Path(__file__).resolve().parents[3]/'tools/nksr_worker.py'
    with tempfile.TemporaryDirectory(prefix='nksr-tiles-',dir=output.parent) as directory:
        scratch=Path(directory)
        event('PARTITIONING',message='Partitioning independent spatial tiles on disk')
        tiles,input_stats=partition_input(settings.input,scratch,settings.tile_size,event)
        meshes=[]
        state=dict(tile_size_m=settings.tile_size,overlap_ratio=0,tiles=tiles,**input_stats)
        atomic_json(manifest,state)
        for index,tile in enumerate(tiles,1):
            tile_dir=tile_root/tile['id'];tile_dir.mkdir()
            binary=scratch/(tile['id']+'.bin')
            if tile['points']<settings.normal_knn:
                tile.update(state='SKIPPED',reason='Fewer points than normal KNN')
                binary.unlink();atomic_json(manifest,state)
                event('RECONSTRUCTING_TILES',tile_index=index,tile_count=len(tiles),
                      message=f'Tile {index}/{len(tiles)} skipped: too few points')
                continue
            tile_input=scratch/'tile_input.npz'
            records=np.memmap(binary,dtype='<f4',mode='r',shape=(tile['points'],6))
            np.savez(tile_input,points=records[:,:3],sensor_origins=records[:,3:])
            del records
            binary.unlink()
            tile_output=tile_dir/'mesh.ply';tile_metadata=tile_dir/'nksr_metadata.json'
            args=[sys.executable,str(worker),'--input',str(tile_input),'--output',str(tile_output),
                  '--metadata',str(tile_metadata),'--progress',str(tile_dir/'progress.json'),
                  '--mode','full','--tile-worker','--device',settings.device,
                  '--normal-knn',str(settings.normal_knn),'--normal-drop-angle-deg',str(settings.normal_drop_angle_deg),
                  '--mise-iter',str(settings.mise_iter)]
            tile['state']='RUNNING';atomic_json(manifest,state)
            event('RECONSTRUCTING_TILES',tile_index=index,tile_count=len(tiles),tile_points=tile['points'],
                  message=f'Tile {index}/{len(tiles)}: {tile["points"]:,} points, isolated worker')
            try:
                returncode=run_tile(args)
            except BaseException:
                tile['state']='INTERRUPTED';atomic_json(manifest,state)
                raise
            finally:
                tile_input.unlink(missing_ok=True)
            tile['returncode']=returncode
            details=json.loads(tile_metadata.read_text()) if tile_metadata.is_file() else {}
            if returncode:
                tile['state']='FAILED';atomic_json(manifest,state)
                code=details.get('error_type','TILE_WORKER_FAILED')
                raise WorkerError(code,f'Tile {index}/{len(tiles)} failed (exit {returncode}). '
                                  'Try a smaller tile size. '+details.get('error','See job.log'),
                                  dict(tile_id=tile['id'],tile_index=index,tile_count=len(tiles)))
            if details.get('status')=='EMPTY_TILE':
                tile.update(state='SKIPPED',reason='No surface after normal filtering or extraction')
            else:
                tile.update(state='COMPLETED',**inspect_mesh(tile_output))
                meshes.append(tile_output)
            atomic_json(manifest,state)
            event('RECONSTRUCTING_TILES',tile_index=index,tile_count=len(tiles),
                  message=f'Tile {index}/{len(tiles)} {tile["state"].lower()}')
        if not meshes: raise WorkerError('MESH_INVALID','No tile produced triangles; increase tile size or check input')
        event('MERGING_MESHES',tile_count=len(tiles),message=f'Assembling {len(meshes)} independent tile meshes')
        stats=merge_meshes(output,meshes)
        metadata=dict(stats,**input_stats,python=sys.executable,requested_mode='low_ram',actual_mode='low_ram',
                      tile_size_m=settings.tile_size,tile_count=len(tiles),completed_tiles=len(meshes),
                      skipped_tiles=len(tiles)-len(meshes),overlap_ratio=0,extraction_max_points=100000,
                      extraction_device='cpu',target_voxel_m=DEFAULT_NKSR_TARGET_VOXEL_M,
                      normal_knn=settings.normal_knn,normal_drop_angle_deg=settings.normal_drop_angle_deg,
                      mise_iter=settings.mise_iter,detail_level=None,requested_detail_level=settings.detail_level,
                      input_path=str(settings.input),mesh_bbox=[stats['bounding_box_min'],stats['bounding_box_max']],
                      validation_status='WARNING',boundary_stitching=False,
                      validation_note='Independent tiles: boundaries are not stitched; gaps or overlaps may remain.',
                      checkpoint_loaded=True,sensor_origins_used=True,elapsed_seconds=time.monotonic()-started)
        atomic_json(settings.metadata or output.parent/'nksr_metadata.json',metadata)
        event('COMPLETED',vertices=stats['vertex_count'],faces=stats['face_count'],tile_count=len(tiles))
    return 0