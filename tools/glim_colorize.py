#!/usr/bin/env python3
"""Standalone camera→LiDAR→colored point cloud pipeline.

Produces the master colored point cloud (PLY + NPZ + stats + validation overlays)
and optional color transfer onto the exact GLIM PLY and the NKSR mesh. Never
overwrites recorded, prepared or exported inputs.

Examples
--------
python3 tools/glim_colorize.py --session /path/to/session --validation-frames 10
python3 tools/glim_colorize.py --session /path/to/session --transfer-glim --glim-ply auto
python3 tools/glim_colorize.py --session /path/to/session --transfer-nksr --nksr-mesh auto
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ui' / 'backend'))

from factory_mapping.colorization import colorize_session  # noqa: E402
from factory_mapping.color_transfer import transfer_colors_to_mesh, transfer_colors_to_points  # noqa: E402
from factory_mapping.storage import atomic_json  # noqa: E402


def _resolve_glim_ply(session):
    exports = sorted((session / 'exports').glob('*.ply'))
    official = [p for p in exports if not p.name.startswith('edit_')]
    chosen = official[-1] if official else (exports[-1] if exports else None)
    if chosen is None:
        raise ValueError('No GLIM PLY export found; pass --glim-ply explicitly')
    return chosen


def _resolve_nksr_mesh(session):
    meshes = sorted((session / 'reconstruction').glob('run_*/output/mesh.ply'))
    if not meshes:
        raise ValueError('No NKSR mesh found; pass --nksr-mesh explicitly')
    return meshes[-1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', required=True, type=Path,
                        help='Session directory under data/sessions/')
    parser.add_argument('--trajectory', type=Path, help='Final GLIM traj_lidar.txt (default: auto)')
    parser.add_argument('--run', help='Reconstruction run ID used for trajectory resolution')
    parser.add_argument('--image-topic', help='Camera Image topic (default: session camera)')
    parser.add_argument('--lidar-topic', help='LiDAR PointCloud2 topic (default: session sensor)')
    parser.add_argument('--voxel-size', type=float, default=0.01,
                        help='World-space voxel size in METERS (default: 0.01)')
    parser.add_argument('--max-time-delta', type=float, default=0.15,
                        help='Temporal association window in seconds (default: 0.15)')
    parser.add_argument('--min-depth', type=float, default=0.0,
                        help='Minimum camera depth in meters (default: 0.0)')
    parser.add_argument('--max-depth', type=float, default=20.0,
                        help='Maximum camera depth in meters (default: 20.0)')
    parser.add_argument('--occlusion-base-tolerance', type=float, default=0.03,
                        help='Occlusion base tolerance in meters (default: 0.03)')
    parser.add_argument('--occlusion-range-scale', type=float, default=0.0075,
                        help='Occlusion depth-proportional tolerance (default: 0.0075)')
    parser.add_argument('--validation-frames', type=int, default=0,
                        help='Number of depth-colored projection overlay JPEGs to generate')
    parser.add_argument('--chunk-points', type=int, default=1_000_000,
                        help='Points processed per projection block (default: 1000000)')
    parser.add_argument('--allow-unvalidated-calibration', action='store_true',
                        help='Allow a calibrated but unvalidated extrinsic')
    parser.add_argument('--transfer-glim', action='store_true',
                        help='Transfer color onto the official GLIM PLY')
    parser.add_argument('--glim-ply', type=Path, help='GLIM PLY path, or "auto"')
    parser.add_argument('--transfer-radius', type=float, default=0.025,
                        help='GLIM transfer search radius in meters (default: 0.025)')
    parser.add_argument('--transfer-k', type=int, default=5,
                        help='GLIM transfer neighbor count (default: 5)')
    parser.add_argument('--transfer-nksr', action='store_true',
                        help='Transfer color onto the NKSR mesh vertices')
    parser.add_argument('--nksr-mesh', type=Path, help='NKSR mesh PLY path, or "auto"')
    parser.add_argument('--surface-transfer-radius', type=float, default=0.025,
                        help='NKSR transfer search radius in meters (default: 0.025)')
    parser.add_argument('--surface-transfer-k', type=int, default=5,
                        help='NKSR transfer neighbor count (default: 5)')
    parser.add_argument('--output-dir', type=Path,
                        help='Explicit output run directory (managed jobs); default: fresh run_*')
    parser.add_argument('--progress-json', type=Path,
                        help='Write a progress message file (managed jobs)')
    args = parser.parse_args(argv)

    def progress(message):
        print(message, flush=True)
        if args.progress_json:
            atomic_json(args.progress_json, {'message': message})

    try:
        session = args.session.resolve()
        output = colorize_session(
            session, trajectory=args.trajectory, run=args.run,
            image_topic=args.image_topic, lidar_topic=args.lidar_topic,
            voxel_size=args.voxel_size, max_time_delta=args.max_time_delta,
            min_depth=args.min_depth, max_depth=args.max_depth,
            occlusion_base_tolerance=args.occlusion_base_tolerance,
            occlusion_range_scale=args.occlusion_range_scale,
            validation_frames=args.validation_frames,
            chunk_points=args.chunk_points,
            allow_unvalidated_calibration=args.allow_unvalidated_calibration,
            progress=progress, output_dir=args.output_dir)

        npz = np_load(output / 'output' / 'colored_points.npz')

        if args.transfer_glim:
            glim_ply = args.glim_ply
            if glim_ply is None or str(glim_ply) == 'auto':
                glim_ply = _resolve_glim_ply(session)
            stats = transfer_colors_to_points(
                glim_ply, npz['points'], npz['Cd'], npz['color_confidence'],
                output / 'output' / 'glim_colored.ply',
                radius=args.transfer_radius, k=args.transfer_k)
            atomic_json(output / 'output' / 'glim_transfer_stats.json', stats)
            print(json.dumps(stats, indent=2))

        if args.transfer_nksr:
            nksr_mesh = args.nksr_mesh
            if nksr_mesh is None or str(nksr_mesh) == 'auto':
                nksr_mesh = _resolve_nksr_mesh(session)
            stats = transfer_colors_to_mesh(
                nksr_mesh, npz['points'], npz['Cd'], npz['color_confidence'],
                output / 'output' / 'nksr_colored.ply',
                radius=args.surface_transfer_radius, k=args.surface_transfer_k)
            atomic_json(output / 'output' / 'nksr_transfer_stats.json', stats)
            print(json.dumps(stats, indent=2))

        print(output)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.exit(1, f'Colorization failed: {error}\n')


def np_load(path):
    import numpy as np
    return np.load(path, allow_pickle=False)


if __name__ == '__main__':
    main()
