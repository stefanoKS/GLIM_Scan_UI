#!/usr/bin/env python3
"""VDBFusion benchmark CLI: staged measurements with no invented numbers.

Run with the isolated VDBFusion interpreter (``scripts/benchmark_vdbfusion.sh``
selects it). Stages deliberately grow, so a whole-factory session is only reached
after the smaller stages and their resource preflights have passed.

    A  synthetic   small generated cloud and trajectory, no external data
    B  subset      the real bag restricted to a point-count budget
    C  roi         one representative 10 m x 10 m factory region
    D  full        the whole recording, only with --allow-full and passing checks

Every reported figure is measured in this process: phase timings, peak RSS, mesh
counts and output size. Nothing is estimated or extrapolated.
"""
import argparse
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ui/backend'))
from factory_mapping import vdbfusion as V  # noqa: E402


def peak_rss_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024


def synthetic_scene(root, frames=6, points_per_frame=20_000, voxel=0.02):
    """Stage A input: a moving sensor looking at a synthetic factory-like room."""
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import PointCloud2, PointField
    rng = np.random.default_rng(20261008)
    stamps = np.linspace(0.0, 10.0, frames)
    speeds = 0.6
    trajectory = np.column_stack((stamps, speeds*stamps, np.zeros(frames), np.zeros(frames),
                                  np.zeros(frames), np.zeros(frames), np.zeros(frames), np.ones(frames)))
    bag = root/'raw_bag'
    root.mkdir(parents=True, exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    writer.create_topic(rosbag2_py.TopicMetadata(name='/livox/lidar', type='sensor_msgs/msg/PointCloud2',
                                                 serialization_format='cdr'))
    count = int(points_per_frame)
    cloud = np.zeros(count, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                   ('intensity', '<f4'), ('timestamp', '<f8')])
    for stamp, position in zip(stamps, trajectory[:, 1:4]):
        local = np.column_stack((rng.uniform(-6.0, 6.0, count),
                                 rng.uniform(-6.0, 6.0, count),
                                 rng.uniform(-1.0, 3.0, count)))
        cloud['x'], cloud['y'], cloud['z'] = local[:, 0]+2.0, local[:, 1], local[:, 2]
        cloud['intensity'] = 1.0
        cloud['timestamp'] = stamp*1e9
        message = PointCloud2()
        message.header.frame_id = 'livox_frame'
        message.height = 1
        message.width = count
        message.point_step = cloud.dtype.itemsize
        message.row_step = message.point_step*count
        message.fields = [PointField(name=name, offset=int(cloud.dtype.fields[name][1]),
                                     datatype=PointField.FLOAT32, count=1)
                          for name in ('x', 'y', 'z', 'intensity')]
        message.fields.append(PointField(name='timestamp', offset=int(cloud.dtype.fields['timestamp'][1]),
                                        datatype=PointField.FLOAT64, count=1))
        message.data = cloud.tobytes()
        writer.write('/livox/lidar', serialize_message(message), int(stamp*1e9))
    del writer
    trajectory_path = root/'traj_lidar.txt'
    np.savetxt(trajectory_path, trajectory)
    return bag, trajectory_path


def subset_trajectory(trajectory, maximum_seconds):
    """Restrict a trajectory to its first ``maximum_seconds`` so a subset stage is real."""
    if maximum_seconds is None:
        return trajectory, False
    stamps = trajectory[:, 0]
    keep = stamps <= stamps[0] + maximum_seconds
    if keep.sum() < 2:
        return trajectory, False
    return trajectory[keep], True


def run_stage(options, bag, trajectory_path, label, output, settings, edit_workspace=None):
    """Run one measured stage and return its result record."""
    report = dict(stage=label, bag=str(bag), trajectory=str(trajectory_path),
                  settings={key: settings[key] for key in
                            ('voxel_size_m', 'sdf_trunc_m', 'space_carving', 'origin_error_budget_m',
                             'mesh_output_mode', 'batch_points', 'roi_min_m', 'roi_max_m')})
    report['started_at'] = time.monotonic()
    started = time.monotonic()
    trajectory, trajectory_report = V.load_trajectory(trajectory_path)
    report['source_validation_seconds'] = time.monotonic() - started
    report['trajectory'] = dict(poses=trajectory_report['poses'], duration_s=trajectory_report['duration_s'],
                                path_length_m=trajectory_report['path_length_m'],
                                quaternion_normalized=trajectory_report['quaternion_normalized'])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reuse the worker's own evenly spread bag sampler so the benchmark and the engine
    # agree on how a scan is summarised.
    from factory_mapping.vdbfusion_worker import sample_bag_extent
    report['source'] = sample_bag_extent(bag, trajectory, options.topic)[0]
    edit_reference = None
    if edit_workspace is not None:
        started = time.monotonic()
        edit_reference = V.load_edited_reference(edit_workspace)
        report['edit_filter_setup_seconds'] = time.monotonic() - started
        report['edit_reference'] = dict(edit_reference.metadata)
    report['preflight'] = V.preflight(output.parent, settings,
                                      observed_points=report['source']['estimated_observations'],
                                      observed_bbox=(report['source']['observed_bbox_min_m'],
                                                     report['source']['observed_bbox_max_m']))
    started_integration = time.monotonic()
    volume, stats = V.integrate_bag(bag, trajectory, options.topic, settings, output=output,
                                    edit_reference=edit_reference, memory_budget_bytes=None,
                                    peak_limit_bytes=options.memory_budget_bytes)
    report['integration'] = {key: value for key, value in stats.items() if not isinstance(value, dict)}
    report['timings'] = stats['timings']
    started_extract = time.monotonic()
    vertices, faces, extraction = V.extract_and_mask(
        volume, edit_reference, association_radius_m=(edit_reference.resolve_association_radius()
                                                      if edit_reference else None))
    del volume
    report['extraction'] = extraction
    report['extraction']['elapsed_seconds'] = time.monotonic() - started_extract
    started_validate = time.monotonic()
    mesh = V.validate_and_report(vertices, faces, settings)
    stats_written = V.write_mesh_ply(output, vertices, faces)
    persisted = V.validate_written_mesh(output, stats_written)
    report['validation'] = dict(elapsed_seconds=time.monotonic() - started_validate,
                                vertex_count=mesh['vertex_count'], face_count=mesh['face_count'],
                                bounding_box_min=mesh['bounding_box_min'], bounding_box_max=mesh['bounding_box_max'],
                                persisted=persisted)
    report['output'] = dict(path=str(output), bytes=output.stat().st_size)
    report['elapsed_seconds'] = time.monotonic() - report['started_at']
    report['peak_rss_bytes'] = peak_rss_bytes()
    return report


def print_report(report):
    print(f"\n=== stage {report['stage']} ===")
    print(f"  bag                 {report['bag']}")
    print(f"  trajectory          {report['trajectory']['poses']} poses, "
          f"{report['trajectory']['duration_s']:.1f} s, {report['trajectory']['path_length_m']:.1f} m")
    print(f"  bag estimate        {report['source']['estimated_observations']:,} observations over "
          f"{report['source']['lidar_frames'] or 0:,} frames "
          f"({report['source']['observations_per_frame']:.0f}/frame, sampled)")
    print(f"  sampled extent      {[round(v, 1) for v in report['source']['observed_bbox_min_m']]} -> "
          f"{[round(v, 1) for v in report['source']['observed_bbox_max_m']]} m")
    settings = report['settings']
    print(f"  TSDF                {settings['voxel_size_m']*1000:.1f} mm voxel, "
          f"{settings['sdf_trunc_m']*1000:.1f} mm truncation, space carving {settings['space_carving']}")
    if report['settings']['roi_min_m']:
        print(f"  ROI                 {report['settings']['roi_min_m']} -> {report['settings']['roi_max_m']}")
    for warning in report['preflight'].get('warnings', []):
        print(f"  WARNING             {warning}")
    if 'integration' in report:
        integration = report['integration']
        print(f"  source validation   {report.get('source_validation_seconds', 0):8.2f} s")
        if 'edit_filter_setup_seconds' in report:
            print(f"  edit reference      {report['edit_filter_setup_seconds']:8.2f} s")
        timings = report['timings']
        print(f"  bag read            {timings.get('bag_read_seconds', 0):8.2f} s")
        print(f"  cloud decode        {timings.get('decode_seconds', 0):8.2f} s")
        print(f"  trajectory interp.  {timings.get('transform_seconds', 0):8.2f} s")
        print(f"  edit filtering      {timings.get('filter_seconds', 0):8.2f} s")
        print(f"  origin grouping     {timings.get('group_seconds', 0):8.2f} s")
        print(f"  TSDF integration    {timings.get('integrate_seconds', 0):8.2f} s "
              f"({integration['origin_groups']:,} origin groups)")
        print(f"  mesh extraction     {report['extraction']['elapsed_seconds']:8.2f} s "
              f"({report['extraction']['masked_triangles']:,} masked triangles)")
        print(f"  mesh validation     {report['validation']['elapsed_seconds']:8.2f} s")
        print(f"  total elapsed       {report['elapsed_seconds']:8.2f} s")
        print(f"  peak RSS            {report['peak_rss_bytes']/1048576:.0f} MiB")
        print(f"  raw observations    {integration['raw_observations']:,} "
              f"(outside trajectory range {integration['outside_trajectory']:,}, "
              f"outside ROI {integration['points_dropped_outside_roi']:,})")
        print(f"  integrated          {integration['integrated_observations']:,} "
              f"· max origin error {integration['max_origin_error_m']*1000:.2f} mm "
              f"of {integration['origin_error_budget_m']*1000:.2f} mm budget")
        if integration['edit_filter_enabled']:
            print(f"  edited mode         points before filter {integration['points_before_filter']:,}")
            print(f"                      retained {integration['points_retained']:,} "
                  f"({integration['filter_retention_ratio']*100:.2f}%)")
            print(f"                      removed-support dominated "
                  f"{integration['points_dropped_removed_support']:,}")
            print(f"                      unsupported {integration['points_dropped_unsupported']:,} "
                  f"({integration['unsupported_observations']})")
            print(f"                      association radius {integration['association_radius_m']:.4f} m "
                  f"from measured sampling {report['edit_reference']['sampling_resolution_m']:.4f} m")
            print(f"                      filtering mode {report['edit_reference']['accuracy']}")
        print(f"  mesh                {report['validation']['vertex_count']:,} vertices, "
              f"{report['validation']['face_count']:,} triangles")
        print(f"  mesh bounds         {[round(v, 2) for v in report['validation']['bounding_box_min']]} -> "
              f"{[round(v, 2) for v in report['validation']['bounding_box_max']]} m")
        print(f"  output size         {report['output']['bytes']/1048576:.1f} MiB on disk")
    else:
        print(f"  preflight RAM/disk  {report['preflight']['free_ram_bytes']/1073741824:.1f} GiB free RAM, "
              f"{report['preflight']['free_disk_bytes']/1073741824:.1f} GiB free disk")
        print(f"  estimated TSDF      {report['preflight']['estimated_tsdf_bytes']/1073741824:.2f} GiB "
              f"(band upper bound {report['preflight']['band_voxels_upper_bound']:,.0f} voxels)")
        print(f"  elapsed             {report['elapsed_seconds']:.2f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=['A', 'B', 'C', 'D'], required=True,
                        help='A synthetic, B subset, C region of interest, D full recording')
    parser.add_argument('--bag', type=Path)
    parser.add_argument('--trajectory', type=Path)
    parser.add_argument('--topic', default='/livox/lidar')
    parser.add_argument('--voxel-size', type=float, default=0.02)
    parser.add_argument('--sdf-trunc', type=float, default=0.06)
    parser.add_argument('--space-carving', choices=['0', '1'], default='0')
    parser.add_argument('--batch-points', type=int)
    parser.add_argument('--subset-seconds', type=float, default=20.0,
                        help='Stage B: seconds of trajectory kept, so the subset is a real prefix')
    parser.add_argument('--roi-min', type=float, nargs=3)
    parser.add_argument('--roi-max', type=float, nargs=3)
    parser.add_argument('--edited-workspace', type=Path,
                        help='Verified map-editor cleanup workspace for edited-mode measurements')
    parser.add_argument('--mesh-output-mode', choices=list(V.MESH_OUTPUT_MODES), default='merged')
    parser.add_argument('--memory-budget-gib', type=float)
    parser.add_argument('--allow-full', action='store_true', help='Stage D requires this and passing checks')
    parser.add_argument('--presets', action='store_true', help='Also measure the FAST/DETAILED/EXPERIMENTAL presets')
    parser.add_argument('--output-dir', type=Path, help='Where meshes are written (default: a temp dir)')
    parser.add_argument('--json', type=Path, help='Also write the full measurements as JSON')
    options = parser.parse_args()
    options.memory_budget_bytes = int(options.memory_budget_gib*1024**3) if options.memory_budget_gib else None

    scratch = None
    if options.output_dir:
        workspace = options.output_dir
        workspace.mkdir(parents=True, exist_ok=True)
    else:
        scratch = tempfile.mkdtemp(prefix='vdbfusion-benchmark-')
        workspace = Path(scratch)
    reports = []
    try:
        if options.stage == 'A':
            root = workspace/'synthetic'
            bag, trajectory = synthetic_scene(root)
        else:
            if not (options.bag and options.trajectory):
                raise SystemExit(f'Stage {options.stage} needs --bag and --trajectory')
            bag, trajectory = options.bag, options.trajectory
        if options.stage == 'B':
            trajectory = workspace/'subset_traj_lidar.txt'
            values, restricted = subset_trajectory(np.loadtxt(options.trajectory, ndmin=2),
                                                   options.subset_seconds)
            if not restricted:
                raise SystemExit('The chosen --subset-seconds covers fewer than two poses')
            np.savetxt(trajectory, values)
        if options.stage == 'D':
            if not options.allow_full:
                raise SystemExit('Stage D processes the whole recording: pass --allow-full after reviewing the '
                                 'smaller stages and their resource preflight')
            if options.voxel_size < 0.005:
                raise SystemExit('Stage D refuses a voxel size below 5 mm; review the preflight first')
        settings = V.resolve_settings(dict(voxel_size_m=options.voxel_size, sdf_trunc_m=options.sdf_trunc,
                                          space_carving=options.space_carving == '1',
                                          batch_points=options.batch_points,
                                          mesh_output_mode=options.mesh_output_mode,
                                          roi_min_m=list(options.roi_min) if options.roi_min else None,
                                          roi_max_m=list(options.roi_max) if options.roi_max else None))
        if options.stage == 'C' and not options.roi_min:
            raise SystemExit('Stage C needs --roi-min and --roi-max for the representative region')
        output = workspace/options.stage/'mesh.ply'
        report = run_stage(options, bag, trajectory, options.stage, output, settings,
                           edit_workspace=options.edited_workspace)
        reports.append(report)
        print_report(report)
        if options.presets:
            for preset in ('fast', 'detailed', 'experimental'):
                preset_settings = V.resolve_settings(dict(preset=preset, mesh_output_mode=options.mesh_output_mode,
                                                          roi_min_m=settings['roi_min_m'],
                                                          roi_max_m=settings['roi_max_m']))
                preset_output = workspace/f'{options.stage}_{preset}'/'mesh.ply'
                preset_report = run_stage(options, bag, trajectory, f'{options.stage}-{preset}', preset_output,
                                          preset_settings, edit_workspace=options.edited_workspace)
                reports.append(preset_report)
                print_report(preset_report)
        if options.json:
            options.json.parent.mkdir(parents=True, exist_ok=True)
            options.json.write_text(json.dumps(dict(schema=1, python=sys.executable,
                                                    interpreter=sys.executable, reports=reports), indent=2,
                                               default=str))
        print('\nAll figures above were measured by this run; nothing here is an estimate or a comparison '
              'against a different tool unless both were measured on the same source and trajectory.')
        return 0
    finally:
        if scratch and os.environ.get('KEEP_BENCHMARK_OUTPUT') != '1':
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
