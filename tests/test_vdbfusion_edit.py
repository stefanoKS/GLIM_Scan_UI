"""GATE 7: edit-aware selection tests on controlled synthetic geometry.

The scene is a dense planar wall with a deliberately removed rectangular patch, a
thin retained strip next to that patch, and a second object joined by a removable
connector. GLIM's submap format is reproduced exactly: ``map_01`` holds the
pre-edit lattice samples and ``saved_map`` holds the retained subset, so the
kept/removed classification is known ground truth rather than an assumption.

Acceptance targets are checked on spatial occupancy at the TSDF resolution, not on
point counts alone. The documented boundary uncertainty band is the reported
association radius; measurements exclude it on both sides.
"""
import json
import subprocess
import numpy as np
import pytest
from plyfile import PlyData

from factory_mapping import vdbfusion as V

WALL = (-4.0, 4.0, 0.0, 2.5)
# A 2.0 x 1.4 m hole in the wall, with a 25 cm retained strip touching its right edge.
PATCH = (-1.5, 0.5, 0.4, 1.8)
STRIP = (0.55, 0.8, 0.4, 1.8)
OBJECT = (5.0, 5.5, 0.0, 2.5)
CONNECTOR = (4.0, 5.0, 0.0, 2.5)
PLANE_Y = 0.0


def region_depth(points, extent):
    """Perpendicular distance from each inside point to the nearest region edge."""
    x0, x1, z0, z1 = extent
    edges = np.minimum(np.minimum(points[:, 0]-x0, x1-points[:, 0]),
                       np.minimum(points[:, 2]-z0, z1-points[:, 2]))
    return np.where(in_region(points, extent), edges, -1.0)


def region_points(extent, spacing):
    """Dense points on the y = 0 plane for one (x0, x1, z0, z1) region."""
    x0, x1, z0, z1 = extent
    xs = np.arange(x0, x1 + spacing/2, spacing)
    zs = np.arange(z0, z1 + spacing/2, spacing)
    grid = np.stack(np.meshgrid(xs, zs, indexing='ij'), axis=-1).reshape(-1, 2)
    return np.column_stack((grid[:, 0], np.full(len(grid), PLANE_Y), grid[:, 1]))


def reference_samples(extent, spacing, jitter, rng):
    """GLIM-style voxel samples: one jittered point per ``spacing`` lattice cell."""
    points = region_points(extent, spacing)
    if jitter:
        points = points + rng.uniform(-jitter, jitter, points.shape)
    return points.astype(np.float32)


def in_region(points, extent):
    x0, x1, z0, z1 = extent
    return ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
            (points[:, 2] >= z0) & (points[:, 2] <= z1))


def write_submap(directory, points):
    directory.mkdir(parents=True, exist_ok=True)
    points.astype('<f4').reshape(-1).tofile(directory/'points_compact.bin')
    # GLIM's data.txt carries T_world_origin; the identity pose keeps world == local here.
    matrix = np.eye(4)
    rows = '\n'.join(' '.join(f'{value:.9g}' for value in row) for row in matrix)
    (directory/'data.txt').write_text(f'id: 0\nT_world_origin: \n{rows}\n')


def build_cleanup(root, spacing=0.1, jitter=0.02, raw_spacing=0.02, seed=5):
    """Build a synthetic pre-edit map, a saved cleanup and the dense raw cloud."""
    rng = np.random.default_rng(seed)
    regions = [WALL, STRIP, OBJECT, CONNECTOR]
    pre_edit = np.concatenate([reference_samples(extent, spacing, jitter, rng) for extent in regions])
    removed = in_region(pre_edit, PATCH) | in_region(pre_edit, CONNECTOR)
    if not removed.any() or removed.all():
        raise AssertionError('the synthetic edit must remove a real subset')
    raw = np.concatenate([region_points(extent, raw_spacing).astype(np.float64) for extent in regions])

    workspace = root/'edits'/'edit_0123456789ab'
    write_submap(workspace/'map_01'/'000000', pre_edit)
    write_submap(workspace/'saved_map'/'000000', pre_edit[~removed])
    (workspace/'saved_map'/'traj_lidar.txt').write_text('\n'.join(
        ['0.0 0 0 0 0 0 0 1', '10.0 0 0 0 0 0 0 1'])+'\n')
    (workspace/'map_01'/'traj_lidar.txt').write_text((workspace/'saved_map'/'traj_lidar.txt').read_text())
    (workspace/'workspace.json').write_text(json.dumps(dict(
        id='edit_0123456789ab', tool='map_editor', state='closed', pose_policy='map_editor_fixed_poses',
        sources=[dict(session='synthetic', run='run_001', dump='dump', config='config')])))
    return dict(workspace=workspace, raw=raw,
                kept_mask=~removed, removed_mask=removed, spacing=spacing, regions=regions)


def write_scene_bag(root, raw, frames=2, sensor_y=-2.0, seed=3):
    """Write a real ROS 2 bag whose LiDAR frames observe the synthetic world.

    The sensor sits ``abs(sensor_y)`` metres in front of the wall plane so every
    observation has a real ray length; the trajectory is a static pose, which is a
    legitimate trajectory origin rather than a fabricated [0, 0, 0] assumption.
    """
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import PointCloud2, PointField
    path = root/'raw_bag'
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    writer.create_topic(rosbag2_py.TopicMetadata(name='/livox/lidar', type='sensor_msgs/msg/PointCloud2',
                                                 serialization_format='cdr'))
    # Identity rotation, so sensor coordinates are world coordinates shifted by the pose.
    local = raw - np.array([0.0, sensor_y, 0.0], dtype=np.float64)
    cloud = np.zeros(len(raw), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                      ('intensity', '<f4'), ('timestamp', '<f8')])
    cloud['x'], cloud['y'], cloud['z'] = local[:, 0], local[:, 1], local[:, 2]
    cloud['intensity'] = 1.0
    for stamp in (0.0, 10.0)[:frames]:
        cloud['timestamp'] = stamp * 1e9
        message = PointCloud2()
        message.header.frame_id = 'livox_frame'
        message.height = 1
        message.width = len(cloud)
        message.point_step = cloud.dtype.itemsize
        message.row_step = message.point_step*len(cloud)
        message.fields = [PointField(name=name, offset=int(cloud.dtype.fields[name][1]),
                                     datatype=PointField.FLOAT32, count=1)
                          for name in ('x', 'y', 'z', 'intensity')]
        message.fields.append(PointField(name='timestamp', offset=int(cloud.dtype.fields['timestamp'][1]),
                                        datatype=PointField.FLOAT64, count=1))
        message.data = cloud.tobytes()
        writer.write('/livox/lidar', serialize_message(message), int(stamp*1e9))
    del writer
    traj = root/'traj_lidar.txt'
    np.savetxt(traj, np.array([[0.0, 0, sensor_y, 0, 0, 0, 0, 1],
                               [10.0, 0, sensor_y, 0, 0, 0, 0, 1]], dtype=np.float64))
    return path, traj


def run_worker(tmp_path, vdbfusion_python, bag, traj, workspace, settings, mode='merged'):
    from factory_mapping import vdbfusion_jobs
    output = tmp_path/'output'/'mesh.ply'
    arguments = [str(vdbfusion_python), str(vdbfusion_jobs.worker_path()),
                 '--bag', str(bag), '--trajectory', str(traj), '--output', str(output),
                 '--metadata', str(output.parent/'vdbfusion_metadata.json'),
                 '--progress', str(tmp_path/'vdbfusion_progress.json'),
                 '--voxel-size', str(settings['voxel_size_m']),
                 '--sdf-trunc', str(settings['sdf_trunc_m']),
                 '--mesh-output-mode', mode]
    if workspace is not None:
        arguments += ['--edited-workspace', str(workspace)]
    for key, flag in (('association_spacing_multiplier', '--association-spacing-multiplier'),
                      ('boundary_margin_m', '--boundary-margin')):
        if settings.get(key) is not None:
            arguments += [flag, str(settings[key])]
    if settings.get('mask_deleted_triangles') is False:
        arguments += ['--mask-deleted-triangles', '0']
    if settings.get('unsupported_observations'):
        arguments += ['--unsupported-policy', settings['unsupported_observations']]
    result = subprocess.run(arguments, env=vdbfusion_jobs.worker_environment(),
                            capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout + result.stderr
    return output, json.loads((output.parent/'vdbfusion_metadata.json').read_text())


def read_triangles(path):
    mesh = PlyData.read(str(path), known_list_len={'face': {'vertex_indices': 3}})
    vertices = np.column_stack([mesh['vertex'][axis] for axis in 'xyz']).astype(np.float64)
    faces = np.asarray(mesh['face']['vertex_indices'])
    if faces.dtype.kind == 'O':
        faces = np.stack(faces)
    return vertices, faces.astype(np.int64)


def plane_occupancy(centroids, extent, size, exclude=(), band=0.0):
    """Spatial occupancy of one coplanar region at the TSDF resolution.

    Cells whose centre lies inside a removed region, or within ``band`` of one, are
    omitted from the measurement: that band is the documented edit-boundary
    uncertainty, described by the reported association radius. Returns
    ``(occupancy, measured_cells)``.
    """
    x0, x1, z0, z1 = extent
    xs = np.arange(x0 + size/2.0, x1, size)
    zs = np.arange(z0 + size/2.0, z1, size)
    if not len(xs) or not len(zs):
        return 1.0, 0
    grid_x, grid_z = np.meshgrid(xs, zs, indexing='ij')
    centres = np.column_stack((grid_x.ravel(), np.full(grid_x.size, PLANE_Y), grid_z.ravel()))
    measured = np.ones(len(centres), dtype=bool)
    for region in exclude:
        rx0, rx1, rz0, rz1 = region
        dx = np.maximum(np.maximum(rx0-centres[:, 0], centres[:, 0]-rx1), 0.0)
        dz = np.maximum(np.maximum(rz0-centres[:, 2], centres[:, 2]-rz1), 0.0)
        distance = np.hypot(dx, dz)
        measured &= (distance > band) if band else ~in_region(centres, region)
    if not measured.any():
        return 1.0, 0
    nx, nz = len(xs), len(zs)
    columns = np.floor((centroids[:, 0]-x0)/size).astype(np.int64)
    rows = np.floor((centroids[:, 2]-z0)/size).astype(np.int64)
    inside = ((centroids[:, 0] >= x0) & (centroids[:, 0] < x1) &
              (centroids[:, 2] >= z0) & (centroids[:, 2] < z1) &
              (columns >= 0) & (columns < nx) & (rows >= 0) & (rows < nz))
    hit = np.zeros(nx*nz, dtype=bool)
    if inside.any():
        hit[(rows[inside]*nx + columns[inside])] = True
    cells = (np.floor((centres[:, 2]-z0)/size).astype(np.int64)*nx +
             np.floor((centres[:, 0]-x0)/size).astype(np.int64))[measured]
    cells = cells[(cells >= 0) & (cells < nx*nz)]
    return float(hit[cells].mean()), int(measured.sum())


def centroids_in_region(vertices, faces, extent, inset=0.0):
    centroids = vertices[faces].mean(axis=1)
    x0, x1, z0, z1 = extent
    if inset:
        x0, x1, z0, z1 = x0+inset, x1-inset, z0+inset, z1-inset
    return int(np.count_nonzero((centroids[:, 0] > x0) & (centroids[:, 0] < x1) &
                                (centroids[:, 2] > z0) & (centroids[:, 2] < z1)))


# --------------------------------------------------------------------------- #
# Reference classification without the native library
# --------------------------------------------------------------------------- #

def _near_region(points, extent, distance):
    """True where a point is within ``distance`` of a region, inside or outside it."""
    x0, x1, z0, z1 = extent
    dx = np.maximum(np.maximum(x0-points[:, 0], points[:, 0]-x1), 0.0)
    dz = np.maximum(np.maximum(z0-points[:, 2], points[:, 2]-z1), 0.0)
    return np.hypot(dx, dz) <= distance


def retained_wall_mask(raw):
    """Dense wall observations that belong to geometry the user kept."""
    return in_region(raw, WALL) & ~in_region(raw, PATCH) & ~in_region(raw, CONNECTOR)


def test_edit_classification_excludes_removed_regions_and_covers_retained_surfaces(tmp_path):
    scene = build_cleanup(tmp_path)
    reference = V.load_edited_reference(scene['workspace'])
    metadata = reference.metadata
    assert metadata['removed_points'] > 0 and metadata['retained_points'] > 0
    assert metadata['accuracy'] == 'validated_approximate'
    assert 'cannot be traced back to individual raw observations' in metadata['limitation']
    radius = reference.resolve_association_radius()
    assert radius == pytest.approx(4.0*metadata['sampling_resolution_m'], rel=1e-6)
    band = metadata['sampling_resolution_m']

    raw = scene['raw']
    classified = reference.classify(raw, radius)
    # B. Removed-region exclusion, measured outside the documented uncertainty band. The
    # band used here is one reference sampling step, a quarter of the reported radius.
    for extent in (PATCH, CONNECTOR):
        core = region_depth(raw, extent) > band
        assert core.sum() > 1000
        exclusion = 1.0 - classified['keep'][core].mean()
        assert exclusion >= 0.98, f'removed-region exclusion {exclusion:.4f} in {extent}'
    # A. Intended retained-surface coverage, including the thin strip beside the hole.
    for name, extent in (('strip', STRIP), ('object', OBJECT)):
        selection = in_region(raw, extent)
        assert selection.sum() > 100
        assert classified['keep'][selection].mean() >= 0.95, (name, extent)
    wall = retained_wall_mask(raw)
    assert classified['keep'][wall & ~_near_region(raw, PATCH, radius)].mean() >= 0.99


def test_documented_uncertainty_band_is_reported_not_hidden(tmp_path):
    """Residual retention only occurs inside one reference-sampling-wide boundary band."""
    scene = build_cleanup(tmp_path)
    reference = V.load_edited_reference(scene['workspace'])
    band = reference.metadata['sampling_resolution_m']
    radius = reference.resolve_association_radius()
    raw = scene['raw']
    classified = reference.classify(raw, radius)
    for extent in (PATCH, CONNECTOR):
        inside = in_region(raw, extent)
        core = inside & (region_depth(raw, extent) > band)
        edge = inside & (region_depth(raw, extent) <= band)
        assert core.sum() > 0 and edge.sum() > 0
        assert 1.0 - classified['keep'][core].mean() == 1.0, 'no removed observation outside the band is retained'
        assert float(classified['keep'][edge].mean()) < 0.5, 'the uncertain band must stay narrow'
    # The limitation names exactly what the band is, so the UI can surface it.
    assert 'uncertainty band' in reference.metadata['limitation']
    assert reference.metadata['association_radius_m'] == pytest.approx(radius)


def test_a_fixed_five_centimetre_tolerance_leaves_holes_where_the_measured_radius_does_not(tmp_path):
    """GATE 7A: the legacy fixed tolerance is below GLIM's own submap sampling."""
    scene = build_cleanup(tmp_path)
    reference = V.load_edited_reference(scene['workspace'])
    raw = scene['raw']
    wall = retained_wall_mask(raw)
    legacy = reference.classify(raw, 0.05)['keep'][wall].mean()
    measured = reference.classify(raw, reference.resolve_association_radius())['keep'][wall].mean()
    assert legacy < 0.75, f'the legacy 5 cm filter should lose retained surface, kept {legacy:.3f}'
    assert measured >= 0.95
    assert measured - legacy > 0.2, 'the measured radius must recover a large share of the wall'
    # The report names the derived radius and its source instead of a magic tolerance.
    assert reference.metadata['method'].startswith('nearest_kept_within_measured_sampling_radius')


def test_edit_sampling_density_changes_the_radius_not_the_decision(tmp_path):
    results = {}
    for name, spacing in (('dense', 0.05), ('sparse', 0.2)):
        root = tmp_path/name
        root.mkdir()
        scene = build_cleanup(root, spacing=spacing)
        reference = V.load_edited_reference(scene['workspace'])
        radius = reference.resolve_association_radius()
        classified = reference.classify(scene['raw'], radius)
        wall = retained_wall_mask(scene['raw'])
        strict = wall & ~_near_region(scene['raw'], PATCH, radius)
        core = region_depth(scene['raw'], PATCH) > reference.metadata['sampling_resolution_m']
        results[name] = dict(radius=radius, coverage=float(classified['keep'][wall].mean()),
                             strict_coverage=float(classified['keep'][strict].mean()),
                             core_exclusion=1.0-float(classified['keep'][core].mean()))
    assert results['sparse']['radius'] > results['dense']['radius']
    for name, metrics in results.items():
        assert metrics['strict_coverage'] >= 0.98, (name, metrics)
        assert metrics['coverage'] >= 0.95, (name, metrics)
        assert metrics['core_exclusion'] >= 0.98, (name, metrics)


def test_missing_or_stale_cleanup_is_reported_not_guessed(tmp_path):
    scene = build_cleanup(tmp_path)
    workspace = scene['workspace']
    # A missing saved submaps directory must not silently fall back to unfiltered geometry.
    (workspace/'saved_map'/'000000'/'points_compact.bin').unlink()
    with pytest.raises(ValueError):
        V.load_edited_reference(workspace)
    (workspace/'saved_map'/'000000'/'points_compact.bin').write_bytes(
        (workspace/'map_01'/'000000'/'points_compact.bin').read_bytes())
    # An unchanged saved map removes nothing, so filtering is unsupported and must be refused.
    reference = V.load_edited_reference(workspace)
    assert reference.metadata['removed_points'] == 0
    from factory_mapping.vdbfusion_worker import load_edit_reference, WorkerError  # noqa: N813
    with pytest.raises(WorkerError) as error:
        load_edit_reference(workspace, None, None)
    assert error.value.code == 'EDIT_REFERENCE_UNSUPPORTED'
    # A mismatched submap set is refused instead of being combined across edits.
    (workspace/'saved_map'/'000001').mkdir()
    (workspace/'saved_map'/'000001'/'points_compact.bin').write_bytes(b'')
    (workspace/'saved_map'/'000001'/'data.txt').write_text('id: 1\nT_world_origin: \n'+'\n'.join(
        [' '.join(['0']*4)]*4)+'\n')
    with pytest.raises(ValueError, match='do not match the pre-edit submaps'):
        V.load_edited_reference(workspace)


# --------------------------------------------------------------------------- #
# End-to-end native runs on the synthetic scene
# --------------------------------------------------------------------------- #

def test_edited_reconstruction_preserves_kept_geometry_and_excludes_removed_geometry(tmp_path, vdbfusion_python):
    scene = build_cleanup(tmp_path)
    bag, traj = write_scene_bag(tmp_path, scene['raw'])
    settings = dict(voxel_size_m=0.02, sdf_trunc_m=0.06)
    output, report = run_worker(tmp_path, vdbfusion_python, bag, traj, scene['workspace'], settings)
    integration = report['integration']
    assert integration['edit_filter_enabled'] and integration['edit_filter_accuracy'] == 'validated_approximate'
    radius = integration['association_radius_m']
    assert integration['max_origin_error_m'] <= integration['origin_error_budget_m']
    assert integration['points_retained'] < integration['points_before_filter']
    assert integration['filter_retention_ratio'] >= 0.75

    vertices, faces = read_triangles(output)
    assert len(faces) > 0
    size = settings['voxel_size_m']
    band = max(report['edited_geometry']['sampling_resolution_m'], size)
    # C. No output triangle inside a removed region, outside the documented band.
    assert centroids_in_region(vertices, faces, PATCH, inset=radius) == 0
    assert centroids_in_region(vertices, faces, CONNECTOR, inset=radius) == 0
    # A. Intended retained-surface coverage measured as spatial occupancy at the TSDF
    # resolution, with the documented uncertainty band around the removed regions.
    centroids = vertices[faces].mean(axis=1)
    removed = (PATCH, CONNECTOR)
    for name, extent in (('wall', WALL), ('thin retained strip', STRIP), ('second object', OBJECT)):
        exclude = removed if name != 'object' else (CONNECTOR,)
        coverage, cells = plane_occupancy(centroids, extent, size, exclude=exclude, band=band)
        assert cells > 100, (name, cells)
        assert coverage >= 0.95, f'{name} occupancy {coverage:.3f} below the 0.95 target'
    # No large artificial holes in the kept wall even without any band allowance,
    # but the removed regions themselves must stay empty.
    strict, cells = plane_occupancy(centroids, WALL, size, exclude=removed, band=0.0)
    assert cells > 100 and strict >= 0.95, f'kept wall occupancy {strict:.3f} without a band'
    # E. No bridging across the deleted connector between the wall and the second object.
    bridge = centroids_in_region(vertices, faces, (WALL[1]+0.2, OBJECT[0]-0.2, 0.2, 2.3))
    assert bridge == 0, 'removed geometry must not be recreated by meshing'
    assert report['extraction']['masked_triangles'] >= 0


def test_unsupported_observations_are_reported_and_never_silently_included(tmp_path, vdbfusion_python):
    scene = build_cleanup(tmp_path)
    bag, traj = write_scene_bag(tmp_path, scene['raw'])
    settings = dict(voxel_size_m=0.05, sdf_trunc_m=0.15)
    _, report = run_worker(tmp_path, vdbfusion_python, bag, traj, scene['workspace'], settings)
    integration = report['integration']
    assert integration['points_dropped_unsupported'] > 0
    assert integration['unsupported_observations'] == 'exclude'
    assert (integration['points_retained'] + integration['points_dropped_removed_support'] +
            integration['points_dropped_unsupported'] == integration['points_before_filter'])
    # The band and its source are reported, so a reader can judge the uncertainty.
    assert report['edit_filter_accuracy'] == 'validated_approximate'
    assert report['association_radius_m'] == integration['association_radius_m']
    assert 'uncertainty band' in report['edited_geometry']['limitation']


def test_unedited_and_edited_runs_both_produce_a_valid_mesh(tmp_path, vdbfusion_python):
    scene = build_cleanup(tmp_path)
    bag, traj = write_scene_bag(tmp_path, scene['raw'])
    settings = dict(voxel_size_m=0.02, sdf_trunc_m=0.06)
    plain = tmp_path/'plain'
    plain.mkdir()
    unedited, unedited_report = run_worker(plain, vdbfusion_python, bag, traj, None, settings)
    assert unedited_report['integration']['edit_filter_enabled'] is False
    unedited_vertices, unedited_faces = read_triangles(unedited)
    # Without an edit the removed patch IS reconstructed, which proves the edit matters.
    assert centroids_in_region(unedited_vertices, unedited_faces, PATCH, inset=0.1) > 0
    edited, edited_report = run_worker(tmp_path, vdbfusion_python, bag, traj, scene['workspace'], settings)
    edited_vertices, edited_faces = read_triangles(edited)
    assert edited_report['integration']['edit_filter_enabled'] is True
    assert centroids_in_region(edited_vertices, edited_faces, PATCH,
                               inset=edited_report['integration']['association_radius_m']) == 0


def test_masked_triangles_are_optional_and_reported(tmp_path, vdbfusion_python):
    scene = build_cleanup(tmp_path)
    bag, traj = write_scene_bag(tmp_path, scene['raw'])
    settings = dict(voxel_size_m=0.02, sdf_trunc_m=0.06, mask_deleted_triangles=False)
    output, report = run_worker(tmp_path, vdbfusion_python, bag, traj, scene['workspace'], settings)
    assert report['extraction']['mask_enabled'] is False
    assert report['extraction']['masked_triangles'] == 0
    # Even without masking, the observation filter removes the removed regions.
    vertices, faces = read_triangles(output)
    assert centroids_in_region(vertices, faces, CONNECTOR, inset=report['integration']['association_radius_m']) == 0
