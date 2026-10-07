# Map editing and surface reconstruction workflow

This guide covers the normal workstation workflow from a completed GLIM map to an
edited point-cloud export and an NKSR triangle mesh. The recording Jetson does not
run native map tools, reconstruction preparation, or NKSR.

## Workflow overview

```text
Completed GLIM processing run
  -> Clean Map
  -> save explicitly as edits/edit_ID/saved_map/
  -> close the native editor
  -> Export saved edited map
  -> Prepare Reconstruction
  -> Check NKSR
  -> Reconstruct Mesh
  -> inspect or download output/mesh.ply
```

Each stage creates derived data. The raw bag and original
`processing/run_NNN/glim_dump/` remain unchanged.

## 1. Process the scan

On the processing workstation:

1. Start the dashboard with `scripts/run_system.sh`.
2. Select the completed session in **Library**.
3. Run **Process** or **Reprocess** if the session does not already have a completed
   GLIM processing run.
4. Select the completed run before opening a native map tool.

Every processing attempt has its own `processing/run_NNN/` directory. Do not edit
that directory directly.

## 2. Choose the correct native tool

The two buttons have different purposes:

| Dashboard button | Native tool | Use it for |
| --- | --- | --- |
| **Edit Map** | `offline_viewer` | Loop constraints, pose-graph changes, alignment, optimization, and merging |
| **Clean Map** | `map_editor` | Selecting, segmenting, annotating, and deleting points while keeping poses fixed |

For deleting unwanted geometry before surfacing, click **Clean Map**, not
**Edit Map**.

Every click on **Edit Map**, **Clean Map**, or **Merge Maps** creates a new,
independent `edits/edit_ID/` workspace. It does not reopen or overwrite an older
workspace. For example:

```text
edits/
  edit_111111111111/  # an earlier native-tool attempt
  edit_222222222222/  # the new Clean Map workspace
```

Use the newly displayed workspace ID for the current operation. Old workspaces
remain available for provenance and export, but the dashboard does not currently
reopen an old workspace for additional native editing.

The new Clean Map workspace contains:

```text
edits/edit_ID/
  map_01/       # copied source map
  saved_map/    # explicit native Save As target
  workspace.json
  tool.log
```

The native `map_editor` window opens on the workstation desktop, not in the
browser. Use its selection, segmentation, annotation, and removal tools to clean
the geometry.

Important rules:

- Keep submap poses fixed in `map_editor`.
- If the map needs pose correction, use **Edit Map** first. Save and export that
  result separately. The current dashboard starts **Clean Map** from a selected
  completed processing run; it does not automatically chain a saved
  `offline_viewer` workspace into a new cleanup workspace.
- Avoid removing every point from a submap when practical. Nearly empty submaps
  are valid cleanup results, but some upstream optimization factors cannot be
  reconstructed from them.
- Choose **Save As** in the native application and save to the exact displayed
  `edits/edit_ID/saved_map/` directory.
- Closing the native window does not save changes.
- Never select the original `processing/run_NNN/glim_dump/` as a save destination.

After saving, close the native editor or click **Stop native tool**.

Do not click **Clean Map** a second time merely to export the first cleanup. That
would create another new `edit_ID`. After closing the first cleanup window, use
the **Export saved edited map** button on the row for that same workspace ID.

## 3. Export the saved edit

Click **Export saved edited map** for the closed workspace. A successful export:

- reads `saved_map/`;
- uses the saved trajectory and poses;
- writes a new `exports/edit_ID_HASH.ply`;
- writes `edits/edit_ID/export.json` with the map and trajectory fingerprints; and
- changes the reconstruction selector from **export required** to **export ready**.

Export uses a temporary hard-linked staging copy. Optimization-only matching-cost
factors are omitted from that staging copy because the official exporter does not
need them to write the fixed-pose point cloud, and those factors can fail when map
cleanup leaves a submap nearly empty. The saved edit, original graph, and source
processing run are not modified. The staging copy is removed after export.

If the UI still reports **export required**:

1. Confirm that the native editor was explicitly saved to `saved_map/`.
2. Close the editor before exporting.
3. Open or download `edits/edit_ID/export.log`.
4. Confirm that `exports/edit_ID_HASH.ply` and `edits/edit_ID/export.json` exist.
5. Restart the backend after updating application code; an already-running backend
   does not load Python changes automatically.

## 4. Prepare reconstruction input

Under **Surface Reconstruction**:

1. Select the trajectory under the same `edits/edit_ID/saved_map/`.
2. Enable **Use saved edited geometry**.
3. Select the corresponding **export ready** cleanup.
4. Set the retained-geometry tolerance. The default is `0.05` meters.
5. Choose the preparation voxel size under **Advanced**. The default is `1.0` cm.
6. Click **Prepare Reconstruction**.

Preparation reads the original raw bag, interpolates the selected saved trajectory,
transforms observations into world coordinates, filters them against the retained
edited geometry, and then performs deterministic voxel selection. The edited PLY
is a spatial reference; the prepared points still come from the original sensor
observations and preserve their paired sensor origins.

The edited-geometry spatial index has a 5 GiB estimated-memory limit. The final
voxel size does not reduce this reference index because filtering happens before
voxel sampling.

A successful run shows **PREPARED** and creates:

```text
reconstruction/run_ID/
  job.json
  job.log
  input/nksr_input.npz
  validation/reconstructed_from_bag.ply
  validation/comparison.json
```

Preparation does not create a mesh and does not require NKSR.

### Which trajectory to select

Use the trajectory that represents the geometry you intend to reconstruct:

- For the original, unedited GLIM result, select
  `processing/run_NNN/glim_dump/traj_lidar.txt`.
- For a saved Clean Map result, select
  `edits/edit_ID/saved_map/traj_lidar.txt` from the same workspace as the
  cleanup export.
- Do not select `edits/edit_ID/map_01/traj_lidar.txt` for the final job.
  `map_01/` is the initial working copy, while `saved_map/` is the explicit
  native save result.

When **Use saved edited geometry** is enabled, the backend verifies the selected
cleanup export and binds preparation to that cleanup's
`saved_map/traj_lidar.txt`. This prevents an export from one edit workspace from
being combined with a trajectory from another workspace. The trajectory selector
should still be set to the matching `saved_map` path so the UI accurately reflects
the intended input.

`map_editor` keeps poses fixed, so its saved trajectory should match its original
working-copy trajectory; preparation verifies this before accepting a filtered
cleanup. A trajectory changed by `offline_viewer` is appropriate for an
unfiltered reconstruction from that saved pose-corrected map, but it cannot be
substituted for the verified trajectory of a different Clean Map export.

## 5. Reconstruct the surface

1. Click **Check NKSR** and wait for **READY**.
2. Select a **PREPARED** input.
3. Choose the reconstruction settings.
4. Click **Reconstruct Mesh**.
5. Follow the reported stage and inspect the reconstruction log if it fails.

NKSR loads the prepared points and sensor origins, evaluates the pretrained model,
extracts a triangle mesh, and validates that the output contains finite vertices
and valid faces. Successful output is stored at:

```text
reconstruction/run_ID/output/mesh.ply
```

**PREPARED** means only that point input exists. **COMPLETED** means the independent
NKSR worker exited successfully and the triangle mesh passed validation.

### Mesh output mode

**Mesh output** selects how the result is saved:

| Mode | Behavior |
| --- | --- |
| **Merged** (default) | Existing behavior: a single fused `output/mesh.ply`. |
| **Per-chunk** | Saves only the individual NKSR chunk meshes under `output/mesh_chunks/` (no fused mesh). Chunked reconstruction reuses the per-chunk fields NKSR already built; no extra reconstruction pass runs. |
| **Both** | Saves the per-chunk meshes and then the fused `output/mesh.ply`. |

Chunk output lives in:

```text
reconstruction/run_ID/output/mesh_chunks/
  chunk_0000.ply
  chunk_0001.ply
  ...
  chunks.json
```

Each `chunk_NNNN.ply` contains world-space vertices in meters in the original
GLIM/world coordinate system, so loading all chunks together (for example in
Houdini, with no transform) aligns them with the point cloud, each other, and
the fused mesh. `chunks.json` records the coordinate system, units, the physical
and scaled chunk size, the scaled stride, and per-chunk grid indices and world
bounding boxes. `core_bbox_min`/`core_bbox_max` are in world meters;
`core_bbox_min_scaled`/`core_bbox_max_scaled`, `field_origin_scaled` and
`chunk_stride_scaled` stay in scaled NKSR coordinates, which is stated in the
field names.

NKSR reconstructs each chunk with overlap to give the field context, and it
skips candidate cells that contain no points, so the reconstructed fields do not
form a complete Cartesian grid. Ownership therefore works as follows:

- every triangle belongs to the active chunk whose nominal cube contains its
  centroid, choosing the nearest chunk center and the lowest field index on a tie
  (this also resolves diagonal overlaps);
- geometry whose centroid lies inside no active cube stays with the chunk that
  emitted it, so a chunk in an unrelated grid cell can never crop it;
- because ownership is decided from the centroid and the full active set, no
  triangle is exported by two neighboring chunks.

Ownership by triangle centroid means a triangle is never split, so neighboring
chunk meshes meet with a hairline seam (under half a triangle wide) instead of
being welded. Neighboring chunks are intentionally **not** welded together.

In full (non-chunked) mode, **Per-chunk** saves a single chunk representing the
whole field, and **Both** saves that chunk plus the normal fused mesh; no extra
reconstruction pass is performed.

Low RAM mode keeps its existing independent tile meshes plus the merged output;
**Mesh output** does not apply there and is disabled in the UI.

## Common failures

| Status or message | Meaning | Action |
| --- | --- | --- |
| `export required` | No completed, fingerprinted edited-map export is available | Save the native edit, close the tool, and export again |
| `source points have not been allocated` | Upstream GLIM tried to rebuild optimization factors for a nearly empty submap | Restart onto a build with staged edited-map export, then export again |
| `No module named scipy` | App dependencies were not installed inside `.venv` | Install the pinned requirements (`requirements.lock` includes SciPy) into `.venv` and retry preparation |
| Spatial index exceeds 5 GiB | The retained edited PLY is too large for the configured preparation guard | Export a smaller cleanup or raise the reviewed memory limit |
| Preparation stops after transforming frames | A later filtering or save step failed | Open `reconstruction/run_ID/job.log`; transformed frames alone do not mean PREPARED |
| NKSR READY but preparation FAILED | NKSR health is independent from point preparation | Fix and rerun preparation before starting the mesh worker |
| Mesh WARNING | A mesh was produced but a geometric validation threshold warned | Inspect the mesh and metadata before accepting it |

## Related guides

- [GLIM tools](glim_tools.md) describes native editing, optimization, and merging.
- [Operation](operation.md) describes capture, transfer, and workstation processing.
- [NKSR](nksr.md) documents installation, reconstruction modes, resource controls,
  output validation, and detailed diagnostics.
- [Troubleshooting](troubleshooting.md) covers process recovery and display/OpenGL
  failures.
