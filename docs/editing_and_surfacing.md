# Map editing and surface reconstruction workflow

This guide covers the normal workstation workflow from a completed GLIM map to an
edited point-cloud export and a triangle mesh. Two surface reconstruction engines are
available: **VDBFusion** (fast TSDF, the default for new jobs when installed) and
**NKSR** (neural reconstruction). The recording Jetson does not run native map tools,
reconstruction preparation, either engine, or their isolated environments.

## Workflow overview

```text
Completed GLIM processing run
  -> Clean Map
  -> save explicitly as edits/edit_ID/saved_map/
  -> close the native editor
  -> Export saved edited map
  -> choose the reconstruction algorithm
  -> Prepare Reconstruction
  -> Check the selected engine
  -> Reconstruct Mesh
  -> inspect or download output/mesh.ply or the separate mesh cells
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

1. Choose the **Reconstruction algorithm** (the first control). **VDBFusion** is the
   default for new jobs when it is installed; an unavailable VDBFusion shows its
   installation status and is never silently replaced by NKSR. Requests and jobs
   without an algorithm stay NKSR.
2. Select the trajectory under the same `edits/edit_ID/saved_map/`.
3. Enable **Use saved edited geometry**.
4. Select the corresponding **export ready** cleanup.
5. For NKSR, set the retained-geometry tolerance. The default is `0.05` meters.
   VDBFusion derives its own association radius from the measured cleanup geometry and
   does not use this field.
6. For NKSR, choose the preparation voxel size under **Advanced**. The default is
   `1.0` cm. VDBFusion needs no prepared point cloud and reports a source-validation
   and preflight summary instead.
7. Click **Prepare Reconstruction**.

The algorithm chosen here is stored in the run's `job.json`, and every later mesh
request for that run uses that engine. A run prepared for one engine is never
reconstructed by the other.

NKSR preparation reads the original raw bag, interpolates the selected saved
trajectory, transforms observations into world coordinates, filters them against the
retained edited geometry, and then performs deterministic voxel selection. The edited
PLY is a spatial reference; the prepared points still come from the original sensor
observations and preserve their paired sensor origins.

VDBFusion preparation validates the same sources and reports free RAM, free disk, the
requested TSDF resolution, the sampled scan extent and a component-wise footprint
estimate with an explicit safety margin. It writes no prepared point cloud: the raw bag
is streamed later, during mesh reconstruction. It also pins the sources and the
semantic settings it was prepared with, so changing the voxel size, truncation, ROI or
edit-filtering policy afterwards marks the prepared input stale and asks for a
re-prepare instead of silently reconstructing something else. A run that cannot fit in
the usable memory is refused before it starts, with mitigation advice, rather than
started and killed.

The edited-geometry spatial index has a 5 GiB estimated-memory limit. The final
voxel size does not reduce this reference index because filtering happens before
voxel sampling.

A successful NKSR preparation shows **PREPARED** and creates:

```text
reconstruction/run_ID/
  job.json
  job.log
  input/nksr_input.npz
  validation/reconstructed_from_bag.ply
  validation/comparison.json
```

A successful VDBFusion preparation creates:

```text
reconstruction/run_ID/
  job.json
  job.log
  input/vdbfusion_prepare.json
```

Neither preparation step creates a mesh, and neither requires the other engine.

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

1. Click **Check NKSR** or **Check VDBFusion** for the selected engine and wait for
   **READY**.
2. Select a **PREPARED** input run. Each entry is labelled with the engine it was
   prepared for.
3. Choose the reconstruction settings for that engine.
4. Click **Reconstruct Mesh**.
5. Follow the reported stage and inspect the reconstruction log if it fails.

NKSR loads the prepared points and sensor origins, evaluates the pretrained model,
extracts a triangle mesh, and validates that the output contains finite vertices
and valid faces. VDBFusion streams the raw bag into one fused TSDF and extracts one
world-space triangle mesh, then optionally masks triangles inside removed regions.
Successful output is stored at:

```text
reconstruction/run_ID/output/mesh.ply                # Single mesh / Both
reconstruction/run_ID/output/mesh_chunks/            # Separate meshes / Both
reconstruction/run_ID/output/tiles/                  # NKSR Low RAM independent tiles
reconstruction/run_ID/output/nksr_metadata.json      # NKSR only
reconstruction/run_ID/output/vdbfusion_metadata.json # VDBFusion only
```

VDBFusion never writes NKSR metadata and never creates independent TSDF tiles. Its
separate meshes are spatial cells of the same single fused mesh. See
[VDBFusion](vdbfusion.md) for its settings, memory behaviour, measured edit results
and its approximate edited-geometry association.

**PREPARED** means only that point input exists. **COMPLETED** means the independent
NKSR worker exited successfully and every saved mesh passed validation. A worker exit
code of 0 alone is never enough.

### Mesh output mode

**Mesh output** selects how the finished geometry is saved. It is independent of the
reconstruction mode:

| Mode | Behavior |
| --- | --- |
| **Single mesh** (default, internal `merged`) | Existing behavior: a single fused `output/mesh.ply`. |
| **Separate meshes** (internal `chunks`) | Saves only the per-cell PLYs under `output/mesh_chunks/`, with no fused mesh. Full and Chunked reconstruct once and split the one final fused surface spatially. |
| **Both** | Saves the separate cells and the fused `output/mesh.ply`. The reconstruction still runs once. |

**One chunk size (meters)** is shared: `Auto` or a positive number. In Chunked mode
it is also passed to NKSR as the reconstruction chunk size, and it is always the edge
of the export cell. Auto resolves the reconstruction mode first and then the size
(Chunked: the existing 20/10/5 m density heuristic; Full and Low RAM: 5 m). After
processing the UI shows the effective size and where it came from
(`user`, `default`, `auto_density`, `oom_retry`, or `legacy_tile_size`).

Chunk output lives in:

```text
reconstruction/run_ID/output/mesh_chunks/
  chunk_0000.ply
  chunk_0001.ply
  ...
  chunks.json
```

Each `chunk_NNNN.ply` contains world-space vertices in meters in the original
GLIM/world coordinate system, so loading all cells together (for example in Houdini,
with no transform) aligns them with the point cloud, each other, and the fused mesh.

Ownership works on the final mesh alone: every triangle belongs to the cell that
contains its float64 centroid on a world-aligned grid of `effective_chunk_size_m`
cubes anchored at `[0, 0, 0]` (`floor(centroid / size)`, so negative coordinates are
correct). Triangles are never cut, so a triangle that crosses a cell boundary stays
whole in one file, neighbouring cells meet with a hairline seam, and shared boundary
vertices are duplicated with identical coordinates. Cells are written z-major, then
y, then x. Cells that own no triangle produce no file. `SUM(cell faces)` therefore
equals the source face count, while `SUM(cell vertices)` is normally larger than the
source because of that duplication.

`chunks.json` records the export strategy, the actual reconstruction mode, the
coordinate system and units, the grid origin and cell ordering, the requested and
effective chunk size with its source, the source face count, the exported totals, and
per-cell grid indices, nominal cell bounds, world bounding boxes, counts, and file
sizes. Filenames are bare relative names, and validation rejects path traversal,
symlink escapes, duplicate names, missing files, invalid indices, and nonpositive
sizes. Older manifests from the former native per-field exporter are still read.

Full and Chunked separate output still require enough memory to reconstruct and
extract the complete surface; tiling reduces file sizes and downstream loading
pressure, not peak reconstruction RAM. The native per-field exporter remains in the
worker for compatibility and future advanced use, but the standard output selection
never uses it.

Low RAM mode also accepts all three output selections and keeps its existing
independent tile layout under `output/tiles/`. **Separate meshes** skips the final
merge entirely and writes no `output/mesh.ply`; **Single mesh** or **Both** assemble
the merged mesh after all tiles finish. Independent tiles are never stitched, so
their boundaries may show seams.

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
- [VDBFusion](vdbfusion.md) documents its isolated installation, TSDF settings,
  streaming and memory behaviour, edited-geometry association and measured results.
- [Troubleshooting](troubleshooting.md) covers process recovery and display/OpenGL
  failures.
