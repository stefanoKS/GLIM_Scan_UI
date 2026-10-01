# Official dependencies

Checkouts here are ignored by Git. `../dependencies.lock` pins exact upstream revisions; `scripts/fetch_dependencies.py` fetches them without resetting dirty trees. Keep each upstream LICENSE with its source and binaries. The top-level MIT license applies only to original project code, not upstream GLIM/Livox/third-party code.

GLIM and glim_ros2 provide the only SLAM backend. Livox SDK2 and livox_ros_driver2 provide sensor acquisition. GTSAM 4.3a0 and gtsam_points provide estimation dependencies. Iridescence provides the official viewer/exporter. Viewer prerequisites belong to the host distro, not copied desktop CUDA packages.

Three.js 0.170.0 is vendored under ui/frontend/vendor with its MIT license. It runs entirely locally; no CDN is required. Python dependencies are pinned in requirements.txt, with their transitive resolution captured after installation.
