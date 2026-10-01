# Troubleshooting

- **Ethernet missing**: select the correct wired interface. **Host IP missing**: configure an address with the OS or change project host_ip to the assigned address. **Unreachable**: check power, cable and subnet; ICMP may be filtered.
- **Driver stopped/failed**: inspect its process state and `.state/driver.log` or the session's `logs/driver.log`. **Topic missing**: verify ROS_DOMAIN_ID, remappings and PointCloud2 xfer_format. **Topic exists/no messages**: check UDP destination IP, firewall, device state and sensor power. **Abnormal rate**: inspect CPU load, DDS loss, Ethernet packets and publish frequency; expected defaults are 10 Hz cloud/200 Hz IMU.
- **ROS Python import errors**: source `scripts/env.sh`, use `.venv/bin/python` based on `/usr/bin/python3`, not Conda Python. Do not pip-install a replacement rclpy.
- **GPU preset fails**: this test PC has no CUDA. Use jetson_cpu, or build the CUDA stack on a compatible host.
- **Official driver shutdown bus error**: observed on this PC after SDK deinitialization. The bag finalized before the driver stopped. Treat driver crash as a reported fault; verify the next start and preserve logs. No unreviewed driver patch was applied.
- **Missing bag metadata**: interrupted recording is not silently marked successful. Preserve the raw bag; inspect with ROS tools. Any reindex/recovery must be an explicit operation on a backup, not part of GLIM processing.
- **GLIM has no valid output**: inspect processing/run_NNN/job.log. Empty input, bad timestamp units, incorrect extrinsics or missing libraries are possible causes. Completed status requires a saved graph, not merely a zero exit status.
- **Exporter/OpenGL failure**: upstream offline_viewer requires a display. Run export on the workstation; a headless build without viewer cannot export directly. Do not invent a different CLI.
- **Orphaned process**: inspect the reported PID and command in .state/process_*.json, then send SIGINT to that known process group. The backend refuses automatic PID adoption. Restart after it exits. Never remove state to hide a still-running recorder.
- **Disk low**: automatic graceful stop begins at the configured free-space threshold. Move completed sessions to external storage before another run.

Use `scripts/diagnose.sh`, UI diagnostics, and per-session logs. Full logs remain on disk even when the UI shows only a bounded recent stream.
