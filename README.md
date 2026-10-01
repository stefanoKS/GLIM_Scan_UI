# Factory Mapping — GLIM + Mid-360

Local ROS 2 acquisition, immutable raw bags, GLIM processing jobs and a lightweight browser interface. Phase 1 uses **only the Mid-360 LiDAR and its IMU**. The PC can perform processing; the future Jetson can focus on acquisition.

## Quick start on this PC

```bash
cd '/home/ubuntu-ros/Documents/GLIM Factory Mapping/factory_mapping'
scripts/run_system.sh
```

Open http://127.0.0.1:8080 and enter the operator token from `.state/operator.token`. For another PC on the trusted LAN, use `scripts/run_system.sh --host 0.0.0.0`, then open `http://192.168.1.135:8080`. The token protects operations; this is a local HTTP application, not an Internet service. One backend instance per repository is allowed.

Use `--mock` for development without ROS hardware. Mock sessions are marked and filtered separately; they cannot be exported as genuine maps. Stop with Ctrl-C to finalize recording and GLIM. `scripts/stop_system.sh` finalizes the active session while keeping the dashboard running.

The configured device is `192.168.1.120`, host `192.168.1.135`, interface `enp6s0`. Network edits only update project configuration; they do not change host networking or sensor firmware settings.

## Terminal-only test

With the UI stopped:

```bash
scripts/record_test.sh --name terminal_test --seconds 30
# Copy the session ID printed by the recorder:
scripts/process_bag.sh YYYYMMDD_HHMMSS_terminal_test --preset jetson_cpu
```

Each run writes a new `processing/run_NNN/` directory. Inspect `job.log`, `job.json` and `glim_dump/`. Never pass an unfinished bag to GLIM.

## Installation and verification

See [installation](docs/installation_jetson.md), [environment report](docs/environment_report.md), [validation results](docs/validation.md), [operation](docs/operation.md) and [upstream interfaces/preset changes](docs/upstream_interfaces.md).

```bash
source scripts/env.sh
.venv/bin/python -m pytest -q --basetemp=.state/pytest
```

No ARM64 build result is claimed from the x86_64 test machine. No AVX/native architecture options are enabled. Dependency revisions and build evidence are in `dependencies.lock`. Build/install/log trees, raw bags and map outputs are excluded from Git.
