#!/usr/bin/env bash
# Verify the real VDBFusion native runtime.
#
# Import success alone is never sufficient: the worker constructs a small TSDF,
# integrates observations, extracts a real triangle mesh and validates finite
# vertices with in-range triangle indices. Nothing in the ROS or NKSR
# environment is touched.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${VDBFUSION_PYTHON:-}"
if [[ -z "$PYTHON" && -f "$ROOT/.state/vdbfusion_python.txt" ]]; then IFS= read -r PYTHON < "$ROOT/.state/vdbfusion_python.txt"; fi
PYTHON="${PYTHON:-$HOME/.cache/factory-mapping/vdbfusion-env/bin/python}"
[[ -x "$PYTHON" ]] || { echo 'VDBFUSION_NOT_INSTALLED: run scripts/setup_vdbfusion.sh'; exit 1; }
# The worker imports without ROS for --check; reconstruction adds ROS paths through
# scripts/env.sh or the managed backend environment.
unset PYTHONHOME
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$PYTHON" "$ROOT/tools/vdbfusion_worker.py" --check \
  --health-output "$ROOT/.state/vdbfusion_health.json" "$@"
