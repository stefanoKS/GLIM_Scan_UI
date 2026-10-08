#!/usr/bin/env bash
# Staged VDBFusion measurements on real data. See tools/benchmark_vdbfusion.py --help.
#
# Nothing here estimates a speedup or compares against another engine unless both
# were measured on the same bag and trajectory.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${VDBFUSION_PYTHON:-}"
if [[ -z "$PYTHON" && -f "$ROOT/.state/vdbfusion_python.txt" ]]; then IFS= read -r PYTHON < "$ROOT/.state/vdbfusion_python.txt"; fi
PYTHON="${PYTHON:-$HOME/.cache/factory-mapping/vdbfusion-env/bin/python}"
[[ -x "$PYTHON" ]] || { echo 'VDBFUSION_NOT_INSTALLED: run scripts/setup_vdbfusion.sh'; exit 1; }
# ROS bindings must stay reachable: the worker streams the raw bag itself.
source "$ROOT/scripts/env.sh" >/dev/null 2>&1 || true
unset PYTHONHOME
exec "$PYTHON" "$ROOT/tools/benchmark_vdbfusion.py" "$@"
