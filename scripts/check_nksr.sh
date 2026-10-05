#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${NKSR_PYTHON:-}"
if [[ -z "$PYTHON" && -f "$ROOT/.state/nksr_python.txt" ]]; then IFS= read -r PYTHON < "$ROOT/.state/nksr_python.txt"; fi
PYTHON="${PYTHON:-$HOME/.cache/factory-mapping/nksr-env/bin/python}"
[[ -x "$PYTHON" ]] || { echo 'NKSR_NOT_INSTALLED: run scripts/setup_nksr.sh'; exit 1; }
unset PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$PYTHON" "$ROOT/tools/nksr_worker.py" --check --device "${NKSR_DEVICE:-auto}" --health-output "$ROOT/.state/nksr_health.json" "$@"
