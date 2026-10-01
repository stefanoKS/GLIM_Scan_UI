#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/sensor_cli.py" "$@"
