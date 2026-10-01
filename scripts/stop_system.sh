#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
# Ask the controlled backend to finalize acquisition; stop server itself with Ctrl-C.
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/fm.py" action session_stop
