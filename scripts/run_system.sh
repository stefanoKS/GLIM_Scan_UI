#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
HOST=127.0.0.1
PORT=8080
while [[ $# -gt 0 ]]; do
 case "$1" in
  --mock) export FACTORY_MAPPING_MOCK=1; shift;;
  --host) HOST="$2"; shift 2;;
  --port) PORT="$2"; shift 2;;
  *) echo "Usage: $0 [--mock] [--host IP] [--port PORT]" >&2; exit 2;;
 esac
done
exec "$ROOT/.venv/bin/python" -m uvicorn factory_mapping.api:app --host "$HOST" --port "$PORT" --timeout-graceful-shutdown 420
