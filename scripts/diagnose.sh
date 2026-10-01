#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
"$ROOT/.venv/bin/python" - <<'PY'
import asyncio,json
from factory_mapping.config import ROOT,load
from factory_mapping.health import network,system_status
from factory_mapping.storage import read_json
c=load();print(json.dumps({'system':system_status(ROOT),'network':asyncio.run(network(c['sensor'])),'sensor':read_json(ROOT/'.state/health.json'),'configuration':c},indent=2))
PY
ros2 topic list -t
ros2 node list
