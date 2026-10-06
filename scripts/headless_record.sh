#!/usr/bin/env bash
set -eo pipefail
if [[ ! -f "/opt/ros/${ROS_DISTRO:-humble}/setup.bash" ]]; then
    echo 'ERROR: ROS2 Humble is missing (/opt/ros/humble/setup.bash).' >&2
    exit 1
fi
source "$(dirname "$0")/env.sh"
exec python3 "$ROOT/scripts/headless_record.py" "$@"
