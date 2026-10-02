#!/usr/bin/env bash
# Read-only diagnostics: no firmware, trigger, networking or pipeline changes.
set -eo pipefail
source "$(dirname "$0")/env.sh"
failed=0
for name in gst-inspect-1.0 tcam-ctrl; do
 command -v "$name" || { echo "Missing $name: install optional camera dependencies"; failed=1; }
done
if command -v gst-inspect-1.0 >/dev/null; then gst-inspect-1.0 tcambin || failed=1; fi
if command -v tcam-ctrl >/dev/null; then
 tcam-ctrl --list || failed=1
 if [[ -n "${1:-}" ]]; then tcam-ctrl --caps "$1" || failed=1; fi
fi
ros2 pkg executables gscam2 || failed=1
"$ROOT/.venv/bin/python" -c 'import cv2,cv_bridge;print("System/ROS preview dependencies:",cv2.__version__)' || failed=1
exit "$failed"
