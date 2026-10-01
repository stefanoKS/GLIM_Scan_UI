#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/docs/environment_report.md}"
mkdir -p "$(dirname "$OUT")"
{
 echo '# Environment report'
 echo
 date -Is
 for cmd in 'uname -a' 'uname -m' 'lsb_release -a' 'cat /etc/nv_tegra_release' 'nvcc --version' 'ls /usr/local/cuda*/version*' 'free -h' 'df -h .' "tr -d '\\0' </proc/device-tree/model" 'printenv ROS_DISTRO' 'ls /opt/ros' 'python3 --version' '/usr/bin/python3 --version' "dpkg-query -W python3-colcon-core" 'ip -brief address'; do
   echo; echo "## $cmd"; echo '```text'; bash -c "$cmd" 2>&1 || true; echo '```'
 done
 echo
 echo 'This report describes the machine that ran this script; it is not evidence of Jetson testing.'
} > "$OUT"
cat "$OUT"
