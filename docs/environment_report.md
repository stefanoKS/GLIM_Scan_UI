# Environment report

2026-10-01T17:05:35+09:00

## uname -a
```text
Linux ubunturos-desktop 6.8.0-138-generic #138~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Aug  7 13:43:15 UTC  x86_64 x86_64 x86_64 GNU/Linux
```

## uname -m
```text
x86_64
```

## lsb_release -a
```text
No LSB modules are available.
Distributor ID:	Ubuntu
Description:	Ubuntu 22.04.5 LTS
Release:	22.04
Codename:	jammy
```

## cat /etc/nv_tegra_release
```text
cat: /etc/nv_tegra_release: No such file or directory
```

## nvcc --version
```text
bash: line 1: nvcc: command not found
```

## ls /usr/local/cuda*/version*
```text
ls: cannot access '/usr/local/cuda*/version*': No such file or directory
```

## free -h
```text
               total        used        free      shared  buff/cache   available
Mem:            15Gi       2.8Gi       8.8Gi        60Mi       3.7Gi        12Gi
Swap:          1.9Gi          0B       1.9Gi
```

## df -h .
```text
Filesystem                 Size  Used Avail Use% Mounted on
/dev/mapper/vgubuntu-root  914G  226G  642G  27% /
```

## tr -d '\0' </proc/device-tree/model
```text
bash: line 1: /proc/device-tree/model: No such file or directory
```

## printenv ROS_DISTRO
```text
humble
```

## ls /opt/ros
```text
humble
```

## python3 --version
```text
Python 3.13.2
```

## /usr/bin/python3 --version
```text
Python 3.10.12
```

## dpkg-query -W python3-colcon-core
```text
python3-colcon-core	0.21.2+upstream-1
```

## ip -brief address
```text
lo               UNKNOWN        127.0.0.1/8 ::1/128 
enp6s0           UP             192.168.1.135/24 fe80::ac18:8a8d:961d:dc50/64 
wlx6c1ff735eaa5  UP             10.103.120.85/16 fe80::5ba7:f49a:f845:d4bf/64 
```

This report describes the machine that ran this script; it is not evidence of Jetson testing.
