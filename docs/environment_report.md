# Environment report

2026-10-02T14:51:31+09:00

## uname -a
```text
Linux stefanojetson-desktop 5.15.199-tegra #1 SMP PREEMPT Thu Jul 16 11:27:38 PDT 2026 aarch64 aarch64 aarch64 GNU/Linux
```

## uname -m
```text
aarch64
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
# R36 (release), REVISION: 5.2, GCID: 46426093, BOARD: generic, EABI: aarch64, DATE: Thu Jul 16 18:56:22 UTC 2026
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
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
Mem:           7.4Gi       4.2Gi       248Mi        57Mi       3.0Gi       3.1Gi
Swap:          3.7Gi       1.0Gi       2.7Gi
```

## df -h .
```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p1  1.8T   50G  1.7T   3% /
```

## tr -d '\0' </proc/device-tree/model
```text
NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super```

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
Python 3.10.12
```

## /usr/bin/python3 --version
```text
Python 3.10.12
```

## dpkg-query -W python3-colcon-core
```text
python3-colcon-core	0.21.3+upstream-1
```

## ip -brief address
```text
lo               UNKNOWN        127.0.0.1/8 ::1/128 
enP8p1s0         DOWN           
wlP1p1s0         UP             10.103.120.97/16 fe80::5f14:f9e:2a3e:7ae4/64 
can0             DOWN           
l4tbr0           DOWN           
usb0             DOWN           
usb1             DOWN           
```

This report describes the machine that ran this script; it is not evidence of Jetson testing.
