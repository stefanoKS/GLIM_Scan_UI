# Environment report

2026-10-07T12:36:46+09:00

## uname -a
```text
Linux sekkei-master 6.8.0-138-generic #138~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Aug  7 13:43:15 UTC  x86_64 x86_64 x86_64 GNU/Linux
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
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```

## ls /usr/local/cuda*/version*
```text
/usr/local/cuda-13.2/version.json
/usr/local/cuda-13/version.json
/usr/local/cuda/version.json
```

## free -h
```text
               total        used        free      shared  buff/cache   available
Mem:            61Gi        23Gi       1.1Gi       165Mi        36Gi        37Gi
Swap:          2.0Gi       2.0Mi       2.0Gi
```

## df -h .
```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda2       1.8T  1.6T  130G  93% /
```

## tr -d '\0' </proc/device-tree/model
```text
bash: line 1: /proc/device-tree/model: No such file or directory
```

## printenv ROS_DISTRO
```text
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
python3-colcon-core	0.18.3-1
```

## ip -brief address
```text
lo               UNKNOWN        127.0.0.1/8 ::1/128 
enp5s0           UP             10.1.6.253/16 fd99:607c:a8e4:4c93:aeb:b274:1645:a3ca/64 fd99:607c:a8e4:4c93:86d6:f26c:1991:8a7a/64 fe80::da66:ceb0:ca88:e9ed/64 
wlp12s0          DOWN           
docker0          DOWN           172.17.0.1/16 
br-3f4a3d6a601a  UP             172.18.0.1/16 fe80::7423:99ff:fe36:a495/64 
br-7750ee8b4868  UP             172.19.0.1/16 fe80::6c9b:4bff:fea5:721b/64 
veth4b1842e@if2  UP             fe80::2c42:72ff:fe49:db9/64 
veth8f6047e@if2  UP             fe80::ecf9:73ff:fe93:5d83/64 
vethd09b2b1@if2  UP             fe80::88c3:54ff:fe43:6159/64 
veth84d830d@if2  UP             fe80::6c2d:80ff:fe1f:b73a/64 
veth082c3ee@if2  UP             fe80::86c:d9ff:fe5c:1d26/64 
vethe74f2b7@if2  UP             fe80::847b:abff:fee7:409a/64 
veth2689cd3@if2  UP             fe80::ce8:37ff:fe6d:b91f/64 
veth5898c8a@if2  UP             fe80::388d:9cff:febf:3243/64 
veth95f8d86@if2  UP             fe80::38e8:8eff:fe73:9273/64 
veth7e2708d@if2  UP             fe80::20ff:daff:fe2d:7314/64 
vethf3c944e@if2  UP             fe80::d05a:3ff:fe86:3032/64 
veth8351d11@if2  UP             fe80::4090:4aff:fe11:d0df/64 
vethbb6f28a@if2  UP             fe80::e44a:62ff:fead:e2e2/64 
vethd952172@if2  UP             fe80::c4cd:bdff:fede:ce00/64 
veth782921e@if2  UP             fe80::986d:62ff:fe4f:a7a0/64 
veth4ba38d1@if2  UP             fe80::b021:48ff:feed:76c/64 
veth358eafd@if2  UP             fe80::90fa:23ff:fea9:c41c/64 
vethcbc1c0e@if2  UP             fe80::6c27:12ff:fe48:6600/64 
veth7f3f967@if2  UP             fe80::3c65:85ff:fefd:1d48/64 
```

This report describes the machine that ran this script; it is not evidence of Jetson testing.
