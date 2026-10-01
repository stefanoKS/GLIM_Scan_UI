from collections import deque
from pathlib import Path
import asyncio, socket, time
import psutil
from .storage import machine, read_json

class Rates:
    def __init__(self): self.times=deque(maxlen=4000); self.count=0; self.first=None; self.last=None; self.points=0; self.stamp=None
    def add(self,stamp,points=0):
        t=time.monotonic(); self.times.append(t); self.first=self.first or t; self.last=t; self.count+=1; self.points=points; self.stamp=stamp
    def view(self):
        t=time.monotonic(); recent=[x for x in self.times if t-x<=5]
        hz=(len(recent)-1)/(recent[-1]-recent[0]) if len(recent)>1 else 0
        age=t-self.last if self.last else None
        if age is None or age>2: hz=0
        return dict(hz=hz,age=age,count=self.count,average_hz=(self.count-1)/(self.last-self.first) if self.count>1 else None,stamp=self.stamp,point_rate=hz*self.points)

def system_status(root):
    temps=psutil.sensors_temperatures(); thermal={k:[x.current for x in v] for k,v in temps.items()}
    gpu=None
    for p in Path('/sys/devices').glob('platform/*gpu/load'):
        try: gpu=float(p.read_text())/10
        except OSError: pass
    interfaces={k:[a.address for a in v if a.family==socket.AF_INET] for k,v in psutil.net_if_addrs().items()}
    mem=psutil.virtual_memory(); disk=psutil.disk_usage(root)
    return dict(**machine(),cpu_percent=psutil.cpu_percent(),ram_percent=mem.percent,ram_available=mem.available,disk_free=disk.free,temperatures=thermal,gpu_percent=gpu,interfaces=interfaces)

async def network(sensor):
    interfaces=psutil.net_if_addrs()
    if sensor['interface'] not in interfaces: return dict(state='ethernet_missing',action='Select the wired interface in network configuration')
    if sensor['host_ip'] not in [a.address for a in interfaces[sensor['interface']]]: return dict(state='host_ip_missing',action='Set host_ip to an address assigned to the selected interface')
    p=await asyncio.create_subprocess_exec('ping','-I',sensor['interface'],'-c','1','-W','1',sensor['lidar_ip'],stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
    await p.wait()
    return dict(state='reachable' if p.returncode==0 else 'unreachable',action=None if p.returncode==0 else 'Check Mid-360 power, Ethernet cable, subnet and LiDAR IP; ICMP alone does not prove message health')
