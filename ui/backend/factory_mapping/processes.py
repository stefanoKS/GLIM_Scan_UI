"""Async process groups with bounded graceful shutdown and identity-safe recovery."""
import asyncio, os, signal, time
from pathlib import Path
import psutil
from .storage import atomic_json, read_json, now

def group_alive(pgid):
    for proc in psutil.process_iter(['pid','status']):
        try:
            if proc.info['status']!=psutil.STATUS_ZOMBIE and os.getpgid(proc.pid)==pgid: return True
        except (ProcessLookupError, PermissionError, psutil.Error): pass
    return False

async def wait_group(pgid):
    while group_alive(pgid): await asyncio.sleep(.1)

class ProcessManager:
    def __init__(self, state):
        self.state=Path(state); self.state.mkdir(parents=True,exist_ok=True)
        self.items={}; self.lock=asyncio.Lock(); self.recovery=[]
        for file in self.state.glob('process_*.json'):
            old=read_json(file,{})
            try:
                p=psutil.Process(old['pid'])
                identity_matches=abs(p.create_time()-old['created'])<0.01
                is_zombie=p.status()==psutil.STATUS_ZOMBIE
                same=identity_matches and not is_zombie
                check_group=identity_matches and is_zombie
            except (psutil.Error,KeyError):
                same=False
                check_group=True
            if not same and check_group and group_alive(old.get('pid',-1)): same=True
            if same:
                # Never kill a possibly unrelated PID or spawn a duplicate after a backend crash.
                self.items[old['key']]={**old,'state':'orphaned','process':None}
                self.recovery.append(f"Managed process {old['key']} is still alive (PID {old['pid']}); stop it before restarting that role.")
            else: file.unlink(missing_ok=True)
    def active(self,key): return key in self.items and self.items[key]['state'] in ('running','stopping','orphaned')
    def view(self): return {k:{a:b for a,b in v.items() if a not in ('process','watcher')} for k,v in self.items.items()}
    async def start(self,key,argv,log,env=None,done=None):
        async with self.lock:
            if self.active(key): raise ValueError(f'{key} is already active')
            log=Path(log); log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('ab',buffering=0) as out:
                p=await asyncio.create_subprocess_exec(*map(str,argv),stdout=out,stderr=asyncio.subprocess.STDOUT,start_new_session=True,env=env,cwd=str(log.parent))
            try: created=psutil.Process(p.pid).create_time()
            except psutil.NoSuchProcess: created=0
            item=dict(key=key,pid=p.pid,created=created,state='running',started_at=now(),log=str(log),argv=list(map(str,argv)),process=p,forced=False,returncode=None)
            self.items[key]=item; atomic_json(self.state/f'process_{key}.json',{k:v for k,v in item.items() if k!='process'})
            async def watch():
                rc=await p.wait(); item['returncode']=rc
                # ros2 run may exit before its native child. Keep ownership/state
                # until the whole process group exits; do not lose the child PID.
                if group_alive(p.pid):
                    item['state']='stopping'
                    try: await asyncio.wait_for(wait_group(p.pid),180)
                    except asyncio.TimeoutError:
                        item['forced']=True
                        try: os.killpg(p.pid,signal.SIGTERM)
                        except ProcessLookupError: pass
                        try: await asyncio.wait_for(wait_group(p.pid),10)
                        except asyncio.TimeoutError:
                            try: os.killpg(p.pid,signal.SIGKILL)
                            except ProcessLookupError: pass
                            await wait_group(p.pid)
                item['state']='cancelled' if item.get('cancel') else ('completed' if rc==0 and not item['forced'] else 'failed')
                item['ended_at']=now(); (self.state/f'process_{key}.json').unlink(missing_ok=True)
                if done:
                    try: await done(item)
                    except Exception as e: item['state']='failed'; item['error']=str(e)
            item['watcher']=asyncio.create_task(watch())
            return self.view()[key]
    async def stop(self,key,timeout=60,cancel=False):
        item=self.items.get(key)
        if not item or not self.active(key): return item
        if item['state']=='orphaned': raise ValueError(f"Orphaned PID {item['pid']}: inspect it and send SIGINT manually; automatic PID adoption is disabled")
        p=item['process']; item['state']='stopping'; item['cancel']=cancel
        try: os.killpg(p.pid,signal.SIGINT)
        except ProcessLookupError: pass
        try: await asyncio.wait_for(asyncio.shield(item['watcher']),timeout)
        except asyncio.TimeoutError:
            item['forced']=True
            try: os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: await asyncio.wait_for(asyncio.shield(item['watcher']),10)
            except asyncio.TimeoutError:
                try: os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                await item['watcher']
        await item['watcher']
        return item
