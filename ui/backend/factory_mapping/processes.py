"""Async process groups with bounded graceful shutdown and identity-safe recovery."""
import asyncio, os, signal, time
from pathlib import Path
import psutil
from .storage import atomic_json, read_json, now

class ProcessManager:
    def __init__(self, state):
        self.state=Path(state); self.state.mkdir(parents=True,exist_ok=True)
        self.items={}; self.lock=asyncio.Lock(); self.recovery=[]
        for file in self.state.glob('process_*.json'):
            old=read_json(file,{})
            try:
                p=psutil.Process(old['pid'])
                same=abs(p.create_time()-old['created'])<0.01 and p.status()!=psutil.STATUS_ZOMBIE
            except (psutil.Error,KeyError): same=False
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
                p=await asyncio.create_subprocess_exec(*map(str,argv),stdout=out,stderr=asyncio.subprocess.STDOUT,start_new_session=True,env=env)
            item=dict(key=key,pid=p.pid,created=psutil.Process(p.pid).create_time(),state='running',started_at=now(),log=str(log),argv=list(map(str,argv)),process=p,forced=False,returncode=None)
            self.items[key]=item; atomic_json(self.state/f'process_{key}.json',{k:v for k,v in item.items() if k!='process'})
            async def watch():
                rc=await p.wait(); item['returncode']=rc
                item['state']='cancelled' if item.get('cancel') else ('completed' if rc==0 else 'failed')
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
        try: await asyncio.wait_for(asyncio.shield(p.wait()),timeout)
        except asyncio.TimeoutError:
            item['forced']=True
            try: os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: await asyncio.wait_for(asyncio.shield(p.wait()),10)
            except asyncio.TimeoutError:
                try: os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                await p.wait()
        await item['watcher']
        return item
