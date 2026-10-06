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
    async def recover(self,key,timeout=20,terminate_timeout=10):
        """Explicitly stop a saved process group; never signal a reused PID."""
        item=self.items.get(key)
        if not item or item['state']!='orphaned': return await self.stop(key,timeout,cancel=True)
        pgid=item['pid']
        identities={pgid:item['created']}
        def verify():
            if pgid<=1 or pgid==os.getpgrp(): raise ValueError('Unsafe recovery process group')
            members=[]
            for proc in psutil.process_iter(['pid','create_time','status','uids']):
                try:
                    if os.getpgid(proc.pid)==pgid and proc.info['status']!=psutil.STATUS_ZOMBIE:
                        members.append(proc)
                except (ProcessLookupError,psutil.NoSuchProcess): pass
            if not members: return False
            try:
                leader=psutil.Process(pgid)
                leader_matches=abs(leader.create_time()-item['created'])<.01 and os.getpgid(pgid)==pgid
            except (ProcessLookupError,psutil.NoSuchProcess): leader_matches=False
            if (any(p.info['uids'].real!=os.getuid() for p in members) or
                not (leader_matches or any(p.pid in identities and abs(p.info['create_time']-identities[p.pid])<.01 for p in members))):
                raise ValueError(f'Cannot verify ownership of {key} (group {pgid}); no signal sent. Inspect Diagnostics.')
            # Retain verified descendants if the launcher exits during shutdown.
            identities.update({p.pid:p.info['create_time'] for p in members})
            return True
        for sig,delay in ((signal.SIGINT,timeout),(signal.SIGTERM,terminate_timeout),(signal.SIGKILL,5)):
            if not verify(): break
            if sig!=signal.SIGINT: item['forced']=True
            try: os.killpg(pgid,sig)
            except ProcessLookupError: break
            try:
                await asyncio.wait_for(wait_group(pgid),delay)
                break
            except asyncio.TimeoutError: pass
        else:
            raise ValueError(f'{key} is still running after recovery; inspect Diagnostics')
        item.update(state='cancelled',ended_at=now(),returncode=None,cancel=True)
        (self.state/f'process_{key}.json').unlink(missing_ok=True)
        return item

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
        if item['state']=='orphaned': raise ValueError(f"Orphaned PID {item['pid']}: use Process recovery in Settings → Advanced / Diagnostics")
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
