import {cameraLabel,captureMessages} from './camera-status.js';
// The browser requests semantic actions; the server owns sequencing and recovery.
const $=id=>document.getElementById(id);
let ctx, latest, actionPending=false, settingsLoaded=false;
export function setupCapture(context){
 ctx=context;
 document.querySelectorAll('[data-page]').forEach(button=>button.onclick=()=>showPage(button.dataset.page));
 window.addEventListener('hashchange',()=>showPage(location.hash.slice(1),false));
 showPage(location.hash.slice(1)||'capture',false);
 document.querySelectorAll('[data-capture]').forEach(button=>button.onclick=async()=>{
  if(actionPending)return;actionPending=true;updateButtons();$('error').hidden=true;
  try{const result=await ctx.json('capture/action',{action:button.dataset.capture});if(result.session)ctx.selectSession(result.session);await ctx.refresh()}catch(e){ctx.error(e)}finally{actionPending=false;updateButtons()}
 });
 $('capture-settings').onsubmit=async e=>{e.preventDefault();try{await ctx.json('capture/settings',{live_glim:$('live-glim-setting').checked,auto_process:$('auto-process-setting').checked,mapping_preset:$('mapping-preset-setting').value},'PUT');settingsLoaded=false;$('preset').value=$('mapping-preset-setting').value==='auto'?latest.capture.capabilities.gpu?'jetson_gpu':'jetson_cpu':$('mapping-preset-setting').value;await ctx.refresh()}catch(e){ctx.error(e)}};
}
export function showPage(name,hash=true){
 if(!['capture','library','calibration','settings'].includes(name))name='capture';
 document.querySelectorAll('[data-page-panel]').forEach(p=>p.hidden=p.dataset.pagePanel!==name);
 document.querySelectorAll('[data-page]').forEach(b=>{if(b.dataset.page===name)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current')});
 if(hash)history.replaceState(null,'','#'+name);
}
function updateButtons(){
 if(!latest)return;const c=latest.capture,scanning=['PREFLIGHT','SCANNING','FINALIZING'].includes(c.state),isCamera=c.mode==='camera';
 $('start-scan').hidden=scanning&&!isCamera;$('stop-scan').hidden=!scanning||isCamera;
 $('record-camera').hidden=scanning&&isCamera;$('stop-camera-recording').hidden=!scanning||!isCamera;
 $('start-scan').disabled=actionPending||!c.can_start;$('record-camera').disabled=actionPending||!c.can_start||!c.capabilities.camera;
 $('stop-scan').disabled=$('stop-camera-recording').disabled=actionPending||c.state==='FINALIZING';
}
export function refreshCapture(s){
 latest=s;const c=s.capture;if(!c)return;const capabilities=c.capabilities,h=s.health;
 $('rgb-recording-warning').hidden=!!s.config.system.camera.enabled;
 const set=(id,label,ok,text)=>{$(id).textContent=`${ok?'✓':'○'} ${label} · ${text}`;$(id).dataset.ready=String(ok)};
 for(const [key,label] of [['lidar','LiDAR'],['imu','IMU']]){const healthy=['healthy','mock'].includes(h[key]?.state);set('health-'+key,label,healthy,healthy?'Ready':key==='lidar'&&s.detection?.mid360?.detected?'Connected':'Waiting')}
 const cam=h.camera||{}, label=cameraLabel(s);set('health-camera','Camera',label.ready,label.text);
 $('health-camera').title=s.camera_selection?.fallback_reason||'';
 const free=s.system.disk_free/1e9;set('health-storage','Storage',free>=s.config.system.storage.minimum_free_gb,`${free.toFixed(0)} GB free`);
 const titles={READY:'Ready to scan',PREFLIGHT:'Checking sensors…',SCANNING:c.mode==='camera'?'Recording camera':'Scanning',FINALIZING:'Saving recording…',PROCESSING:'Processing scan…',COMPLETE:'Ready for next scan'};
 $('capture-state').textContent=titles[c.state]||c.state;
 const hints={READY:'Press Start Scan to begin.',PREFLIGHT:'Starting the required sensors and checking their data.',SCANNING:'Raw data is being saved.',FINALIZING:'Keep the application open while recording finishes.',PROCESSING:'The raw scan is safe. Building the map.',COMPLETE:capabilities.record_only?'Scan saved. Export it from Library to process on a workstation.':'Open Library to view, rename or export the recording.'};
 const blocked=!c.busy&&!c.can_start?(Object.values(s.processes).some(p=>p.state==='orphaned')?'A previous sensor process is still running. Review recovery in Settings → Advanced / Diagnostics.':'Finish the active session or job before starting another capture.'):null;
 const messages=captureMessages(c,blocked,hints[c.state]);
 $('capture-message').textContent=messages.current;
 $('previous-capture').textContent=messages.previous;
 const seconds=Math.floor(s.recording_elapsed||0);$('capture-elapsed').textContent=`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;
 $('pip-empty').hidden=!!cam.camera_running;
 $('system-profile').textContent=`${capabilities.profile} · ${capabilities.record_only?'Recording only':capabilities.gpu?'GPU and CPU processing':'CPU processing'}`;
 $('library-capability').textContent=capabilities.record_only?'Export recorded scans here, then import them on the processing workstation.':'Import Jetson projects or select a scan to process and edit.';
 if(!settingsLoaded){$('mapping-preset-setting').value=capabilities.settings.mapping_preset||'auto';$('live-glim-setting').checked=capabilities.settings.live_glim;$('auto-process-setting').checked=capabilities.settings.auto_process;settingsLoaded=true}
 for(const option of $('mapping-preset-setting').options)option.disabled=['jetson_gpu','offline_quality'].includes(option.value)&&!capabilities.gpu;
 $('mapping-preset-setting').disabled=!capabilities.processing||c.busy;
 $('live-glim-setting').disabled=$('auto-process-setting').disabled=!capabilities.processing||c.busy;
 updateButtons();
}
