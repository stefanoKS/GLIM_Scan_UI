import * as THREE from 'three';
import {setupCamera,refreshCamera} from '/camera.js';
import {setupCapture,refreshCapture,showPage} from '/capture.js';
import {OrbitControls} from '/vendor/OrbitControls.js';
const $=id=>document.getElementById(id);let selected=null,sessionData=[],previewWS,logWS,logKey='',configured=false,live=true,latestStatus=null,editLoadedId=null;
function error(e){$('error').hidden=false;$('error').textContent=String(e.message||e)}
async function api(path,body,method){const r=await fetch('/api/'+path,{method:method||(body?'POST':'GET'),headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});if(!r.ok){const e=await r.json();throw Error(e.detail||r.statusText)}return r}
async function json(path,body,method){return(await api(path,body,method)).json()}
const bytes=n=>n==null?'—':n>1e9?(n/1e9).toFixed(2)+' GB':(n/1e6).toFixed(1)+' MB';const hz=n=>n==null?'—':n.toFixed(1)+' Hz';const sec=n=>n==null?'—':Math.round(n)+' s';
function el(tag,text){const e=document.createElement(tag);e.textContent=text;return e}
async function refresh(){try{const s=await json('status');latestStatus=s;refreshCapture(s);$('mode').textContent=s.mock?'MOCK · simulated data':'LIVE HARDWARE MODE';const x=s.system;
$('system').textContent=`${x.model}\nCPU ${x.cpu_percent}% · RAM ${x.ram_percent}%\nGPU ${x.gpu_percent==null?'unavailable':x.gpu_percent+'%'}\nDisk free ${bytes(x.disk_free)}\n${s.config.sensor.interface}: ${s.config.sensor.host_ip||'no matching wired address'}\nROS domain ${s.config.sensor.ros_domain_id}\nTemperature ${Object.entries(x.temperatures).map(([k,v])=>k+': '+v.join('/')+'°C').join(', ')||'unavailable'}`;
const l=s.health.lidar||{},i=s.health.imu||{};$('sensor').textContent=`${s.detection?.mid360?.detected?'✓ Mid-360 detected (IP reachable)':'○ Mid-360 not detected'}\nNetwork: ${s.network.state||'checking'}\nDriver: ${s.processes.driver?.state||'stopped'}\n${s.config.sensor.points_topic}\nLiDAR ${l.state}: ${hz(l.hz)}\nPoints/s ${Math.round(l.point_rate||0)}\n${s.config.sensor.imu_topic}\nIMU ${i.state}: ${hz(i.hz)}\nLatest stamp ${l.stamp||'—'}`;
const g=s.processes.glim;$('glim').textContent=`${s.glim_available?(g?.state||'stopped'):'not installed · record-only available'} · ${s.live_preset?(['jetson_cpu','pc_dense'].includes(s.live_preset)?'CPU':'CUDA'):'—'}\nLoop detection ${s.loop_detection}\nSession ${s.active_session||'none'}\nRuntime ${g?.started_at?sec(((g.ended_at?Date.parse(g.ended_at):Date.now())-Date.parse(g.started_at))/1000):'—'}`;
$('record').textContent=`${s.processes.recording?.state||'stopped'}\nElapsed ${sec(s.recording_elapsed)}\nBag ${bytes(s.bag_size_bytes)}\n${s.active_session||'No active session'}\nFree-space time ${s.recording_elapsed>3&&s.bag_size_bytes>0?sec(x.disk_free/(s.bag_size_bytes/s.recording_elapsed)):'waiting for size samples'}`;
$('diagnostics').textContent=JSON.stringify({network:s.network,health:s.health,processes:s.processes,errors:s.errors},null,2);
if(!configured){$('preset').value=s.capture.capabilities.preset;for(const [k,type] of [['lidar_ip','text'],['interface','text'],['points_topic','text'],['imu_topic','text'],['publish_freq','number'],['ros_domain_id','number']]){const label=el('label',k);const input=document.createElement('input');input.name=k;input.type=type;input.value=k==='interface'?(s.config.sensor.interface_setting||s.config.sensor.interface):s.config.sensor[k];label.append(input);$('network').querySelector('.fields').append(label)}configured=true}
await sessions();await toolsPanel();await refreshCamera(s);if(!previewWS||previewWS.readyState>1)connectPreview();}catch(e){error(e)}}
async function sessions(){sessionData=await json('sessions');$('sessions').replaceChildren();for(const m of sessionData){const tr=document.createElement('tr');tr.dataset.id=m.id;if(m.id===selected)tr.className='selected';for(const text of [m.name+'\n'+m.created_at,sec(m.duration),bytes(m.bag_size_bytes),m.state,m.notes])tr.append(el('td',text));tr.tabIndex=0;tr.onclick=()=>{selected=m.id;logKey='';sessions().catch(error)};tr.onkeydown=e=>{if(e.key==='Enter')tr.click()};$('sessions').append(tr)}const m=sessionData.find(x=>x.id===selected);$('project-export').disabled=!m||!!latestStatus?.capture?.busy;
if(editLoadedId!==selected){$('session-rename').value=m?.name||'';$('session-notes').value=m?.notes||'';editLoadedId=selected}
const canProcess=!!m&&m.kind!=='camera'&&!!latestStatus?.glim_available&&!latestStatus?.capture?.busy;
for(const b of document.querySelectorAll('[data-action=process],#open-viewer,#open-editor,#merge-maps'))b.disabled=!canProcess;
$('selected').textContent=m?.name||'None';$('selected-job').textContent=m?.name||'None';const old=$('runs').value;$('runs').replaceChildren();for(const j of m?.processing||[]){if(j){const o=el('option',`${j.id} · ${j.preset} · ${j.state}`);o.value=j.id;$('runs').append(o)}}if([...$('runs').options].some(o=>o.value===old))$('runs').value=old;else if($('runs').options.length)$('runs').selectedIndex=$('runs').options.length-1;
$('exports').replaceChildren();for(const path of m?.exports||[]){const row=el('div',path.split('/').pop()+' ');if(path.endsWith('.ply')){const b=el('button','Preview');b.onclick=()=>showCloud(path);row.append(b);const c=el('button','Convert PCD');c.onclick=()=>json(`sessions/${selected}/pcd?path=${encodeURIComponent(path)}`,{}).then(sessions).catch(error);row.append(c)}const d=el('button','Download');d.onclick=()=>download(path);row.append(d);$('exports').append(row)}connectLogs()}
async function action(a){$('error').hidden=true;try{const out=await json('action',{action:a,session:selected,preset:$('preset').value,run:$('runs').value||null});if(a==='diagnose'){$('diagnostics').textContent=JSON.stringify(out,null,2);$('diagnostics').parentElement.open=true}await refresh()}catch(e){error(e)}}
document.querySelectorAll('[data-action]').forEach(b=>b.onclick=async()=>{b.disabled=true;await action(b.dataset.action);b.disabled=false});
$('refresh').onclick=refresh;
$('session-edit').onsubmit=async e=>{e.preventDefault();try{if(!selected)throw Error('Select a scan first');await json(`sessions/${selected}`,{name:$('session-rename').value,notes:$('session-notes').value},'PATCH');await refresh()}catch(e){error(e)}};
$('project-export').onclick=()=>{if(!selected)return;const link=document.createElement('a');link.href=`/api/sessions/${encodeURIComponent(selected)}/project`;link.click()};
$('project-import').onclick=async()=>{
 const file=$('project-file').files[0],button=$('project-import'),status=$('project-transfer-status');
 if(!file){error(Error('Select a project ZIP file'));return}
 button.disabled=true;status.textContent='Importing';$('error').hidden=true;
 try{
  const imported=await new Promise((resolve,reject)=>{
   const upload=new XMLHttpRequest();upload.open('POST','/api/projects/import');upload.setRequestHeader('Content-Type','application/zip');
   upload.upload.onprogress=event=>{if(event.lengthComputable)status.textContent=`Uploading ${Math.round(100*event.loaded/event.total)}%`};
   upload.onload=()=>{let result;try{result=JSON.parse(upload.responseText)}catch{reject(Error('Invalid import response'));return}if(upload.status>=200&&upload.status<300)resolve(result);else reject(Error(result.detail||'Import failed'))};
   upload.onerror=()=>reject(Error('Project upload failed'));upload.send(file);
  });
  selected=imported.id;logKey='';$('project-file').value='';status.textContent='Project imported';await refresh();
 }catch(e){status.textContent='';error(e)}finally{button.disabled=false}
};
$('create').onclick=async()=>{try{const m=await json('sessions',{name:$('name').value,notes:$('notes').value});selected=m.id;await refresh()}catch(e){error(e)}};
$('network').onsubmit=async e=>{e.preventDefault();const body=Object.fromEntries(new FormData(e.target));body.publish_freq=Number(body.publish_freq);body.ros_domain_id=Number(body.ros_domain_id);try{await json('network',body,'PUT');await refresh()}catch(e){error(e)}};
$('delete-derived').onclick=()=>{if(confirm('Delete this generated processing run? The raw bag will be retained.'))action('delete_derived')};
$('quality').onclick=async()=>{try{const q=await json(`sessions/${selected}/quality/${$('runs').value}`);const m=sessionData.find(x=>x.id===selected);const ply=m?.exports.filter(x=>x.endsWith('.ply')).at(-1);if(ply)Object.assign(q,await json(`sessions/${selected}/cloud_stats?path=${encodeURIComponent(ply)}`));$('quality-result').textContent=JSON.stringify(q,null,2)}catch(e){error(e)}};
async function download(path){try{const r=await api(`sessions/${selected}/artifact?path=${encodeURIComponent(path)}`);const u=URL.createObjectURL(await r.blob());const a=el('a','');a.href=u;a.download=path.split('/').pop();a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}catch(e){error(e)}}
$('download-log').onclick=()=>download(`processing/${$('runs').value}/job.log`);
function socket(path){return new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}${path}`)}
function connectLogs(){const key=selected+'/'+$('runs').value;if(key===logKey)return;logWS?.close();logKey=key;$('logs').textContent='';if(!selected||!$('runs').value)return;logWS=socket('/ws/logs/'+key);logWS.onmessage=e=>{try{if(JSON.parse(e.data).job)return}catch{}$('logs').textContent=($('logs').textContent+e.data).slice(-60000);$('logs').scrollTop=$('logs').scrollHeight}}$('runs').onchange=()=>{logKey='';connectLogs()};
const scene=new THREE.Scene();scene.background=new THREE.Color('#080f17');const camera=new THREE.PerspectiveCamera(55,1,.05,10000);camera.up.set(0,0,1);camera.position.set(8,-10,8);const renderer=new THREE.WebGLRenderer({antialias:false});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));$('canvas').append(renderer.domElement);const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
const viewRotation=new THREE.Quaternion();
const grid=new THREE.GridHelper(30,30,0x395366,0x1d303f);grid.rotation.x=Math.PI/2;scene.add(grid,new THREE.AxesHelper(2));const material=new THREE.PointsMaterial({size:2,sizeAttenuation:false,vertexColors:true});const points=new THREE.Points(new THREE.BufferGeometry(),material);scene.add(points);
function applyViewRotation(){points.quaternion.copy(live?viewRotation:new THREE.Quaternion())}
async function loadViewRotation(){const r=await json('preview/orientation');viewRotation.fromArray(r.quaternion);applyViewRotation();$('orientation-status').textContent=r.saved_at?'View orientation saved. Reorient if the mounting angle changes.':'Using the original sensor orientation.'}
for(const [id,method] of [['orient-lidar','POST'],['reset-lidar-orientation','DELETE']])$(id).onclick=async()=>{const buttons=[$('orient-lidar'),$('reset-lidar-orientation')];buttons.forEach(b=>b.disabled=true);$('orientation-status').textContent=method==='POST'?'Applying orientation...':'Resetting…';try{await json('preview/orientation',method==='POST'?{}:null,method);await loadViewRotation();reset()}catch(e){$('orientation-status').textContent=e.message;error(e)}finally{buttons.forEach(b=>b.disabled=false)}};
loadViewRotation().catch(error);
function renderCloud(buffer){applyViewRotation();const d=new DataView(buffer);if(d.byteLength<16||d.getUint32(0,true)!==0x43504d46)return;const n=d.getUint32(4,true);if(n>100000||buffer.byteLength!==16+n*16)return;const xyz=new Float32Array(n*3),colors=new Float32Array(n*3);for(let i=0;i<n;i++){for(let j=0;j<3;j++)xyz[3*i+j]=d.getFloat32(16+16*i+4*j,true);const v=Math.min(1,Math.max(0,d.getFloat32(28+16*i,true)/255));colors[3*i]=.2+.8*v;colors[3*i+1]=.45+.5*v;colors[3*i+2]=1-.65*v}const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(xyz,3));geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));points.geometry.dispose();points.geometry=geometry;$('viewer-note').textContent=`${n.toLocaleString()} displayed points · ${live?'Raw sensor frame':'Optimized export preview'} · Drag / right-drag / scroll to orbit / pan / zoom`}
function connectPreview(){previewWS=socket('/ws/preview');previewWS.binaryType='arraybuffer';previewWS.onmessage=e=>{if(live&&e.data instanceof ArrayBuffer)renderCloud(e.data)}}
async function showCloud(path){try{const r=await api(`sessions/${selected}/cloud?path=${encodeURIComponent(path)}`);live=false;showPage('capture');renderCloud(await r.arrayBuffer());$('preview-label').textContent='Map preview';reset()}catch(e){error(e)}}
$('live-preview').onclick=()=>{live=true;applyViewRotation();$('preview-label').textContent='LiDAR view'};
function reset(){points.geometry.computeBoundingSphere();const sphere=points.geometry.boundingSphere;const center=(sphere?.center||new THREE.Vector3()).clone().applyQuaternion(points.quaternion);const radius=Math.max(5,Math.min(1000,sphere?.radius||10));controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(radius,-radius,radius));controls.update()}
$('reset').onclick=reset;$('point-size').oninput=e=>material.size=Number(e.target.value);
new ResizeObserver(()=>{const w=$('canvas').clientWidth,h=$('canvas').clientHeight;if(!w||!h)return;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix()}).observe($('canvas'));
setupCamera({json,refresh,error});
setupCapture({json,refresh,error,selectSession:id=>{selected=id;logKey=''}});
renderer.setAnimationLoop(()=>{if($('capture-page').hidden)return;controls.update();renderer.render(scene,camera)});refresh();let polling=false;setInterval(async()=>{if(!polling){polling=true;await refresh();polling=false}},2500);

async function toolsPanel(){
 const catalog=await json('tools');$('tool-catalog').replaceChildren();
 for(const t of catalog)$('tool-catalog').append(el('p',`${t.label}: ${t.installed?'installed':'not built'}${t.display&&!t.display_available?' · no server display':''}. ${t.features.join(' · ')}`));
 const old=[...$('merge-inputs').selectedOptions].map(o=>o.value);$('merge-inputs').replaceChildren();
 for(const m of sessionData)for(const r of m.processing||[])if(r?.state==='completed'&&!(m.id===selected&&r.id===$('runs').value)){
  const option=el('option',m.name+' / '+r.id);option.value=JSON.stringify({session:m.id,run:r.id});option.selected=old.includes(option.value);$('merge-inputs').append(option);
 }
 $('edit-workspaces').replaceChildren();if(!selected)return;
 for(const edit of await json(`sessions/${selected}/edits`)){
  const row=el('div',edit.id+' · '+edit.tool+' · '+edit.state+' ');
  const info=el('button','Show workspace');info.onclick=()=>$('tool-workspace').textContent=JSON.stringify(edit,null,2);row.append(info);
  const exportButton=el('button','Export saved edited map');exportButton.onclick=async()=>{try{await json('action',{action:'export_edit',session:selected,run:edit.id});refresh()}catch(e){error(e)}};row.append(exportButton);
  const log=el('button','Download tool log');log.onclick=()=>download(`edits/${edit.id}/tool.log`);row.append(log);$('edit-workspaces').append(row);
 }
}
async function openTool(tool,merge=false){try{if(merge&&!$('merge-inputs').selectedOptions.length)throw Error('Select at least one additional processed map to merge');const result=await json('tools/open',{session:selected,run:$('runs').value,tool,additional:merge?[...$('merge-inputs').selectedOptions].map(o=>JSON.parse(o.value)):[]});$('tool-workspace').textContent=JSON.stringify(result,null,2);await refresh()}catch(e){error(e)}}
$('merge-maps').onclick=()=>openTool('offline_viewer',true);
$('open-viewer').onclick=()=>openTool('offline_viewer');$('open-editor').onclick=()=>openTool('map_editor');$('stop-tool').onclick=()=>{if(confirm('Stop the native tool? Save your edits in its window first; unsaved edits may be lost.'))action('tool_stop')};
