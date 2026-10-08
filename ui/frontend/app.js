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
async function refresh(){try{const s=await json('status');latestStatus=s;refreshCapture(s);$('mode').textContent=s.mock?'MOCK · simulated data':'LIVE HARDWARE MODE';$('mode').dataset.mode=s.mock?'mock':'live';const x=s.system;
$('system').textContent=`${x.model}\nCPU ${x.cpu_percent}% · RAM ${x.ram_percent}%\nGPU ${x.gpu_percent==null?'unavailable':x.gpu_percent+'%'}\nDisk free ${bytes(x.disk_free)}\n${s.config.sensor.interface}: ${s.config.sensor.host_ip||'no matching wired address'}\nROS domain ${s.config.sensor.ros_domain_id}\nTemperature ${Object.entries(x.temperatures).map(([k,v])=>k+': '+v.join('/')+'°C').join(', ')||'unavailable'}`;
const l=s.health.lidar||{},i=s.health.imu||{};$('sensor').textContent=`${s.detection?.mid360?.detected?'✓ Mid-360 detected (IP reachable)':'○ Mid-360 not detected'}\nNetwork: ${s.network.state||'checking'}\nDriver: ${s.processes.driver?.state||'stopped'}\n${s.config.sensor.points_topic}\nLiDAR ${l.state}: ${hz(l.hz)}\nPoints/s ${Math.round(l.point_rate||0)}\n${s.config.sensor.imu_topic}\nIMU ${i.state}: ${hz(i.hz)}\nLatest stamp ${l.stamp||'—'}`;
const g=s.processes.glim;$('glim').textContent=`${s.glim_available?(g?.state||'stopped'):'not installed · record-only available'} · ${s.live_preset?(['jetson_cpu','pc_dense'].includes(s.live_preset)?'CPU':'CUDA'):'—'}\nLoop detection ${s.loop_detection}\nSession ${s.active_session||'none'}\nRuntime ${g?.started_at?sec(((g.ended_at?Date.parse(g.ended_at):Date.now())-Date.parse(g.started_at))/1000):'—'}`;
$('record').textContent=`${s.processes.recording?.state||'stopped'}\nElapsed ${sec(s.recording_elapsed)}\nBag ${bytes(s.bag_size_bytes)}\n${s.active_session||'No active session'}\nFree-space time ${s.recording_elapsed>3&&s.bag_size_bytes>0?sec(x.disk_free/(s.bag_size_bytes/s.recording_elapsed)):'waiting for size samples'}`;
$('diagnostics').textContent=JSON.stringify({network:s.network,health:s.health,processes:s.processes,errors:s.errors},null,2);
if(!configured){$('preset').value=s.capture.capabilities.preset;for(const [k,type] of [['lidar_ip','text'],['interface','text'],['points_topic','text'],['imu_topic','text'],['publish_freq','number'],['ros_domain_id','number']]){const label=el('label',k);const input=document.createElement('input');input.name=k;input.type=type;input.value=k==='interface'?(s.config.sensor.interface_setting||s.config.sensor.interface):s.config.sensor[k];label.append(input);$('network').querySelector('.fields').append(label)}configured=true}
await sessions();await toolsPanel();await refreshCamera(s);if(!previewWS||previewWS.readyState>1)connectPreview();}catch(e){error(e)}}
async function sessions(){sessionData=await json('sessions');$('sessions').replaceChildren();for(const m of sessionData){const tr=document.createElement('tr');tr.dataset.id=m.id;tr.dataset.state=String(m.state||'').toLowerCase();if(m.id===selected)tr.className='selected';for(const text of [m.name+'\n'+m.created_at,sec(m.duration),bytes(m.bag_size_bytes),m.state,m.notes])tr.append(el('td',text));tr.tabIndex=0;tr.onclick=()=>{selected=m.id;logKey='';sessions().catch(error)};tr.onkeydown=e=>{if(e.key==='Enter')tr.click()};$('sessions').append(tr)}const m=sessionData.find(x=>x.id===selected);$('project-export').disabled=!m||!!latestStatus?.capture?.busy;
if(editLoadedId!==selected){$('session-rename').value=m?.name||'';$('session-notes').value=m?.notes||'';editLoadedId=selected}
const canProcess=!!m&&m.kind!=='camera'&&!!latestStatus?.glim_available&&!latestStatus?.capture?.busy;
for(const b of document.querySelectorAll('[data-action=process],#open-viewer,#open-editor,#merge-maps'))b.disabled=!canProcess;
$('selected').textContent=m?.name||'None';$('selected-job').textContent=m?.name||'None';const old=$('runs').value;$('runs').replaceChildren();for(const j of m?.processing||[]){if(j){const o=el('option',`${j.id} · ${j.preset} · ${j.state}`);o.value=j.id;$('runs').append(o)}}if([...$('runs').options].some(o=>o.value===old))$('runs').value=old;else if($('runs').options.length)$('runs').selectedIndex=$('runs').options.length-1;
$('exports').replaceChildren();for(const path of m?.exports||[]){const row=el('div',path.split('/').pop()+' ');if(path.endsWith('.ply')){const b=el('button','Preview');b.onclick=()=>showCloud(path);row.append(b);const c=el('button','Convert PCD');c.onclick=()=>json(`sessions/${selected}/pcd?path=${encodeURIComponent(path)}`,{}).then(sessions).catch(error);row.append(c)}const d=el('button','Download');d.onclick=()=>download(path);row.append(d);$('exports').append(row)}connectLogs();await reconstructionPanel();await colorizationPanel()}
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

let reconstructionPending=false,reconstructionData=null,reconstructionSession=null;
// Individual mesh links stay bounded; the manifest lists every cell.
const MESH_LINK_LIMIT=24,meshFileCache=new Map();
// Stable engine identifiers shared with the API. VDBFusion is preferred when available.
const VDBFUSION_PRESETS={fast:[0.02,0.06],detailed:[0.01,0.03],experimental:[0.005,0.015]};
function algorithm(){return $('algorithm-nksr').checked?'nksr':'vdbfusion'}
function vdbfusionAvailable(){const h=reconstructionData?.vdbfusion;return !!h&&h.status!=='VDBFUSION_NOT_INSTALLED'&&h.status!=='RECORD_ONLY'}
function roiValues(){
 const parse=value=>{const parts=String(value).trim().split(/[\s,]+/).filter(Boolean);if(!parts.length)return null;
  if(parts.length!==3)throw Error('A region of interest needs exactly three world coordinates in meters');
  const numbers=parts.map(Number);if(numbers.some(v=>!Number.isFinite(v)))throw Error('Region-of-interest coordinates must be finite meters');return numbers};
 const min=parse($('vdbfusion-roi-min').value),max=parse($('vdbfusion-roi-max').value);
 if((min&&!max)||(max&&!min))throw Error('Provide both a region-of-interest minimum and maximum corner');
 return {roi_min_m:min,roi_max_m:max};
}
function vdbfusionSettings(){
 const voxel=Number($('vdbfusion-voxel').value),trunc=Number($('vdbfusion-trunc').value);
 if(!(voxel>0))throw Error('TSDF voxel size must be a positive size in meters');
 if(!(trunc>0))throw Error('TSDF truncation distance must be a positive distance in meters');
 if(trunc<voxel)throw Error('TSDF truncation distance must be at least the voxel size');
 const optional=(id)=>{const raw=$(id).value.trim();if(!raw)return null;const value=Number(raw);if(!Number.isFinite(value))throw Error('Enter a finite number or leave the field empty');return value};
 const preset=$('vdbfusion-preset').value;
 return {preset:preset||null,voxel_size_m:voxel,sdf_trunc_m:trunc,space_carving:$('vdbfusion-space-carving').checked,
  mesh_output_mode:$('vdbfusion-output-mode').value,...roiValues(),
  origin_error_budget_m:optional('vdbfusion-origin-budget'),
  association_spacing_multiplier:optional('vdbfusion-association-multiplier'),
  boundary_margin_m:optional('vdbfusion-boundary-margin'),
  unsupported_observations:$('vdbfusion-unsupported').value,
  mask_deleted_triangles:$('vdbfusion-mask-triangles').checked,
  memory_budget_gib:optional('vdbfusion-memory-budget')};
}
function presetLabel(){const preset=$('vdbfusion-preset').value;const note=VDBFUSION_PRESETS[preset];
 return preset==='experimental'?'Experimental 5 mm TSDF: whole-factory scans can need very large memory and long runtimes.'
  :(note?`${preset} preset: ${(note[0]*1000).toFixed(0)} mm voxel, ${(note[1]*1000).toFixed(0)} mm truncation.`:'Custom TSDF settings.')}
function nksrSettings(){
 return {preparation_voxel_size_m:Number($('voxel-size-cm').value)/100.0,mode:$('nksr-mode').value,device:$('nksr-device').value,
  mesh_output_mode:$('nksr-output-mode').value,
  detail_level:Number($('nksr-detail').value),chunk_size:$('nksr-chunk').value?Number($('nksr-chunk').value):null,
  normal_knn:Number($('nksr-knn').value),normal_drop_angle_deg:Number($('nksr-angle').value),mise_iter:Number($('nksr-mise').value)};
}
function renderReconstruction(){
 const data=reconstructionData,job=data?.jobs.find(j=>j.id===$('nksr-input-run').value),meta=job?.metadata;
 const engine=algorithm(),engineJob=meta?.algorithm||job?.algorithm||'nksr';
 // Each engine owns its own managed process, so busy state is engine specific.
 const engineBusy=['running','stopping','orphaned'].includes(latestStatus?.processes?.[engine]?.state);
 const busy=reconstructionPending||engineBusy||
  ['running','stopping','orphaned'].includes(latestStatus?.processes?.reconstruction?.state);
 const filter=$('filter-edited-geometry').checked,source=$('edited-source').value,tolerance=Number($('edited-tolerance').value);
 const needsTolerance=filter&&engine==='nksr';
 $('edited-source-label').hidden=!filter;$('edited-tolerance-label').hidden=!needsTolerance;$('edited-filter-note').hidden=!filter;
 $('reconstruction-trajectory').disabled=filter;$('edited-source').disabled=!filter;$('edited-tolerance').disabled=!needsTolerance;
 $('vdbfusion-advanced').hidden=engine!=='vdbfusion';$('nksr-advanced').hidden=engine!=='nksr';
 $('vdbfusion-preset-note').textContent=presetLabel();
 $('edited-filter-note').textContent=engine==='nksr'
  ?'Approximate retained-geometry filtering. It does not prevent NKSR from bridging deleted regions.'
  :'VDBFusion classifies every observation against the verified kept and removed submap geometry, then masks any triangle inside a removed region. It is an approximate spatial association, not an exact edit transfer.';
 // Edit status (GATE 7E): verified saved edit, filtering strategy and retained share.
 const editSource=data?.edited_sources.find(edit=>edit.id===source);
 const editAccuracy=meta?.edit_filter_accuracy||job?.metadata?.edit_filter_accuracy||null;
 $('edited-status').hidden=!filter||!selected;
 if(filter)$('edited-status').textContent=
  (editSource?`Saved edit verified${editSource.export_ready?'':' · export required'}`:'Select a saved cleanup')+
  (meta?.edited_geometry?` · Filtering: ${editAccuracy==='validated_approximate'?'validated approximate':(editAccuracy||'unavailable')} · cleanup retains ${((meta.edited_geometry.retained_ratio||0)*100).toFixed(1)}% of submap samples`:'')+
  (editAccuracy==='validated_approximate'?'':' · Warning: large areas may have insufficient coverage');
 // Never silently substitute an engine: unavailable VDBFusion blocks VDBFusion jobs.
 const vdb=data?.vdbfusion;
 $('vdbfusion-availability').textContent=vdb?`VDBFusion: ${vdb.status}${vdb.vdbfusion_version?' · v'+vdb.vdbfusion_version:''}${vdb.message?' · '+vdb.message:''}`:'VDBFusion: select a scan to check availability';
 const sourceSettings=job?.edited_geometry_source||{},sourceStale=filter!==!!job?.filter_edited_geometry||
  (filter&&(sourceSettings.edit_id!==source||(engine==='nksr'&&Math.abs((sourceSettings.tolerance_m||0)-tolerance)>1e-12)));
 const preparationVoxel=Number($('voxel-size-cm').value)/100;
 const prepared=job?.state==='PREPARED',stale=prepared&&(engine!==engineJob||
  (engine==='nksr'&&Math.abs((meta.voxel_size_m||0)-preparationVoxel)>1e-12)||
  (!filter&&job.trajectory!==$('reconstruction-trajectory').value)||sourceStale);
 $('reconstruction-results').replaceChildren();$('nksr-mesh-results').replaceChildren();
 const preparing=data?.jobs.find(j=>j.state==='PREPARING');
 $('reconstruction-status').textContent=preparing?`PREPARING · ${preparing.progress}`:stale?'Prepared input is stale. Prepare again with the selected algorithm, trajectory and settings.':prepared?`PREPARED · ${engineJob} · ready for mesh reconstruction.`:data?.jobs.at(-1)?.state||'NOT_PREPARED';
 $('reconstruction-status').dataset.state=preparing?'preparing':stale?'stale':prepared?'prepared':String(data?.jobs.at(-1)?.state||'').toLowerCase();
 if(prepared&&meta?.points_after_voxel!==undefined){
  const label=meta.voxel_size_m===0?'sampling disabled':`after ${(meta.voxel_size_m*100).toFixed(1)} cm voxel sampling`;
  const filtered=meta.filter_enabled?` · ${meta.points_before_filter.toLocaleString()} raw → ${meta.points_after_filter.toLocaleString()} retained before sampling`:'';
  $('reconstruction-results').append(el('p',`Points: ${meta.points_before_voxel.toLocaleString()} ${label}${filtered}`));
  for(const [name,file] of [['Download NKSR input','input/nksr_input.npz'],['Preview input','validation/reconstructed_from_bag.ply'],['Metadata','validation/comparison.json']]){
   const button=el('button',name),path=`reconstruction/${job.id}/${file}`;
   button.onclick=()=>name==='Preview input'?showCloud(path):download(path);$('reconstruction-results').append(button);
  }
 }
 if(prepared&&engineJob==='vdbfusion'&&meta?.settings){
  const scan=meta.preflight||{},settings=meta.settings;
  $('reconstruction-results').append(el('p',`VDBFusion source validation: ${(meta.lidar_frames||0).toLocaleString()} LiDAR frames, about ${(meta.points_before_voxel||0).toLocaleString()} raw observations (sampled estimate). No prepared point cloud is written; the raw bag is streamed during reconstruction.`));
  $('reconstruction-results').append(el('p',`TSDF: ${(settings.voxel_size_m*1000).toFixed(0)} mm voxel, ${(settings.sdf_trunc_m*1000).toFixed(0)} mm truncation, space carving ${settings.space_carving?'on':'off'}.`));
  if(scan.observed_extent_m)$('reconstruction-results').append(el('p',`Sampled extent: ${scan.observed_extent_m.map(v=>v.toFixed(1)).join(' × ')} m · free RAM ${(scan.free_ram_bytes/1073741824).toFixed(1)} GiB · free disk ${(scan.free_disk_bytes/1073741824).toFixed(1)} GiB`));
  for(const warning of scan.warnings||[])$('reconstruction-results').append(el('p',`Warning: ${warning}`));
  if(meta.edit_filter_accuracy){
   const edit=meta.edited_geometry||{};
   $('reconstruction-results').append(el('p',`Filtering: ${meta.edit_filter_accuracy==='validated_approximate'?'validated approximate':'exact'} · saved cleanup keeps ${((edit.retained_ratio||0)*100).toFixed(1)}% of ${(edit.reference_points||0).toLocaleString()} submap samples · association radius ${(edit.association_radius_m||0).toFixed(3)} m`));
  }
  for(const [name,file] of [['Download preparation metadata','input/vdbfusion_prepare.json']]){
   const button=el('button',name);button.onclick=()=>download(`reconstruction/${job.id}/${file}`);$('reconstruction-results').append(button);
  }
 }
 const h=data?.nksr;
 $('nksr-health').textContent=h?`NKSR: ${h.status}${h.gpu_name?' · GPU: '+h.gpu_name:''}${h.message?' · '+h.message:''}`:'NKSR: select a scan to check availability';
 $('nksr-check').disabled=!h||h.status==='NKSR_NOT_INSTALLED'||h.status==='CHECKING'||busy;
 $('vdbfusion-check').disabled=!vdb||vdb.status==='VDBFUSION_NOT_INSTALLED'||vdb.status==='CHECKING'||busy;
 $('vdbfusion-health').hidden=engine!=='vdbfusion';$('vdbfusion-check').hidden=engine!=='vdbfusion';
 $('nksr-health').hidden=engine!=='nksr';$('nksr-check').hidden=engine!=='nksr';
 const nksrReady=h?.smoke_passed&&(h.status==='READY'||(h.cpu_ready&&$('nksr-device').value!=='cuda'));
 const ready=engine==='vdbfusion'?(!!vdb&&vdb.smoke_passed&&vdb.status==='READY'):!!nksrReady;
 $('reconstruct-mesh').disabled=!prepared||stale||!ready||busy||!!preparing||!!latestStatus?.capture?.busy;
 const lowRam=$('nksr-mode').value==='low_ram';
 $('nksr-detail').disabled=lowRam||$('nksr-mode').value==='chunked';
 if($('nksr-chunk').value&&!$('nksr-chunk').checkValidity())$('reconstruct-mesh').disabled=true;
 $('nksr-mode-note').textContent=lowRam?'Independent tile boundaries may contain gaps or overlaps. No boundary stitching. Chunk size is the independent tile edge in meters.':
  'Auto selects full or chunked inference from point count and available GPU memory. Detail level applies only to full mode. Chunked extraction uses CPU. CPU inference can be very slow.';
 const mesh=job?.mesh;
 $('cancel-mesh').hidden=mesh?.state!=='RUNNING';
 // The progress file and the job record can carry the same sentence; show it once.
 const meshDetail=[mesh?.message,mesh?.progress?.message].filter((t,i,a)=>t&&a.indexOf(t)===i).join(' · ');
 $('nksr-mesh-status').textContent=mesh?`${mesh.stage||mesh.state}${meshDetail?' · '+meshDetail:''}`:'NOT_RECONSTRUCTED';
 $('nksr-mesh-status').dataset.state=mesh?String(mesh.state||'').toLowerCase():'';
 if(mesh?.state==='COMPLETED'&&(mesh.metadata?.engine==='vdbfusion'||engineJob==='vdbfusion')){
  const m=mesh.metadata||{},integration=m.integration||{},preflight=m.preflight||{};
  $('nksr-mesh-results').append(el('p',`VDBFusion mesh · one fused TSDF · ${(m.face_count||0).toLocaleString()} triangles · ${(m.vertex_count||0).toLocaleString()} vertices${m.output_bytes!=null?` · ${(m.output_bytes/1048576).toFixed(1)} MiB on disk`:''}`));
  if(m.bounding_box_min)$('nksr-mesh-results').append(el('p',`Bounds: ${m.bounding_box_min.map(v=>v.toFixed(2)).join(', ')} → ${m.bounding_box_max.map(v=>v.toFixed(2)).join(', ')} m · world meters, no local origin shift`));
  $('nksr-mesh-results').append(el('p',`Integrated ${(integration.integrated_observations||0).toLocaleString()} of ${(integration.raw_observations||0).toLocaleString()} raw observations in ${(integration.origin_groups||0).toLocaleString()} motion-aware origin groups · max origin approximation error ${((integration.max_origin_error_m||0)*1000).toFixed(1)} mm of the ${((integration.origin_error_budget_m||0)*1000).toFixed(1)} mm budget`));
  if(integration.edit_filter_enabled){
   $('nksr-mesh-results').append(el('p',`Filtering: ${integration.edit_filter_accuracy==='validated_approximate'?'validated approximate':'exact'} · retained ${(integration.points_retained||0).toLocaleString()} of ${(integration.points_before_filter||0).toLocaleString()} observations (${((integration.filter_retention_ratio||0)*100).toFixed(2)}%) · removed-support dominated ${(integration.points_dropped_removed_support||0).toLocaleString()} · unsupported ${(integration.points_dropped_unsupported||0).toLocaleString()} (${integration.unsupported_observations}) · masked triangles ${(m.extraction?.masked_triangles||0).toLocaleString()}`));
   $('nksr-mesh-results').append(el('p',`Filtering limitation: ${integration.edit_reference?.limitation||''}`));
  }
  $('nksr-mesh-results').append(el('p',`Elapsed ${(m.elapsed_seconds||0).toFixed(1)} s · peak worker RSS ${((integration.peak_rss_bytes||0)/1048576).toFixed(0)} MiB · estimated TSDF footprint ${(preflight.estimated_tsdf_bytes/1073741824).toFixed(1)} GiB`));
  if(m.validation_note)$('nksr-mesh-results').append(el('p',m.validation_note));
  if(m.mesh_output_mode!=='chunks'){
   const button=el('button','Download merged mesh');button.onclick=()=>download(`reconstruction/${job.id}/output/mesh.ply`);$('nksr-mesh-results').append(button);
  }
  const metadata=el('button','Download VDBFusion metadata');metadata.onclick=()=>download(`reconstruction/${job.id}/output/vdbfusion_metadata.json`);$('nksr-mesh-results').append(metadata);
  if(m.mesh_output_mode!=='merged'&&m.chunk_count!=null){
   $('nksr-mesh-results').append(el('p',`${m.chunk_count} export cells at ${m.chunk_size_m} m (${m.chunk_size_source})`));
   const manifest=el('button','Download cells manifest');manifest.onclick=()=>download(`reconstruction/${job.id}/output/mesh_chunks/chunks.json`);$('nksr-mesh-results').append(manifest);
  }
  if(job){const log=el('button','Preparation / reconstruction log');log.onclick=()=>download(`reconstruction/${job.id}/job.log`);$('reconstruction-results').append(log)}
  return;
 }
 if(mesh?.state==='COMPLETED'){
  const m=mesh.metadata,lowRamResult=m.actual_mode==='low_ram';
  const modeLabel=lowRamResult?`Low RAM · ${m.completed_tiles}/${m.tile_count} independent tiles`:m.actual_mode;
  const outputLabel={merged:'single mesh',chunks:'separate meshes',both:'single mesh + separate meshes'}[m.mesh_output_mode]||m.mesh_output_mode;
  const sectionLabel=lowRamResult?'tiles':'export cells';
  const sizeLabel=m.effective_chunk_size_m!=null?` · ${m.effective_chunk_size_m} m ${lowRamResult?'tile edge':'cells'}${m.chunk_size_source&&m.chunk_size_source!=='user'?` (${m.chunk_size_source.replace(/_/g,' ')})`:''}`:'';
  const countLabel=m.chunk_count!=null?` · ${m.chunk_count} ${sectionLabel}`:'';
  const size=(m.output_bytes!=null)?` · ${(m.output_bytes/1048576).toFixed(1)} MiB on disk`:'';
  $('nksr-mesh-results').append(el('p',`NKSR mesh · ${modeLabel} · ${outputLabel} · ${(m.face_count||0).toLocaleString()} triangles${countLabel}${sizeLabel}${size} · Bounds: ${m.validation_status}`));
  if(m.vertex_count!=null)$('nksr-mesh-results').append(el('p',`${m.vertex_count.toLocaleString()} saved vertices; shared section vertices are duplicated`));
  if(m.validation_note)$('nksr-mesh-results').append(el('p',m.validation_note));
  if(m.mesh_output_mode!=='chunks'){
   const button=el('button','Download merged mesh');button.onclick=()=>download(`reconstruction/${job.id}/output/mesh.ply`);$('nksr-mesh-results').append(button);
  }
  if(m.mesh_output_mode!=='merged'){
   const manifestFile=lowRamResult?'tiles.json':'mesh_chunks/chunks.json';
   const manifest=el('button','Download tiles manifest');
   manifest.onclick=()=>download(`reconstruction/${job.id}/output/${manifestFile}`);$('nksr-mesh-results').append(manifest);
   // The run panel re-renders on every refresh, so fetched file names are cached per manifest.
   const cacheKey=`${job.id}/${manifestFile}`,files=meshFileCache.get(cacheKey);
   if(!files){
    const listing=el('button','List individual meshes');
    listing.onclick=async()=>{
     listing.disabled=true;
     try{
      const state=await json(`sessions/${selected}/artifact?path=${encodeURIComponent(`reconstruction/${job.id}/output/${manifestFile}`)}`);
      // Low RAM tiles all share the file name mesh.ply, so label them by tile id.
      meshFileCache.set(cacheKey,lowRamResult
       ?state.tiles.filter(t=>t.state==='COMPLETED').map(t=>[`tiles/${t.id}/mesh.ply`,t.id])
       :state.chunks.filter(c=>c.file).map(c=>[`mesh_chunks/${c.file}`,c.file]));
      renderReconstruction();
     }catch(e){listing.disabled=false;error(e)}
    };
    $('nksr-mesh-results').append(listing);
   }else if(files.length){
    const row=document.createElement('div');
    for(const [file,label] of files.slice(0,MESH_LINK_LIMIT)){
     const link=el('button',label);
     link.onclick=()=>download(`reconstruction/${job.id}/output/${file}`);row.append(link);
    }
    if(files.length>MESH_LINK_LIMIT)row.append(el('span',`showing ${MESH_LINK_LIMIT} of ${files.length}; download the manifest for the rest`));
    $('nksr-mesh-results').append(row);
   }
  }
 }
 if(job){const button=el('button','Preparation / reconstruction log');button.onclick=()=>download(`reconstruction/${job.id}/job.log`);$('reconstruction-results').append(button)}
}
async function reconstructionPanel(){
 const sid=selected;
 $('prepare-reconstruction').disabled=!sid||reconstructionPending||!!latestStatus?.capture?.busy||['running','stopping','orphaned'].includes(latestStatus?.processes?.reconstruction?.state)||['running','stopping','orphaned'].includes(latestStatus?.processes?.nksr?.state);
 if(!sid){reconstructionData=null;$('reconstruction-trajectory').replaceChildren();$('nksr-input-run').replaceChildren();renderReconstruction();return}
 const data=await json(`sessions/${sid}/reconstruction`);if(sid!==selected)return;
 reconstructionData=data;
 $('reconstruction-input-status').textContent=`${data.raw_bag?'✓':'○'} Raw bag · ${data.trajectories.length?'✓':'○'} GLIM trajectory · ${data.edited_sources.filter(x=>x.export_ready).length?'✓':'○'} saved cleanup export`;
 const select=$('reconstruction-trajectory'),old=select.value;select.replaceChildren();
 for(const path of data.trajectories){const option=el('option',path);option.value=path;select.append(option)}
 if(data.trajectories.includes(old))select.value=old;
 const editSelect=$('edited-source'),previousEdit=editSelect.value;editSelect.replaceChildren();
 for(const edit of data.edited_sources){const option=el('option',`${edit.id} · ${edit.export_ready?'export ready':'export required'}`);option.value=edit.id;option.disabled=!edit.export_ready;editSelect.append(option)}
 if(data.edited_sources.some(edit=>edit.id===previousEdit&&edit.export_ready))editSelect.value=previousEdit;
 $('prepare-reconstruction').disabled ||= !data.trajectories.length||!data.raw_bag;
 const runs=$('nksr-input-run'),previous=reconstructionSession===sid?runs.value:'';runs.replaceChildren();
 // The engine is part of the label because a prepared run is bound to one algorithm.
 for(const job of data.jobs){const option=el('option',`${job.id} · ${job.algorithm==='vdbfusion'?'VDBFusion':'NKSR'} · ${job.state}`);option.value=job.id;runs.append(option)}
 if(data.jobs.some(j=>j.id===previous))runs.value=previous;else if(runs.options.length)runs.selectedIndex=runs.options.length-1;
 reconstructionSession=sid;renderReconstruction();
}
$('reconstruction-form').onsubmit=async event=>{
 event.preventDefault();if(reconstructionPending||!selected)return;
 const engine=algorithm(),cm=Number($('voxel-size-cm').value);
 if(engine==='nksr'&&(!Number.isFinite(cm)||cm<0.2||cm>20)){error(Error('Preparation voxel must be between 0.2 and 20 cm'));return}
 const filtering=$('filter-edited-geometry').checked,tolerance=Number($('edited-tolerance').value);
 if(filtering&&!$('edited-source').value){error(Error('Select an exported saved cleanup'));return}
 // Only NKSR needs a proximity tolerance; VDBFusion measures its own association radius.
 if(engine==='nksr'&&filtering&&(!Number.isFinite(tolerance)||tolerance<=0)){error(Error('Select an exported saved cleanup and a positive tolerance'));return}
 if(engine==='vdbfusion'&&!vdbfusionAvailable()){error(Error(`VDBFusion is not available: ${reconstructionData?.vdbfusion?.status||'check the installation'}. Install it with scripts/setup_vdbfusion.sh. NKSR is never substituted automatically.`));return}
 let vdbfusion=null;
 try{vdbfusion=engine==='vdbfusion'?vdbfusionSettings():null}catch(e){error(e);return}
 reconstructionPending=true;$('prepare-reconstruction').disabled=true;
 const body={trajectory:$('reconstruction-trajectory').value,algorithm:engine,voxel_size_m:cm/100.0,
  filter_edited_geometry:filtering,...(filtering?{edit_id:$('edited-source').value,filter_tolerance_m:tolerance}:{}),
  ...(vdbfusion?{vdbfusion}:{})};
 try{await json(`sessions/${selected}/reconstruction`,body);reconstructionSession=null;await refresh()}catch(e){error(e)}finally{reconstructionPending=false;await reconstructionPanel()}
};
for(const id of ['voxel-size-cm','reconstruction-trajectory','filter-edited-geometry','edited-source','edited-tolerance','nksr-input-run','nksr-mode','nksr-device','nksr-output-mode','nksr-detail','nksr-chunk','nksr-knn','nksr-angle','nksr-mise','vdbfusion-voxel','vdbfusion-trunc','vdbfusion-space-carving','vdbfusion-output-mode','vdbfusion-roi-min','vdbfusion-roi-max','vdbfusion-origin-budget','vdbfusion-association-multiplier','vdbfusion-boundary-margin','vdbfusion-unsupported','vdbfusion-mask-triangles','vdbfusion-memory-budget'])$(id).addEventListener('input',()=>{if(id.startsWith('vdbfusion-voxel')||id==='vdbfusion-trunc')$('vdbfusion-preset').value='';renderReconstruction()});
for(const id of ['algorithm-vdbfusion','algorithm-nksr'])$(id).addEventListener('change',renderReconstruction);
$('vdbfusion-preset').onchange=()=>{const preset=VDBFUSION_PRESETS[$('vdbfusion-preset').value];
 if(preset){$('vdbfusion-voxel').value=preset[0];$('vdbfusion-trunc').value=preset[1]}renderReconstruction()};
$('nksr-check').onclick=async()=>{try{await json('nksr/check',{device:$('nksr-device').value});await refresh()}catch(e){error(e)}};
$('vdbfusion-check').onclick=async()=>{try{await json('vdbfusion/check',{});await refresh()}catch(e){error(e)}};
$('reconstruct-mesh').onclick=async()=>{
 if(reconstructionPending)return;
 const job=reconstructionData?.jobs.find(j=>j.id===$('nksr-input-run').value);
 const engine=job?.algorithm||job?.metadata?.algorithm||'nksr';
 let body;
 try{body=engine==='vdbfusion'?{mesh_output_mode:$('vdbfusion-output-mode').value,vdbfusion:vdbfusionSettings()}:nksrSettings()}
 catch(e){error(e);return}
 reconstructionPending=true;renderReconstruction();
 try{await json(`sessions/${selected}/reconstruction/${$('nksr-input-run').value}/mesh`,body);await refresh()}catch(e){error(e)}finally{reconstructionPending=false;await reconstructionPanel()}
};
$('cancel-mesh').onclick=async()=>{try{await json(`sessions/${selected}/reconstruction/${$('nksr-input-run').value}/cancel`,{});await refresh()}catch(e){error(e)}};

let colorizationPending=false,colorizationData=null;
function renderColorization(){
 const data=colorizationData,last=data?.jobs?.at(-1);
 const busy=colorizationPending||['running','stopping','orphaned'].includes(latestStatus?.processes?.colorization?.state);
 const enabled=!!selected&&!!data?.raw_bag&&!!data?.camera_enabled;
 $('colorization-input-status').textContent=data?`${data.raw_bag?'✓':'○'} Raw bag · ${data.camera_enabled?'✓':'○'} RGB recorded`:'Select a scan recorded with RGB camera images.';
 $('colorize-start').disabled=!enabled||busy||!!latestStatus?.capture?.busy;
 $('colorize-cancel').hidden=last?.state!=='RUNNING';
 $('colorization-results').replaceChildren();
 if(!last){$('colorization-status').textContent='NOT_RUN';$('colorization-status').dataset.state='';return}
 $('colorization-status').textContent=last.state==='RUNNING'?`RUNNING · ${last.progress||''}`:
  (last.state==='COMPLETED'?`COMPLETED · ${(last.percentage_colored??0).toFixed(1)}% colored`:last.state);
 $('colorization-status').dataset.state=String(last.state||'').toLowerCase();
 if(last.state==='COMPLETED'){
  $('colorization-results').append(el('p',`${(last.colored_point_count??0).toLocaleString()} / ${(last.final_point_count??0).toLocaleString()} points colored · ${(last.percentage_colored??0).toFixed(1)}%`));
  for(const path of last.outputs||[]){
   const row=el('div',path.split('/').pop()+' ');
   if(path.endsWith('.ply')){const b=el('button','Preview');b.onclick=()=>showCloud(path);row.append(b)}
   const d=el('button','Download');d.onclick=()=>download(path);row.append(d);
   $('colorization-results').append(row);
  }
  const log=el('button','Colorization log');log.onclick=()=>download(`colorization/${last.id}/job.log`);$('colorization-results').append(log);
 }
}
async function colorizationPanel(){
 const sid=selected;
 if(!sid){colorizationData=null;renderColorization();return}
 const data=await json(`sessions/${sid}/colorization`);if(sid!==selected)return;
 colorizationData=data;renderColorization();
}
$('colorize-start').onclick=async()=>{
 if(colorizationPending||!selected)return;colorizationPending=true;renderColorization();
 try{await json(`sessions/${selected}/colorization`,{allow_unvalidated_calibration:$('colorization-allow-unvalidated').checked});await refresh()}catch(e){error(e)}finally{colorizationPending=false;await colorizationPanel()}
};
$('colorize-cancel').onclick=async()=>{try{const last=colorizationData?.jobs?.at(-1);if(last)await json(`sessions/${selected}/colorization/${last.id}/cancel`,{});await refresh()}catch(e){error(e)}};
