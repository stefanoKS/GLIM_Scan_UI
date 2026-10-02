// Parallel RGB acquisition and a separate calibration dataset UI.
const $=id=>document.getElementById(id);
let ctx,selected='',cameraEnabled=false,cameraRunning=false,previewBusy=false,previewURL=null;
export function setupCamera(context){
 ctx=context;
 $("camera-enabled").onchange=async()=>{try{await ctx.json("camera/enabled",{enabled:$("camera-enabled").checked});await ctx.refresh()}catch(e){ctx.error(e);$("camera-enabled").checked=cameraEnabled}};
 $('camera-preview').onerror=()=>{$('camera-preview').hidden=true;};
 $('import-intrinsics').onclick=async()=>{
  try{const file=$('intrinsics-file').files[0];if(!file)throw Error('Select a measured ROS calibration YAML file');if(file.size>65536)throw Error('Intrinsic YAML must be <=64 KiB');await ctx.json('camera/intrinsics',{yaml_text:await file.text()});await ctx.refresh();}catch(e){ctx.error(e)}
 };
 $('calibration-create').onclick=async()=>{try{const m=await ctx.json('calibrations',{name:$('calibration-name').value});selected=m.id;await ctx.refresh()}catch(e){ctx.error(e)}};
 $('calibration-select').onchange=()=>{selected=$('calibration-select').value;ctx.refresh()};
 document.querySelectorAll('[data-cal-action]').forEach(button=>button.onclick=async()=>{
  button.disabled=true;$('error').hidden=true;
  try{if(!selected)throw Error('Select a calibration dataset');await ctx.json(`calibrations/${selected}/action`,{action:button.dataset.calAction,notes:$('calibration-evidence').value});await ctx.refresh()}catch(e){ctx.error(e)}finally{button.disabled=false}
 });
 setInterval(async()=>{
  if(!cameraEnabled||!cameraRunning||previewBusy){if(!cameraEnabled||!cameraRunning)$('camera-preview').hidden=true;return}
  previewBusy=true;
  try{const r=await fetch('/api/camera/preview',{cache:'no-store'});if(!r.ok)throw Error('Preview unavailable; raw recording is independent');const blob=await r.blob();const old=previewURL;previewURL=URL.createObjectURL(blob);$('camera-preview').src=previewURL;$('camera-preview').hidden=false;if(old)URL.revokeObjectURL(old);$('camera-preview-note').textContent='Low-rate JPEG preview · raw images remain in the bag'}catch(e){$('camera-preview').hidden=true;$('camera-preview-note').textContent=e.message}finally{previewBusy=false}
 },1000);
}
export async function refreshCamera(s){
 cameraEnabled=!!s.config.system.camera.enabled;$('camera-enabled').checked=cameraEnabled;const c=s.health.camera||{};cameraRunning=!!c.camera_running;
 $('camera-status').textContent=`${s.detection?.camera?.detected?'✓ DFK33UX287 detected (USB)':'○ DFK33UX287 not detected'}\n${cameraEnabled?'ENABLED':'DISABLED'} · ${c.state||'unknown'}\nProcess ${cameraRunning?'running':'stopped'}\nImage FPS ${(c.hz||0).toFixed(2)} · ${c.width||'—'} × ${c.height||'—'}\nLast frame age ${c.image_age==null?'—':c.image_age.toFixed(3)+' s'}\nSource timestamp ${c.last_image_timestamp??'—'}\nCameraInfo ${c.camera_info_valid?'VALID':c.camera_info_seen?'INVALID':'MISSING'}\nFrame ${c.frame_id||'—'}\nTimestamp jitter ${c.timestamp_jitter_sec??'—'} s`;
 const intr=s.camera_calibration?.intrinsics||{},ext=s.camera_calibration?.extrinsics||{};
 $('camera-intrinsics').textContent=`Camera Intrinsics: ${intr.status||'MISSING'}\nModel: ${intr.model||'—'}\nResolution: ${intr.width||'—'} × ${intr.height||'—'}\nfx/fy/cx/cy: ${intr.intrinsics?.slice(0,4).join(', ')||'—'}\n${intr.error||''}`;
 $('camera-extrinsics').textContent=`LiDAR–camera extrinsics: ${ext.calibrated?'AVAILABLE':'NOT AVAILABLE'} · ${ext.validated?'VALIDATED':'NOT VALIDATED'}\nT_lidar_camera: ${ext.T_lidar_camera?.join(', ')||'—'}\nTime offset: ${s.config.camera?.time_offset_sec??0} s (configured, not estimated)`;
 const rows=await ctx.json('calibrations');const select=$('calibration-select');select.replaceChildren(new Option('Select dataset',''));
 for(const m of rows)select.append(new Option(`${m.name} · ${m.state}${m.mock?' · MOCK':''}`,m.id));select.value=selected;
 if(!selected){$('calibration-detail').textContent='No dataset selected';$('calibration-log').textContent='';return}
 const m=await ctx.json(`calibrations/${selected}`);
 $('calibration-detail').textContent=JSON.stringify({id:m.id,state:m.state,mock:m.mock,error:m.error,validated:m.validated,captures:m.captures,jobs:m.jobs},null,2);
 const latest=m.jobs.at(-1);
 if(latest){const r=await fetch(`/api/calibrations/${selected}/logs/${latest.id}`);$('calibration-log').textContent=r.ok?await r.text():'Tool log is not yet available'}
}
