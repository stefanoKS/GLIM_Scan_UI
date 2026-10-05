// Parallel RGB acquisition and a separate calibration dataset UI.
const $=id=>document.getElementById(id);
let ctx,selected='',cameraEnabled=false,cameraRunning=false,previewBusy=false,previewURL=null;
export function setupCamera(context){
 ctx=context;
 const wizard=async action=>{try{if(!selected)throw Error('Create or select an alignment first');await ctx.json(`calibrations/${selected}/wizard`,{action,notes:$('wizard-evidence').value});await ctx.refresh()}catch(e){ctx.error(e)}};
 $('wizard-capture').onclick=()=>wizard('capture');$('wizard-stop').onclick=()=>wizard('stop');$('wizard-next').onclick=()=>wizard('advance');
 $('wizard-cancel').onclick=async()=>{try{await ctx.json(`calibrations/${selected}/action`,{action:'cancel'});await ctx.refresh()}catch(e){ctx.error(e)}};

 $("camera-enabled").onchange=async()=>{try{await ctx.json("camera/enabled",{enabled:$("camera-enabled").checked});await ctx.refresh()}catch(e){ctx.error(e);$("camera-enabled").checked=cameraEnabled}};
 $('camera-profile').onchange=async()=>{try{await ctx.json('camera/profile',{profile:$('camera-profile').value},'PUT');await ctx.refresh()}catch(e){ctx.error(e);await ctx.refresh()}};
 $('camera-preview').onerror=()=>{$('camera-preview').hidden=true;};
 $('import-intrinsics').onclick=async()=>{
  try{const file=$('intrinsics-file').files[0];if(!file)throw Error('Select a measured ROS calibration YAML file');if(file.size>65536)throw Error('Intrinsic YAML must be <=64 KiB');await ctx.json('camera/intrinsics',{yaml_text:await file.text()});await ctx.refresh();}catch(e){ctx.error(e)}
 };
 $('calibration-create').onclick=async()=>{try{const m=await ctx.json('calibrations',{name:$('calibration-name').value||'Alignment '+new Date().toLocaleString()});selected=m.id;await ctx.refresh()}catch(e){ctx.error(e)}};
 $('calibration-select').onchange=()=>{selected=$('calibration-select').value;ctx.refresh()};
 document.querySelectorAll('[data-cal-action]').forEach(button=>button.onclick=async()=>{
  button.disabled=true;$('error').hidden=true;
  try{if(!selected)throw Error('Select a calibration dataset');await ctx.json(`calibrations/${selected}/action`,{action:button.dataset.calAction,notes:$('calibration-evidence').value});await ctx.refresh()}catch(e){ctx.error(e)}finally{button.disabled=false}
 });
 setInterval(async()=>{
  if(!cameraRunning||previewBusy){if(!cameraRunning)$('camera-preview').hidden=true;return}
  previewBusy=true;
  try{const r=await fetch('/api/camera/preview',{cache:'no-store'});if(!r.ok)throw Error('Preview unavailable; raw recording is independent');const blob=await r.blob();const old=previewURL;previewURL=URL.createObjectURL(blob);$('camera-preview').src=previewURL;$('camera-preview').hidden=false;if(old)URL.revokeObjectURL(old);$('camera-preview-note').textContent='Low-rate JPEG preview · raw images remain in the bag'}catch(e){$('camera-preview').hidden=true;$('camera-preview-note').textContent=e.message}finally{previewBusy=false}
 },1000);
}
export async function refreshCamera(s){
 cameraEnabled=!!s.config.system.camera.enabled;$('camera-enabled').checked=cameraEnabled;$('camera-profile').value=s.config.system.camera.profile||'d405';$('camera-profile').disabled=!!s.capture?.busy||!s.capture?.can_start;const c=s.health.camera||{};cameraRunning=!!c.camera_running;
 $('camera-status').textContent=`${s.detection?.camera?.detected?'✓':'○'} ${s.config.camera?.model||'Camera'} ${s.detection?.camera?.detected?'detected (USB)':'not detected'}\n${cameraEnabled?'ENABLED':'DISABLED'} · ${c.state||'unknown'}\nProcess ${cameraRunning?'running':'stopped'}\nImage FPS ${(c.hz||0).toFixed(2)} · ${c.width||'—'} × ${c.height||'—'}\nLast frame age ${c.image_age==null?'—':c.image_age.toFixed(3)+' s'}\nSource timestamp ${c.last_image_timestamp??'—'}\nCameraInfo ${c.camera_info_valid?'VALID':c.camera_info_seen?'INVALID':'MISSING'}\nFrame ${c.frame_id||'—'}\nTimestamp jitter ${c.timestamp_jitter_sec??'—'} s`;
 const factory=s.config.camera?.source==='realsense';
 document.querySelectorAll('a[href="/intrinsics.html"]').forEach(link=>link.hidden=factory);
 $('wizard-import').hidden=factory;
 $('intrinsics-help').textContent=factory?'D405 factory lens calibration is loaded automatically. No printed board is needed. Camera–LiDAR mounting alignment is still required.':'Use the camera wizard to capture a printed board, or import a measured calibration below.';
 const intr=s.camera_calibration?.intrinsics||{},ext=s.camera_calibration?.extrinsics||{};
 $('calibration-summary').textContent=`Camera lens: ${intr.status==='VALID'?'calibrated':'calibration needed'} · Camera–LiDAR alignment: ${ext.validated?'validated':ext.calibrated?'needs independent review':'not calibrated'}`;
 $('calibration-create').disabled=intr.status!=='VALID'||!cameraEnabled||!!s.capture?.busy;
 if(!cameraEnabled)$('calibration-summary').textContent+=' · Enable RGB in Settings for LiDAR alignment.';
 $('camera-intrinsics').textContent=`Camera Intrinsics: ${intr.status||'MISSING'}\nModel: ${intr.model||'—'}\nResolution: ${intr.width||'—'} × ${intr.height||'—'}\nfx/fy/cx/cy: ${intr.intrinsics?.slice(0,4).join(', ')||'—'}\n${intr.error||''}`;
 $('camera-extrinsics').textContent=`LiDAR–camera extrinsics: ${ext.calibrated?'AVAILABLE':'NOT AVAILABLE'} · ${ext.validated?'VALIDATED':'NOT VALIDATED'}\nT_lidar_camera: ${ext.T_lidar_camera?.join(', ')||'—'}\nTime offset: ${s.config.camera?.time_offset_sec??0} s (configured, not estimated)`;
 const rows=await ctx.json('calibrations');const select=$('calibration-select');select.replaceChildren(new Option('Select dataset',''));
 for(const m of rows)select.append(new Option(`${m.name}${m.validated?' · Validated':''}${m.mock?' · MOCK':''}`,m.id));select.value=selected;
 if(!selected){$('calibration-detail').textContent='No dataset selected';$('calibration-log').textContent='';wizardState(null);return}
 const m=await ctx.json(`calibrations/${selected}`);wizardState(m);
 $('calibration-detail').textContent=JSON.stringify({id:m.id,state:m.state,mock:m.mock,error:m.error,validated:m.validated,captures:m.captures,jobs:m.jobs},null,2);
 const latest=m.jobs.at(-1);
 if(latest){const r=await fetch(`/api/calibrations/${selected}/logs/${latest.id}`);$('calibration-log').textContent=r.ok?await r.text():'Tool log is not yet available'}
}

function wizardState(m){
 const state=m?.state||'NONE',busy=m?.jobs?.some(j=>j.state==='running');
 const steps={NONE:['Select an alignment','Check camera intrinsics above, then choose New alignment.'],CREATED:['Record reference views','Keep both sensors still, with visible structure in their overlapping view. Record several short views.'],CAPTURING:['Recording reference view','Keep the rig still. Stop when the reference view is captured.'],CAPTURED:['Prepare reference views','Add another reference view, or continue to prepare the captured data.'],PREPROCESSED:['Align the sensors','Continue to open manual alignment on the server desktop. Save the alignment and close the window.'],INITIALIZED:['Refine the alignment','Continue to refine the saved alignment. This requires the server desktop.'],CALIBRATED:['Save alignment','Continue to preserve and apply this result. Existing calibration history is retained.'],IMPORTED:['Validate independently','Review overlays and measured residuals using independent views. Record evidence below before marking the alignment validated.'],VALIDATED:['Alignment validated','The saved alignment and its validation evidence remain available.']};
 const [title,hint]=steps[state]||steps.NONE;$('wizard-step').textContent=busy?'Working…':title;$('wizard-instruction').textContent=hint;$('wizard-error').textContent=m?.error||'';
 $('wizard-capture').hidden=!['CREATED','CAPTURED'].includes(state);$('wizard-stop').hidden=state!=='CAPTURING';
 $('wizard-next').hidden=['NONE','CREATED','CAPTURING','VALIDATED'].includes(state);$('wizard-next').disabled=!!busy;
 $('wizard-next').textContent=state==='IMPORTED'?'Record validation':state==='CALIBRATED'?'Save alignment':'Continue';
 $('wizard-cancel').hidden=!busy;$('wizard-evidence-label').hidden=state!=='IMPORTED';
}
