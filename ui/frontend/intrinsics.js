import {apiErrorMessage} from './api-errors.js';
const $=id=>document.getElementById(id);
let views=0,required=4,cameraReady=false,cameraRunning=false,saved=false,busy=false,previewURL=null,loading=false;
async function request(path,method='GET'){
 const response=await fetch('/api/'+path,{method,cache:'no-store'});
 if(!response.ok){const payload=await response.json().catch(()=>null);throw Error(apiErrorMessage(payload,`Request failed (${response.status})`))}
 return response.json();
}
function message(text){$('intrinsics-error').textContent=text;$('intrinsics-error').hidden=!text}
function controls(){
 $('capture').disabled=busy||!cameraReady||views>=40;
 $('calibrate').disabled=busy||views<required;
 $('reset-views').disabled=busy||views===0;
 $('delete-intrinsics').disabled=busy||!saved;
 $('start-camera').disabled=busy||cameraRunning;
 $('view-count').textContent=`${views} / ${required} views`;
}
async function refresh(){
 try{
  const [state,progress]=await Promise.all([request('status'),request('camera/intrinsics/views')]);
    const health=state.health.camera||{};cameraReady=!!health.healthy;cameraRunning=!!health.camera_running;
  $('camera-state').textContent=cameraReady?`Camera ready · ${health.hz.toFixed(1)} fps`:health.camera_running?'Waiting for camera frames':'Camera stopped';
  $('resolution').textContent=`${state.config.camera.width} × ${state.config.camera.height}`;
  views=progress.views;required=progress.required;
  const intrinsics=progress.intrinsics||{};saved=intrinsics.status==='VALID'||intrinsics.status==='INVALID';
  $('intrinsics-result').textContent=intrinsics.status==='VALID'?`VALID · ${intrinsics.width} × ${intrinsics.height}\nfx ${intrinsics.intrinsics[0].toFixed(2)} · fy ${intrinsics.intrinsics[1].toFixed(2)}\ncx ${intrinsics.intrinsics[2].toFixed(2)} · cy ${intrinsics.intrinsics[3].toFixed(2)}`:intrinsics.status==='INVALID'?`INVALID · ${intrinsics.error}`:'No measured intrinsics saved';
  controls();
 }catch(error){message(error.message)}
}
async function preview(){
 if(!cameraReady||loading){if(!cameraReady){$('intrinsics-preview').hidden=true;$('preview-state').hidden=false}return}
 loading=true;
 try{
  const response=await fetch('/api/camera/preview',{cache:'no-store'});
  if(!response.ok)throw Error('Waiting for camera preview');
  const next=URL.createObjectURL(await response.blob()),image=$('intrinsics-preview');
  image.onload=()=>{if(previewURL)URL.revokeObjectURL(previewURL);previewURL=next;image.hidden=false;$('preview-state').hidden=true};
  image.src=next;
 }catch(error){$('preview-state').textContent=error.message;$('preview-state').hidden=false}finally{loading=false}
}
async function action(button,operation){
 if(busy)return;
 busy=true;controls();message('');
 try{await operation();await refresh()}catch(error){message(error.message)}finally{busy=false;controls()}
}
$('start-camera').onclick=()=>action('start-camera',()=>request('calibration/prepare','POST'));
$('capture').onclick=()=>action('capture',async()=>{const result=await request('camera/intrinsics/views','POST');$('capture-feedback').textContent=`View ${result.views} captured · ${result.corners} corners detected`});
$('reset-views').onclick=()=>action('reset',async()=>{await request('camera/intrinsics/views','DELETE');$('capture-feedback').textContent='Views cleared'});
$('calibrate').onclick=()=>action('calibrate',async()=>{const result=await request('camera/intrinsics/calibrate','POST');$('calibration-quality').textContent=`Saved · ${result.quality.views} views · ${result.quality.rms.toFixed(2)} px reprojection error`});
$('delete-intrinsics').onclick=()=>{if(confirm('Delete the active camera intrinsics? The previous file will be archived.'))action('delete',async()=>{await request('camera/intrinsics','DELETE');$('calibration-quality').textContent='';$('capture-feedback').textContent=''})};
refresh();preview();setInterval(refresh,2500);setInterval(preview,700);