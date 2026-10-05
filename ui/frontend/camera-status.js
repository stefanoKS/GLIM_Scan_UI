// Sensor state is independent of the previous capture result.
export function cameraLabel(status) {
 const selection=status.camera_selection||{}, h=status.health.camera||{};
 const name=selection.active_profile==='d405'?'D405':selection.active_profile==='dfk33ux287'?'DFK 33UX287':null;
 if(name && h.healthy && h.camera_info_seen && selection.candidates?.[selection.active_profile]?.stream_healthy!==false) return {ready:true,text:`${name} · Ready · ${h.width}×${h.height} @ ${(h.image_hz||0).toFixed(1)} FPS${selection.fallback_used?' · fallback':''}`};
 if(name && h.camera_running) return {ready:false,text:`${name} · starting stream`};
 const detected=Object.entries(selection.candidates||{}).filter(([,v])=>v.detected);
 if(detected.length && !selection.fallback_reason) return {ready:false,text:`${detected[0][0]==='d405'?'D405':'DFK 33UX287'} · USB detected · stream not ready`};
 return {ready:false,text:status.config.system.camera.enabled?'RGB unavailable · LiDAR-only scan available':'RGB disabled'};
}

export function captureMessages(c,blocked,hint) {
 return {
  current:blocked||(c.state==='COMPLETE'?'Press Start Scan to begin a new recording.':c.error||c.warnings?.at(-1)||hint),
  previous:c.state==='COMPLETE'?'Previous scan: '+(c.error||c.warnings?.join(' ')||c.outcome||'saved'):''
 };
}
