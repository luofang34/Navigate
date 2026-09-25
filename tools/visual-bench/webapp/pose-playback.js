export function supportedPose(h){return Boolean(h&&(h.accepted||h.tracking_supported||h.scene_supported)&&h.position_enu_m?.length===3&&h.eye_to_enu_xyzw?.length===4)}

export function interpolatePose(a,b,t){
 const first=a.eye_to_enu_xyzw,raw=b.eye_to_enu_xyzw;
 const sign=first.reduce((s,v,i)=>s+v*raw[i],0)<0?-1:1,last=raw.map(v=>v*sign);
 const dot=Math.min(1,Math.max(-1,first.reduce((s,v,i)=>s+v*last[i],0)));
 const angle=Math.acos(dot),denominator=Math.sin(angle);
 const u=dot>.9995?1-t:Math.sin((1-t)*angle)/denominator,v=dot>.9995?t:Math.sin(t*angle)/denominator;
 const q=first.map((x,i)=>u*x+v*last[i]),norm=Math.hypot(...q);
 return {position_enu_m:a.position_enu_m.map((x,i)=>x+(b.position_enu_m[i]-x)*t),eye_to_enu_xyzw:q.map(x=>x/norm)};
}

export function connectedSamples(a,b,{maxGap=.5}={}){
 if(!a||!b||!supportedPose(a.h)||!supportedPose(b.h)||!(b.time>a.time)||b.time-a.time>maxGap||b.h.continuity_break)return false;
 const am=a.h.map_manifest_sha256??a.h.tracking_anchor?.map_manifest_sha256,bm=b.h.map_manifest_sha256??b.h.tracking_anchor?.map_manifest_sha256;
 if(!am||am!==bm||(a.h.track_id??a.h.candidate_id)!==(b.h.track_id??b.h.candidate_id)||a.h.anchor_observation_sha256!==b.h.anchor_observation_sha256)return false;
 return JSON.stringify(a.h.tracking_anchor)===JSON.stringify(b.h.tracking_anchor);
}

// Interpolation is a display operation. It cannot create an accepted observation.
export function playbackPose(samples,time,{maxGap=.5}={}){
 if(!samples.length||time<samples[0].time||time>samples.at(-1).time)return null;
 let low=0,high=samples.length-1;
 while(low<high){const mid=Math.ceil((low+high)/2);if(samples[mid].time<=time)low=mid;else high=mid-1}
 const a=samples[low],b=samples[low+1];
 if(!supportedPose(a.h))return null;
 if(Math.abs(time-a.time)<1e-6)return {pose:interpolatePose(a.h,a.h,0),sample:a,interpolated:false};
 if(!connectedSamples(a,b,{maxGap}))return null;
 return {pose:interpolatePose(a.h,b.h,(time-a.time)/(b.time-a.time)),sample:a,interpolated:true};
}

export class VideoPoseClock {
 constructor(video,update){
  this.video=video;this.update=update;this.handle=null;this.closed=false;this.mediaTime=null;
  this.event=()=>update(this.time);this.invalidate=()=>{this.mediaTime=null};
  for(const name of ['seeking','loadstart'])video.addEventListener(name,this.invalidate);
  for(const name of ['seeked','loadeddata','pause'])video.addEventListener(name,this.event);
  this.schedule();
 }
 get time(){return this.mediaTime??this.video.currentTime}
 schedule(){if(this.closed)return;if(this.video.requestVideoFrameCallback){this.handle=this.video.requestVideoFrameCallback((_,metadata)=>{this.mediaTime=metadata.mediaTime;this.update(this.time);this.schedule()})}else{this.handle=requestAnimationFrame(()=>{this.update(this.time);this.schedule()})}}
 close(){this.closed=true;if(this.video.cancelVideoFrameCallback)this.video.cancelVideoFrameCallback(this.handle);else cancelAnimationFrame(this.handle);for(const name of ['seeked','loadeddata','pause'])this.video.removeEventListener(name,this.event);for(const name of ['seeking','loadstart'])this.video.removeEventListener(name,this.invalidate)}
}
