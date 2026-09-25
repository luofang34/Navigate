// A media seek time can differ from the presentation timestamp of its decoded frame.
export class DecodedVideoTime {
 constructor(video,{visibility=globalThis.document}={}){this.video=video;this.visibility=visibility;this.visibilityWaiters=new Set();this.value=null;this.waiters=new Set();this.closed=false;this.schedule()}
 schedule(){if(this.closed||!this.video.requestVideoFrameCallback)return;this.handle=this.video.requestVideoFrameCallback((_,metadata)=>{this.value=metadata.mediaTime;for(const resolve of this.waiters)resolve(this.value);this.waiters.clear();this.schedule()})}
 beforeSeek(){this.video.cancelVideoFrameCallback?.(this.handle);this.value=null;this.schedule()}
 async read(){
  if(!this.video.requestVideoFrameCallback||this.closed)return null;
  if(this.visibility?.hidden)await this.waitUntilVisible();
  if(this.closed)return null;
  if(this.value!==null)return this.value;
  let finish,timer;
  let value;
  try{value=await new Promise(resolve=>{finish=resolve;this.waiters.add(finish);timer=setTimeout(()=>resolve(null),5000)})}
  finally{clearTimeout(timer);this.waiters.delete(finish)}
  if(value===null&&!this.closed&&this.visibility?.hidden)return this.read();
  return value;
 }
 waitUntilVisible(){
  return new Promise(resolve=>{
   const finish=()=>{this.visibility.removeEventListener('visibilitychange',changed);this.visibilityWaiters.delete(finish);resolve()};
   const changed=()=>{if(!this.visibility.hidden||this.closed)finish()};
   this.visibilityWaiters.add(finish);this.visibility.addEventListener('visibilitychange',changed);changed();
  });
 }
 close(){this.closed=true;for(const finish of this.visibilityWaiters)finish();this.video.cancelVideoFrameCallback?.(this.handle);for(const resolve of this.waiters)resolve(null);this.waiters.clear()}
}

// Seeking to a rounded decoded timestamp can select the preceding frame.
export function replaySeekTime(frame){
 const time=frame.requested_time_s??frame.capture_time_ns/1e9;
 if(!Number.isFinite(time)||time<0)throw Error('Saved video seek time is invalid');
 return time;
}
