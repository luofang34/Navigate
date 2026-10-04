import {gray} from './observation.js';

// Real-time processing follows the playback clock and never waits for frames to be decoded on demand.
// `cancel` releases the callback when another event ends the wait first.
export function presentedFrame(video){
 let handle=null;
 const frame=new Promise((resolve,reject)=>{
  if(typeof video.requestVideoFrameCallback!=='function'){reject(Error('This browser cannot report presented video frames'));return}
  handle=video.requestVideoFrameCallback((now,metadata)=>{handle=null;resolve({mediaTime:metadata.mediaTime,presentedFrames:metadata.presentedFrames,now})});
 });
 return {frame,cancel(){if(handle!==null)video.cancelVideoFrameCallback?.(handle);handle=null}};
}

// Latency statistics use the most recent `capacity` records, so a live stream keeps a fixed memory size.
export class LatencyWindow {
 constructor(capacity=512){if(!Number.isInteger(capacity)||capacity<1)throw Error('Invalid latency window');this.capacity=capacity;this.values=[];this.next=0;this.last=null}
 add(value){if(this.values.length<this.capacity)this.values.push(value);else this.values[this.next]=value;this.next=(this.next+1)%this.capacity;this.last=value}
 summary(){
  const sorted=[...this.values].sort((a,b)=>a-b),at=q=>sorted.length?sorted[Math.min(sorted.length-1,Math.floor(q*sorted.length))]:null;
  return {p50:at(.5),p90:at(.9),last:this.last,samples:sorted.length,scope:`wall time from frame presentation to its estimate; the most recent ${this.capacity} frames`};
 }
}

export function supportedPose(result){return Boolean(result?.candidate_hypotheses?.some(h=>h.accepted||h.tracking_supported))}

// Frames presented while an estimate runs are sampled into a bounded buffer. Once a pose is supported,
// the loop tracks forward through the buffer in hops of `hop` media seconds, faster than the source
// plays, until the pose is current again. A failed hop halves the hop and retries closer to the
// supported pose; a supported hop widens it. A hop above the minimum is marked cheap to retry; the
// nearest frame gets the full tracking and map checks.
export class CatchUp {
 constructor({rateHz=2,maxSeconds=120,minHop=.5,maxHop=4}={}){Object.assign(this,{rateHz,maxSeconds,minHop,maxHop});this.frames=[];this.tried=new Set();this.hop=1}
 wants(mediaTime){const last=this.frames.at(-1);return !last||mediaTime-last.time>=1/this.rateHz}
 add(frame){this.frames.push(frame);while(this.frames.length&&frame.time-this.frames[0].time>this.maxSeconds)this.frames.shift()}
 clear(){this.frames=[];this.tried.clear()}
 // Returns the buffered frame to track after the supported pose at `anchor`, or null when the newest
 // frame is within one hop or every buffered frame was tried. A failed frame is not tried again.
 next(anchor){
  this.frames=this.frames.filter(f=>f.time>anchor);for(const time of this.tried)if(time<=anchor)this.tried.delete(time);
  const newest=this.frames.at(-1);if(!newest||newest.time-anchor<=this.hop)return null;
  const open=this.frames.filter(f=>!this.tried.has(f.time)),frame=open.find(f=>f.time>=anchor+this.hop)??open.at(-1);
  if(!frame)return null;
  this.tried.add(frame.time);return {...frame,catch_up:this.hop>this.minHop?'hop':'nearest'};
 }
 settle(supported){this.hop=supported?Math.min(this.maxHop,this.hop*1.5):Math.max(this.minHop,this.hop/2)}
}

function compact(observation){
 // The worker rebuilds its canvas from gray pixels; only the saved frame needs an image file.
 const {canvas,valid:_valid,...pixels}=observation;
 return canvas?{...pixels,blob:new Promise(resolve=>canvas.toBlob(resolve,'image/png'))}:pixels;
}

// File and live sources play on their own clock. Frames presented during a long estimate, such as an
// area search, wait in the catch-up buffer instead of pausing the source. The caller receives each
// result through `estimate`; the run keeps only counters and a fixed latency window.
export async function runRealtime({video,camera,estimate,signal,capture=grab,catchUp=new CatchUp(),supported=supportedPose,onDropped=()=>{},clock=()=>performance.now(),latencyWindow=512}){
 const latencies=new LatencyWindow(latencyWindow);let processed=0,anchor=null,presented=0,lastMediaTime=-Infinity,dropped=0,busyMs=0,buffered=0,sampling=false,sampleHandle=null;const started=clock();
 const ended=()=>video.ended||Boolean(signal?.aborted);
 // One listener for each event serves the whole run. A stopped or frozen live stream presents no
 // further frame, so a stop request must not wait for one.
 let finish;const finished=new Promise(resolve=>{finish=()=>resolve(null)});
 video.addEventListener('ended',finish);signal?.addEventListener('abort',finish);
 const sample=()=>{if(!sampling||ended())return;sampleHandle=video.requestVideoFrameCallback((now,metadata)=>{
  sampleHandle=null;
  if(sampling&&catchUp.wants(metadata.mediaTime)){catchUp.add({...compact(capture(video,camera,metadata.mediaTime)),time:metadata.mediaTime,presented_at:now});buffered++}
  sample();
 })};
 const stopSampling=()=>{sampling=false;if(sampleHandle!==null)video.cancelVideoFrameCallback?.(sampleHandle);sampleHandle=null};
 try{
  await video.play();
  while(!ended()){
   let observation=anchor===null?null:catchUp.next(anchor);
   if(anchor===null)catchUp.clear();
   if(!observation){
    const next=presentedFrame(video);let frame;
    try{frame=await Promise.race([next.frame,finished])}finally{next.cancel()}
    if(!frame||ended())break;
    if(frame.mediaTime<=lastMediaTime)continue;
    if(presented)dropped+=Math.max(0,frame.presentedFrames-presented-1);
    presented=frame.presentedFrames;onDropped(dropped);
    observation={...capture(video,camera,frame.mediaTime),presented_at:frame.now};catchUp.clear();
   }
   lastMediaTime=Math.max(lastMediaTime,observation.time);
   const t=clock();let result;sampling=true;sample();
   try{result=await estimate(observation,processed)}finally{stopSampling()}
   busyMs+=clock()-t;processed++;
   if(observation.catch_up)catchUp.settle(supported(result));
   // A deferred frame keeps the last supported pose as the tracking reference; any other result without
   // support means the pipeline searches again from the newest frame.
   if(supported(result))anchor=observation.time;else if(result?.decision!=='search_deferred')anchor=null;
   if(Number.isFinite(observation.presented_at))latencies.add(clock()-observation.presented_at);
  }
 }finally{stopSampling();video.removeEventListener('ended',finish);signal?.removeEventListener('abort',finish);video.pause()}
 const wallMs=clock()-started;
 return {summary:{processed,dropped_frames:dropped,buffered_frames:buffered,wall_ms:wallMs,busy_fraction:wallMs>0?busyMs/wallMs:0,
  media_seconds:video.currentTime,latency_ms:latencies.summary(),
  scope:'newest presented frame when no pose is supported; otherwise buffered frames until the pose is current'}};
}

// A live camera has no duration and cannot be paused for acquisition; the real-time loop runs until the
// person stops it.
export async function openLiveCamera(video){
 if(!navigator.mediaDevices?.getUserMedia)throw Error('This browser cannot open a camera stream');
 const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false});
 video.muted=true;video.playsInline=true;video.srcObject=stream;
 await loadedMetadata(video,'The camera stream cannot be decoded');
 return {type:'video',source:video,live:true,duration:Infinity,close(){for(const track of stream.getTracks())track.stop();video.srcObject=null}};
}

// Waits for decoded dimensions and removes the listener that did not fire.
export function loadedMetadata(video,message){
 return new Promise((resolve,reject)=>{
  const done=()=>{video.removeEventListener?.('loadedmetadata',loaded);video.removeEventListener?.('error',failed)};
  const loaded=()=>{done();resolve()},failed=()=>{done();reject(Error(message))};
  video.addEventListener('loadedmetadata',loaded,{once:true});video.addEventListener('error',failed,{once:true});
 });
}

export function orderedInsert(frames,blobs,frame,blob){
 const at=frames.findIndex(f=>f.capture_time_ns>frame.capture_time_ns),index=at<0?frames.length:at;
 frames.splice(index,0,frame);blobs.splice(index,0,blob);return index;
}

export function grab(video,camera,mediaTime){
 const image=gray(video,camera.width,camera.height);
 return {...image,time:mediaTime,requested_time_s:mediaTime,timing:'browser decoded frame presentation timestamp; real-time playback'};
}
