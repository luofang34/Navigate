import {gray} from './observation.js';

// Real-time processing follows the playback clock. Frames that arrive while an estimate runs are
// dropped; each estimate uses the newest presented frame, so latency stays bounded by one estimate.
export function presentedFrame(video){
 return new Promise((resolve,reject)=>{
  if(typeof video.requestVideoFrameCallback!=='function'){reject(Error('This browser cannot report presented video frames'));return}
  video.requestVideoFrameCallback((now,metadata)=>resolve({mediaTime:metadata.mediaTime,presentedFrames:metadata.presentedFrames,now}));
 });
}

// `estimate(observation,index,hold)` calls `hold()` when it starts work that takes longer than the
// frame interval, such as an area search. A file source pauses for the rest of that estimate, so the
// search does not consume the recording; a live source cannot pause and resumes from its newest frame.
export async function runRealtime({video,camera,estimate,signal,holdable=true,capture=grab,onDropped=()=>{},clock=()=>performance.now()}){
 const results=[];let presented=0,lastMediaTime=-Infinity,dropped=0,busyMs=0,heldMs=0;const started=clock();
 const ended=()=>video.ended||Boolean(signal?.aborted);
 // A stopped or frozen live stream presents no further frame, so a stop request must not wait for one.
 const stopped=new Promise(resolve=>signal?.addEventListener('abort',()=>resolve(null),{once:true}));
 await video.play();
 try{
  while(!ended()){
   const frame=await Promise.race([presentedFrame(video),stopped,new Promise(resolve=>video.addEventListener('ended',()=>resolve(null),{once:true}))]);
   if(!frame||ended())break;
   if(frame.mediaTime<=lastMediaTime)continue;
   if(presented)dropped+=Math.max(0,frame.presentedFrames-presented-1);
   presented=frame.presentedFrames;lastMediaTime=frame.mediaTime;onDropped(dropped);
   const observation=capture(video,camera,frame.mediaTime),t=clock();let heldAt=null;
   const hold=()=>{if(holdable&&heldAt===null){heldAt=clock();video.pause()}};
   let result;
   try{result=await estimate(observation,results.length,hold)}finally{if(heldAt!==null&&!ended())await video.play()}
   busyMs+=clock()-t;if(heldAt!==null)heldMs+=clock()-heldAt;
   results.push(result);
  }
 }finally{video.pause()}
 const wallMs=clock()-started;
 return {results,summary:{processed:results.length,dropped_frames:dropped,wall_ms:wallMs,held_ms:heldMs,busy_fraction:wallMs>0?busyMs/wallMs:0,
  media_seconds:video.currentTime,scope:'newest presented frame per estimate; frames presented during an estimate are not processed'}};
}


// A live camera has no duration and cannot be paused for acquisition; the real-time loop runs until the
// person stops it.
export async function openLiveCamera(video){
 if(!navigator.mediaDevices?.getUserMedia)throw Error('This browser cannot open a camera stream');
 const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false});
 video.muted=true;video.playsInline=true;video.srcObject=stream;
 await new Promise((resolve,reject)=>{video.addEventListener('loadedmetadata',resolve,{once:true});video.addEventListener('error',()=>reject(Error('The camera stream cannot be decoded')),{once:true})});
 return {type:'video',source:video,live:true,duration:Infinity,close(){for(const track of stream.getTracks())track.stop();video.srcObject=null}};
}

export function grab(video,camera,mediaTime){
 const image=gray(video,camera.width,camera.height);
 return {...image,time:mediaTime,requested_time_s:mediaTime,timing:'browser decoded frame presentation timestamp; real-time playback'};
}
