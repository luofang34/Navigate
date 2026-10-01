import assert from 'node:assert/strict';
import {runRealtime,openLiveCamera} from '../webapp/realtime-video.js';

// A video that presents 30 frames per second on a virtual clock until `frames` have been shown.
function fakeVideo(frames){
  let presented=0,now=0,playing=false;const listeners=new Map();
  const video={ended:false,currentTime:0,
    async play(){playing=true},pause(){playing=false},
    addEventListener(name,fn){listeners.set(name,fn)},
    requestVideoFrameCallback(fn){setTimeout(()=>{
      if(!playing)return;
      // Frames keep being presented while an estimate runs; the next callback reports the newest one.
      presented=Math.min(frames,Math.max(presented+1,Math.floor(now*30)));video.currentTime=presented/30;
      if(presented>=frames){video.ended=true;listeners.get('ended')?.();return}
      fn(now*1000,{mediaTime:presented/30,presentedFrames:presented});
    },0)},
    advance(ms){now+=ms/1000}};
  return video;
}

const video=fakeVideo(90);
const seen=[];
const run=await runRealtime({video,camera:{width:4,height:3},clock:()=>0,
  capture:(_video,_camera,mediaTime)=>({time:mediaTime}),
  estimate:async(observation,index)=>{seen.push(observation.time);video.advance(100);return {index}}});
assert.ok(run.summary.processed>=25&&run.summary.processed<=31,`about 10 estimates per second for 3 s, got ${run.summary.processed}`);
assert.ok(run.summary.dropped_frames>=50,'frames presented during an estimate are skipped');
assert.ok(seen.every((t,i)=>i===0||t>seen[i-1]),'each estimate uses a newer presented frame');

const stop=new AbortController();const stopped=fakeVideo(900);
const partial=await runRealtime({video:stopped,camera:{width:4,height:3},clock:()=>0,signal:stop.signal,
  capture:(_video,_camera,mediaTime)=>({time:mediaTime}),
  estimate:async(_observation,index)=>{stopped.advance(100);if(index===4)stop.abort();return {index}}});
assert.equal(partial.summary.processed,5,'a stop request keeps the processed frames and ends the run');
const held=fakeVideo(900);let pauses=0;
held.pause=(()=>{const pause=held.pause;return ()=>{pauses++;pause()}})();
const acquired=await runRealtime({video:held,camera:{width:4,height:3},clock:()=>0,
  capture:(_video,_camera,mediaTime)=>({time:mediaTime}),
  estimate:async(_observation,index,hold)=>{if(index===0||index===2)hold();if(index>=3)held.ended=true;return {index}}});
assert.equal(pauses,3,'held during the two searching estimates, then paused at the end');
assert.equal(acquired.summary.processed,4);
const live=fakeVideo(900);let livePauses=0;
live.pause=(()=>{const pause=live.pause;return ()=>{livePauses++;pause()}})();
await runRealtime({video:live,camera:{width:4,height:3},clock:()=>0,holdable:false,
  capture:(_video,_camera,mediaTime)=>({time:mediaTime}),
  estimate:async(_observation,index,hold)=>{hold();if(index>=1)live.ended=true;return {index}}});
assert.equal(livePauses,1,'a live source is never held; it pauses only when the run ends');
const repeated=fakeVideo(900),times=[];
const deliver=repeated.requestVideoFrameCallback;let calls=0;
repeated.requestVideoFrameCallback=fn=>{calls++;if(calls===2){setTimeout(()=>fn(0,{mediaTime:0,presentedFrames:1}),0);return}deliver(fn)};
await runRealtime({video:repeated,camera:{width:4,height:3},clock:()=>0,
  capture:(_video,_camera,mediaTime)=>({time:mediaTime}),
  estimate:async(observation,index)=>{times.push(observation.time);repeated.advance(100);if(index>=2)repeated.ended=true;return {index}}});
assert.ok(times.every((t,i)=>i===0||t>times[i-1]),`a frame that is not newer is skipped: ${times}`);
let released=0;const stream={getTracks:()=>[{stop:()=>released++}]};
Object.defineProperty(globalThis,'navigator',{value:{mediaDevices:{getUserMedia:async constraints=>{assert.equal(constraints.audio,false);return stream}}},configurable:true});
const element={addEventListener(name,fn){if(name==='loadedmetadata')setTimeout(fn,0)}};
const camera=await openLiveCamera(element);
assert.equal(camera.live,true,'a camera stream is a live source');assert.strictEqual(element.srcObject,stream);assert.equal(camera.duration,Infinity);
camera.close();assert.equal(released,1,'closing the source releases the camera');assert.equal(element.srcObject,null);
const frozen=fakeVideo(900);frozen.requestVideoFrameCallback=()=>{};const halt=new AbortController();
const pending=runRealtime({video:frozen,camera:{width:4,height:3},clock:()=>0,signal:halt.signal,estimate:async()=>({})});
setTimeout(()=>halt.abort(),0);
assert.equal((await pending).summary.processed,0,'a stop request ends a stream that presents no further frame');
console.info('Real-time playback processes the newest frame and skips stale ones');
