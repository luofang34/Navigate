import assert from 'node:assert/strict';
import {runRealtime,openLiveCamera} from '../webapp/realtime-video.js';

// A video that presents 30 frames per second on a virtual clock until `frames` have been shown.
function fakeVideo(frames){
  let presented=0,now=0,playing=false;const listeners=new Map(),callbacks=new Set();
  // Listeners follow the DOM: each registration is kept until it is removed or fires with `once`.
  const video={ended:false,currentTime:0,listeners,callbacks,
    async play(){playing=true},pause(){playing=false},
    addEventListener(name,fn,options){if(!listeners.has(name))listeners.set(name,new Map());listeners.get(name).set(fn,options?.once===true)},
    removeEventListener(name,fn){listeners.get(name)?.delete(fn)},
    listenerCount(name){return listeners.get(name)?.size??0},
    dispatch(name){for(const [fn,once] of [...(listeners.get(name)??[])]){if(once)listeners.get(name).delete(fn);fn()}},
    cancelVideoFrameCallback(handle){callbacks.delete(handle)},
    requestVideoFrameCallback(fn){const handle=Symbol('frame');callbacks.add(handle);setTimeout(()=>{
      if(!callbacks.delete(handle)||!playing)return;
      // Frames keep being presented while an estimate runs; the next callback reports the newest one.
      presented=Math.min(frames,Math.max(presented+1,Math.floor(now*30)));video.currentTime=presented/30;
      if(presented>=frames){video.ended=true;video.dispatch('ended');return}
      fn(now*1000,{mediaTime:presented/30,presentedFrames:presented});
    },0);return handle},
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
assert.equal(video.listenerCount('ended'),0,'the run removes its end listener');
assert.equal(run.results,undefined,'the run keeps no per-frame result list');

// A long live stream never ends by itself. Listeners, pending frame callbacks, and latency records stay bounded.
const live=fakeVideo(Infinity),liveStop=new AbortController();let peakEnded=0,peakCallbacks=0;
const long=await runRealtime({video:live,camera:{width:4,height:3},clock:()=>0,signal:liveStop.signal,latencyWindow:16,
  capture:(_video,_camera,mediaTime)=>({time:mediaTime}),
  estimate:async(_observation,index)=>{peakEnded=Math.max(peakEnded,live.listenerCount('ended'));peakCallbacks=Math.max(peakCallbacks,live.callbacks.size);live.advance(40);if(index===399)liveStop.abort();return {index}}});
assert.equal(long.summary.processed,400);
assert.ok(peakEnded<=1,`one end listener for the whole run, found ${peakEnded}`);
assert.ok(peakCallbacks<=2,`frame callbacks do not build up, found ${peakCallbacks}`);
assert.equal(long.summary.latency_ms.samples,16,'latency statistics use a fixed window');
assert.equal(live.listenerCount('ended')+live.listenerCount('abort'),0,'the run removes all listeners');

const stop=new AbortController();const stopped=fakeVideo(900);
const partial=await runRealtime({video:stopped,camera:{width:4,height:3},clock:()=>0,signal:stop.signal,
  capture:(_video,_camera,mediaTime)=>({time:mediaTime}),
  estimate:async(_observation,index)=>{stopped.advance(100);if(index===4)stop.abort();return {index}}});
assert.equal(partial.summary.processed,5,'a stop request keeps the processed frames and ends the run');
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
