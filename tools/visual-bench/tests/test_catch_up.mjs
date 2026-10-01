import assert from 'node:assert/strict';
import {CatchUp,runRealtime} from '../webapp/realtime-video.js';

const buffer=new CatchUp({rateHz:2,maxSeconds:10});
for(let step=0;step<=200;step++){const time=step/10;if(buffer.wants(time))buffer.add({time})}
assert.equal(buffer.frames[0].time,10,'the buffer keeps only the newest maxSeconds of frames');
assert.equal(buffer.frames.length,21,'frames are sampled at the buffer rate');
assert.equal(buffer.next(10).time,11,'a supported pose tracks one hop forward');
buffer.settle(true);assert.equal(buffer.hop,1.5);
const hop=buffer.next(11);assert.equal(hop.time,12.5,'a supported hop widens the next hop');assert.equal(hop.catch_up,true);
buffer.settle(false);assert.equal(buffer.hop,.75);
assert.equal(buffer.next(11).time,12,'a failed hop retries closer to the same supported pose');
buffer.settle(false);buffer.settle(false);assert.equal(buffer.hop,.5);
assert.equal(buffer.next(11).time,11.5,'the hop never falls below its minimum');
assert.equal(buffer.next(11).time,13,'a failed frame is not tried again');
assert.equal(buffer.next(19.5),null,'within one hop of the newest frame the loop uses the newest frame');
assert.equal(new CatchUp().next(0),null,'an empty buffer has nothing to catch up');

// A live source keeps presenting frames while a slow first estimate runs. After the pose is supported,
// the loop processes buffered frames forward instead of jumping to the newest frame.
let now=0,presented=0;const listeners=new Map();
const video={ended:false,currentTime:0,async play(){},pause(){},addEventListener(name,fn){listeners.set(name,fn)},
 requestVideoFrameCallback(fn){setTimeout(()=>{if(video.ended)return;presented=Math.max(presented+1,Math.floor(now*30));video.currentTime=presented/30;fn(now*1000,{mediaTime:presented/30,presentedFrames:presented})},0)}};
const times=[];
await runRealtime({video,camera:{width:4,height:3},clock:()=>now*1000,catchUp:new CatchUp({rateHz:2}),
 capture:(_video,_camera,mediaTime)=>({time:mediaTime}),
 estimate:async(observation,index)=>{times.push(observation.time);
  // The first estimate takes ten seconds of source time; later ones take a quarter second.
  const cost=index===0?10:.25;const until=now+cost;while(now<until){now=Math.min(until,now+1/30);await new Promise(r=>setTimeout(r,0))}
  if(now>16)video.ended=true;return {candidate_hypotheses:[{tracking_supported:true}]}}});
assert.ok(times[1]<3,`the first catch-up frame is near the acquired frame, not the newest: ${times.slice(0,4)}`);
const caught=times.findIndex((t,i)=>i>0&&t>=10);assert.ok(caught>3,`several buffered frames precede the current frame: ${times.map(t=>t.toFixed(2))}`);
assert.ok(16-times.at(-1)<1,`the pose is current again before the source ends: ${times.at(-1)}`);
assert.ok(times.every((t,i)=>i===0||t>times[i-1]),'catch-up frames move forward in time');

// A deferred catch-up frame keeps the supported anchor; the loop retries closer to it.
now=0;presented=0;video.ended=false;const retried=[];
await runRealtime({video,camera:{width:4,height:3},clock:()=>now*1000,catchUp:new CatchUp({rateHz:2,maxHop:4}),
 capture:(_video,_camera,mediaTime)=>({time:mediaTime}),
 estimate:async(observation,index)=>{retried.push(observation.time);
  const cost=index===0?10:.25;const until=now+cost;while(now<until){now=Math.min(until,now+1/30);await new Promise(r=>setTimeout(r,0))}
  if(index>=3)video.ended=true;
  return index===1?{decision:'search_deferred',candidate_hypotheses:[]}:{candidate_hypotheses:[{tracking_supported:true}]}}});
assert.ok(retried[2]<retried[1]&&retried[2]>retried[0],`a failed hop is retried closer to the supported frame: ${retried.map(t=>t.toFixed(2))}`);
console.info('Catch-up tracks forward through buffered frames until the pose is current');
