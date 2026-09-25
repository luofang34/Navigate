import assert from 'node:assert/strict';
import {DecodedVideoTime,replaySeekTime} from '../webapp/video-timing.js';
let next=0;const callbacks=new Map(),cancelled=[];
const video={currentTime:.3,requestVideoFrameCallback(callback){const id=next=(next+1)>>>0;callbacks.set(id,callback);return id},cancelVideoFrameCallback(id){cancelled.push(id);callbacks.delete(id)}};
function decoded(time){const [id,callback]=callbacks.entries().next().value;callbacks.delete(id);callback(0,{mediaTime:time})}
const clock=new DecodedVideoTime(video),first=clock.read();
decoded(.266667);assert.equal(await first,.266667);assert.equal(await clock.read(),.266667);
clock.beforeSeek();assert.ok(cancelled.length>0);const sought=clock.read();
decoded(.6);assert.equal(await sought,.6,'a new seek cannot reuse a cached decoded timestamp');
clock.beforeSeek();const pending=clock.read();clock.close();assert.equal(await pending,null);assert.equal(callbacks.size,0);assert.equal(await clock.read(),null);
const unsupported=new DecodedVideoTime({});assert.equal(await unsupported.read(),null);unsupported.close();
console.info('Decoded timestamps replace seek requests; seeks invalidate stale values and close releases pending readers');

const frame={requested_time_s:10.2,capture_time_ns:10176833000};
assert.equal(replaySeekTime(frame),10.2,'saved frame selection repeats the decoder request that produced its pixels');
assert.equal(frame.capture_time_ns,10176833000,'replay does not rewrite the pose timestamp');
assert.equal(replaySeekTime({capture_time_ns:2500000000}),2.5,'legacy frames retain their only available seek time');
for(const requested_time_s of [-1,NaN,Infinity])assert.throws(()=>replaySeekTime({requested_time_s,capture_time_ns:0}),/invalid/);
assert.throws(()=>replaySeekTime({}),/invalid/);

const visibility=new EventTarget();visibility.hidden=true;
const hidden=new DecodedVideoTime(video,{visibility});let completed=false;const hiddenRead=hidden.read().then(v=>{completed=true;return v});
decoded(.7);await Promise.resolve();assert.equal(completed,false,'a hidden page does not time out into a seek-time observation');
visibility.hidden=false;visibility.dispatchEvent(new Event('visibilitychange'));assert.equal(await hiddenRead,.7);
hidden.beforeSeek();visibility.hidden=true;const cancelledHidden=hidden.read();hidden.close();assert.equal(await cancelledHidden,null);assert.equal(hidden.visibilityWaiters.size,0,'close releases visibility listeners');
console.info('Background sampling waits for visibility and retains decoded timestamps without blocking cancellation.');
