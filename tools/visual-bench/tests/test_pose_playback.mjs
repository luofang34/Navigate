import assert from 'node:assert/strict';
import {interpolatePose,playbackPose,VideoPoseClock} from '../webapp/pose-playback.js';
const pose=(x,angle=0)=>({tracking_supported:true,candidate_id:0,anchor_observation_sha256:'a',map_manifest_sha256:'m',position_enu_m:[x,0,100],eye_to_enu_xyzw:[0,0,Math.sin(angle/2),Math.cos(angle/2)]});
const samples=[{time:1,h:pose(0)},{time:1.2,h:pose(20,Math.PI/2)}];
const middle=playbackPose(samples,1.1);
assert.ok(Math.abs(middle.pose.position_enu_m[0]-10)<1e-10);
assert.ok(Math.abs(middle.pose.eye_to_enu_xyzw[2]-Math.sin(Math.PI/8))<1e-10);
assert.equal(middle.interpolated,true);
assert.equal(middle.pose.accepted,undefined,'display interpolation does not create an accepted estimate');
assert.equal(playbackPose(samples,.9),null);assert.equal(playbackPose(samples,1.21),null);
for(const field of ['candidate_id','anchor_observation_sha256','map_manifest_sha256']){
 const changed=structuredClone(samples);changed[1].h[field]='different';assert.equal(playbackPose(changed,1.1),null,field);
}
for(const change of [{tracking_supported:false},{continuity_break:true},{tracking_anchor:{observation_sha256:'b'}}]){
 const changed=structuredClone(samples);Object.assign(changed[1].h,change);assert.equal(playbackPose(changed,1.1),null);
}
assert.equal(playbackPose([{time:1,h:pose(0)},{time:4,h:pose(10)}],2),null);
const opposite=pose(2);opposite.eye_to_enu_xyzw=[0,0,0,-1];
assert.deepEqual(interpolatePose(pose(0),opposite,.5).eye_to_enu_xyzw,[0,0,0,1]);
const a=pose(0,179*Math.PI/180),b=pose(0,-179*Math.PI/180);
assert.ok(Math.abs(interpolatePose(a,b,.5).eye_to_enu_xyzw[3])<1e-10,'rotation crosses the short arc');
class Video extends EventTarget {currentTime=9;callback=null;requestVideoFrameCallback(fn){this.callback=fn;return 7}cancelVideoFrameCallback(id){assert.equal(id,7);this.callback=null}}
const video=new Video(),times=[],clock=new VideoPoseClock(video,t=>times.push(t));
video.callback(0,{mediaTime:1.123});assert.deepEqual(times,[1.123],'follow uses the decoded presentation timestamp');
assert.equal(clock.time,1.123,'controls use the timestamp of the displayed frame, not the requested seek time');
video.dispatchEvent(new Event('pause'));assert.deepEqual(times,[1.123,1.123]);
video.dispatchEvent(new Event('seeking'));assert.equal(clock.time,9,'a new seek invalidates the previous decoded frame');
video.dispatchEvent(new Event('seeked'));assert.deepEqual(times,[1.123,1.123,9]);
video.callback(0,{mediaTime:9.033});assert.equal(clock.time,9.033);
video.dispatchEvent(new Event('seeked'));assert.equal(times.at(-1),9.033,'seek completion does not replace a newer decoded timestamp');
clock.close();assert.equal(video.callback,null);const count=times.length;
video.dispatchEvent(new Event('seeked'));assert.equal(times.length,count);
video.dispatchEvent(new Event('seeking'));assert.equal(clock.time,9.033,'closed clock releases invalidation listeners');

const unknownMap=structuredClone(samples);for(const sample of unknownMap)delete sample.h.map_manifest_sha256;assert.equal(playbackPose(unknownMap,1.1),null,'unknown map identity does not establish a shared interpolation frame');
