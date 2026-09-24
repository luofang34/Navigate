import assert from 'node:assert/strict';
import {selectedInputs,sequenceTrack} from '../webapp/input-sequence.js';
const file=(name,type='image/png')=>({name,type,size:10});
assert.deepEqual(selectedInputs([file('frame10.png'),file('frame2.png')]).map(f=>f.name),['frame2.png','frame10.png']);
assert.throws(()=>selectedInputs([file('a'),file('v','video/mp4')]));
assert.throws(()=>selectedInputs(Array.from({length:121},()=>file('a'))));
const h=(id,lon,map='release-a')=>({accepted:true,candidate_id:id,track_id:`branch-${id}`,longitude_deg:lon,latitude_deg:40,altitude_m:110,map_manifest_sha256:map,eye_to_enu_xyzw:[0,0,0,1]});
const frame=(id,candidates)=>({observation_sha256:id,decision:'unresolved',candidate_hypotheses:candidates,timing_scope:'still image',capture_time_ns:0});
const track=sequenceTrack([frame('a',[h(0,-74),h(1,-75)]),frame('b',[h(0,-73),h(1,-76)]),frame('c',[]),frame('d',[h(0,-72)]),frame('e',[h(0,-71,'release-b')])]);
const points=track.features.filter(f=>f.geometry.type==='Point'),lines=track.features.filter(f=>f.geometry.type==='LineString');
assert.equal(points.length,6);assert.equal(lines.length,2);assert.deepEqual(lines[0].geometry.coordinates,[[-74,40],[-73,40]]);assert.equal(lines[0].properties.kind,'illustrative hypothesis path');assert.deepEqual(lines[0].properties.observations,['a','b']);assert.deepEqual(lines[1].geometry.coordinates,[[-75,40],[-76,40]]);assert.equal(track.gaps[0].observation_sha256,'c');assert.equal(points[0].geometry.coordinates.length,2);assert.equal(points[0].properties.altitude_m,110);assert.match(track.sequence_scope,/times are unknown/);assert.match(lines[0].properties.correlation,/unknown/);
console.info('Image ordering, mixed-input rejection, distinct alternatives, track gaps and reference-version boundaries passed');
assert.throws(()=>selectedInputs([{...file('large.png'),size:300*1024*1024}]),/image/);
assert.equal(selectedInputs([{...file('flight.mp4','video/mp4'),size:4*1024**3}]).length,1,'a streamed video does not require a whole-file memory buffer');
const conditional={...h(1,-73),accepted:false,tracking_supported:true,tracking_anchor:{observation_sha256:'anchor'}};
const relativeTrack=sequenceTrack([frame('anchor',[h(0,-74)]),{...frame('relative',[conditional]),decision:'relative_tracking'}]);
const relativePoint=relativeTrack.features.find(f=>f.properties.observation_sha256==='relative'||f.properties.frame_index===1);
assert.equal(relativePoint.properties.accepted,false);assert.equal(relativePoint.properties.tracking_supported,true);assert.equal(relativePoint.properties.tracking_anchor.observation_sha256,'anchor');
assert.equal(sequenceTrack([frame('unknown1',[h(0,-74,null)]),frame('unknown2',[h(0,-73,null)])]).features.filter(f=>f.geometry.type==='LineString').length,0,'unknown map identity does not imply a shared reference frame');

const {videoTimes}=await import('../webapp/input-sequence.js');
const complete=videoTimes(146.55,69,{mode:'whole',period:.2});
assert.equal(complete[0],0);assert.equal(complete.length,733);assert.ok(complete.at(-1)>146.3);
assert.deepEqual(videoTimes(146,69,{mode:'frame'}),[69]);
assert.deepEqual(videoTimes(146,69,{mode:'sequence',period:1,maxFrames:3}),[69,70,71]);
assert.throws(()=>videoTimes(146,69,{period:0}),/interval/);
assert.throws(()=>videoTimes(6000,0,{period:.1}),/5000/);

const unassociated=[frame('independent-a',[{...h(0,-74),track_id:undefined}]),frame('independent-b',[{...h(0,-73),track_id:undefined}])];
assert.equal(sequenceTrack(unassociated).features.filter(f=>f.geometry.type==='LineString').length,0,'candidate number reuse does not establish a track identity');
const restart=sequenceTrack([frame('s0',[h(0,-74)]),frame('s1',[h(0,-73)]),frame('s2',[{...h(0,-72),continuity_break:true}]),frame('s3',[h(0,-71)])]);
assert.deepEqual(restart.features.filter(f=>f.geometry.type==='LineString').map(f=>f.geometry.coordinates),[[[-74,40],[-73,40]],[[-72,40],[-71,40]]],'map restarts remain disconnected');
const videoFrame=(name,time)=>({...frame(name,[h(0,-74)]),timing_scope:'video',capture_time_ns:time});
assert.equal(sequenceTrack([videoFrame('v0',100),videoFrame('v1',50)]).features.filter(f=>f.geometry.type==='LineString').length,0,'backward video time breaks the path');
assert.equal(sequenceTrack([frame('same',[h(0,-74)]),frame('same',[h(0,-73)])]).features.filter(f=>f.geometry.type==='LineString').length,0,'reused evidence does not create a new path segment');
console.info('Exported paths preserve explicit alternatives, observation identity, time order and map restarts');

assert.equal(sequenceTrack([videoFrame('unknown-time',undefined),videoFrame('known-time',100)]).features.filter(f=>f.geometry.type==='LineString').length,0,'unknown timing does not imply an earlier timestamp');
assert.equal(sequenceTrack([frame(undefined,[h(0,-74)]),frame('identified',[h(0,-73)])]).features.filter(f=>f.geometry.type==='LineString').length,0,'missing observation identity cannot establish continuity');
