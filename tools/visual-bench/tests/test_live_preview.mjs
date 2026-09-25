import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
import {TrackPreview} from '../webapp/track-preview.js';
import {hypotheses,frameLabel} from '../webapp/hypotheses.js';
import {missionSummary,resultSummary,executionSummary} from '../webapp/result-summary.js';
import {replaySeekTime} from '../webapp/video-timing.js';

class Element extends EventTarget {
 constructor(){super();this.value='';this.children=[];this.hidden=false;this.disabled=false;this.currentTime=0;this.duration=.6;this.width=640;this.height=360;this.clientWidth=640;this.dataset={};this.style={setProperty(){}};this.classList={toggle(){},remove(){}}}
 get options(){return this.children}
 replaceChildren(...children){this.children=children;this.textContent='';this.value=String(children[0]?.value??'')}
 append(...children){this.children.push(...children)}
 setAttribute(name,value){this[name]=value}
 removeAttribute(name){delete this[name]}
 load(){}
 requestVideoFrameCallback(callback){this.callback=callback;return 1}
 cancelVideoFrameCallback(){}
 getContext(){return {clearRect(){}}}
}
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});return {promise,resolve,reject}}
const nodes=new Map(),node=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id)};
for(const [id,value] of Object.entries({latitude:40.543,longitude:-74.456,radius:500,agl:110,fov:82.1,period:.2,'max-frames':10,'frame-time':0,'video-mode':'whole'}))node(id).value=String(value);
const camera={width:960,height:544},pack={pack_id:'test-pack',anchor_lat_lon:[40.543,-74.456],files:[],attribution:'Test source',elevation_datum:'unknown absolute datum'},region={id:'region',pack_id:pack.pack_id,anchor_lat_lon:pack.anchor_lat_lon,bounds:[-75,40,-74,41]};
const requests=[],waiters=[],saved=new Map(),ready=deferred(),errors=[];let activePreview;
const pipeline={beginSequence:async()=>{},finishSequence:async()=>[],estimate(image,prior,sequence,progress){const item={...deferred(),image,sequence,progress};if(waiters.length)waiters.shift()(item);else requests.push(item);return item.promise}};
const nextEstimate=()=>requests.length?Promise.resolve(requests.shift()):new Promise(resolve=>waiters.push(resolve));
const storage={get:async(store,key)=>store==='packs'?pack:saved.get(key),all:async()=>[...saved.values()],put:async(store,key,value)=>{if(store==='missions')saved.set(key,structuredClone(value))},verifyPack:async()=>true,queryBlob:async()=>new Blob(['saved']),saveLocalMission:async(view)=>{const value={id:saved.size?'saved-failure':'saved-mission',view,pack_id:pack.pack_id};saved.set(value.id,structuredClone(value));return value}};
const moves=[];
const context={setTimeout:()=>1,clearTimeout(){},console:{error:e=>errors.push(e)},URL,Blob,Option:class{constructor(text,value){this.text=text;this.value=String(value)}},Event,JSON,Math,Number,Error,Set,Map,
 document:{getElementById:node,querySelector:node,createElement:()=>new Element(),documentElement:{dataset:{}}},window:{addEventListener(){},dispatchEvent(event){if(event.type==='visual-ready')ready.resolve()}},
 navigator:{storage:{estimate:async()=>({usage:1,quota:1e9}),persisted:async()=>true}},location:{search:''},URLSearchParams,
 TrackPreview:class extends TrackPreview{constructor(...args){super(...args);activePreview=this}},
 MapView:class{constructor(canvas){this.canvas=canvas}async load(p,c){this.pack=p;this.pack_id=p.pack_id;this.preview={}}setMinimumClearance(){}async setCalibration(){}async setPose(pose){moves.push(pose)}displayCamera(){return {width:960,height:544,fx:689,fy:694,cx:479.5,cy:271.5}}height(){return 100}},
 PipelineSession:class{acquire(){this.current=pipeline;return Promise.resolve(pipeline)}close(){}},
 DataService:class{static=true;request(){return Promise.resolve([region])}},CoverageLoader:class{},
 hypotheses,frameLabel,missionSummary,resultSummary,executionSummary,replaySeekTime,
 assetUrl:p=>p,localPosition:()=>[0,0,100],selectedInputs:files=>files,cameraForImage:()=>camera,matchingOptions:()=>({longEdge:960}),
 videoTimes:()=>[0,.2,.4],openInput:async()=>({type:'video',source:{videoWidth:1920,videoHeight:1080,src:'test-video'},duration:.6,close(){}}),
 frameAt:async(_,time)=>({blob:new Blob([String(time)]),time}),requireOfflinePack:async()=>{},refineSequenceBackward:async()=>{},storage,
};
const source=(await fs.readFile(new URL('../webapp/app.js',import.meta.url),'utf8')).replace(/^import .*;\n/gm,'');
vm.runInNewContext(source,context);await ready.promise;
node('input').files=[{name:'flight.mp4',type:'video/mp4',size:10}];await node('input').onchange();
const run=node('locate').onclick(),first=await nextEstimate();
const pose=(id,x)=>({accepted:true,candidate_id:id,track_id:'path-'+id,map_manifest_sha256:'map',position_enu_m:[x,0,100],eye_to_enu_xyzw:[0,0,0,1],inliers:30});
const frame=(sequence,items)=>({capture_time_ns:sequence*200e6,requested_time_s:sequence*.2,observation_sha256:'frame-'+sequence,accepted:false,decision:items.length?'unresolved':'rejected',candidate_hypotheses:items});
first.resolve(frame(0,[pose(0,0),pose(1,20)]));const second=await nextEstimate();
assert.equal(JSON.parse(node('details').textContent).observation.observation_sha256,'frame-0');
assert.equal(JSON.parse(node('details').textContent).elevation_datum,'unknown absolute datum');
assert.equal(node('hypotheses').disabled,false,'the user can inspect alternatives while a later frame is processing');
assert.equal(node('frames').disabled,false);assert.equal(node('frame-time').disabled,false);
node('hypotheses').value='1';await node('hypotheses').onchange();
assert.equal(activePreview.key,'path-1');assert.equal(moves.at(-1).position_enu_m[0],20);
second.progress('Following image features…');
assert.match(node('run-status').textContent,/Frame 2\/3/);
assert.equal(JSON.parse(node('details').textContent).observation.observation_sha256,'frame-0','worker progress cannot replace the reviewed frame');
second.resolve(frame(1,[]));const third=await nextEstimate();
assert.equal(activePreview.key,'path-1','new results cannot replace the selected alternative at a paused time');
assert.equal(node('hypotheses').value,'1');
assert.equal(JSON.parse(node('details').textContent).observation.observation_sha256,'frame-0');
node('track-camera').onclick();assert.match(node('map-label').textContent,/SELECTED HYPOTHESIS/,'camera view labels the displayed estimate');
node('track-follow').checked=true;node('track-follow').dispatchEvent(new Event('change'));
node('video').currentTime=.3;node('video').dispatchEvent(new Event('seeked'));
assert.match(node('result').textContent,/No supported camera pose/);assert.equal(node('details').textContent,'');assert.equal(node('compute-status').textContent,'');assert.equal(node('hypotheses').disabled,true);
assert.match(node('map-label').textContent,/NO SUPPORTED CAMERA POSE/);
assert.equal(node('frame-time').value,'0.3');assert.equal(node('frame-time-label').textContent,'0.30 / 0.60 s');
node('frames').value='0';await node('frames').onchange();
assert.equal(node('video').currentTime,replaySeekTime(frame(0,[])),'frame selection seeks the uploaded video during processing');
assert.equal(JSON.parse(node('details').textContent).observation.observation_sha256,'frame-0');
third.resolve(frame(2,[]));await run;
assert.equal(saved.get('saved-mission').view.frames.length,3);assert.equal(errors.length,0);
assert.equal(node('cancel').disabled,true);assert.equal(node('frames').disabled,false);
assert.equal(node('map-label').textContent,'TRACK OVERVIEW');
const reads=[],readStarted=deferred();storage.queryBlob=(mission,frame)=>{const request={...deferred(),frame};reads.push(request);readStarted.resolve();return request.promise};
node('missions').value='saved-mission';const loaded=node('missions').onchange();
await readStarted.promise;reads.shift().resolve(new Blob(['frame 0']));await loaded;
node('frames').value='1';const older=node('frames').onchange();
node('frames').value='2';const newer=node('frames').onchange();
reads[1].resolve(new Blob(['frame 2']));await newer;
reads[0].resolve(new Blob(['frame 1']));await older;
assert.equal(node('frames').value,'2','a slow earlier read cannot replace the latest selected frame');
assert.equal(JSON.parse(node('details').textContent).observation.observation_sha256,'frame-2');
reads.length=0;node('video').hidden=false;node('video').currentTime=0;node('video').dispatchEvent(new Event('seeked'));
node('frames').value='0';const beforePlayback=node('frames').onchange();
node('video').currentTime=.3;node('video').dispatchEvent(new Event('seeked'));
assert.match(node('result').textContent,/No supported camera pose/);
const movesBeforeRead=moves.length;
reads[0].resolve(new Blob(['late frame 0']));await beforePlayback;
assert.match(node('result').textContent,/No supported camera pose/,'a pending saved-frame read cannot replace newer playback evidence');
assert.equal(node('details').textContent,'');
assert.equal(moves.length,movesBeforeRead,'a late frame read cannot move the camera away from current playback');
node('region').value='';await node('region').onchange();
assert.equal(node('frames').disabled,true);assert.equal(node('hypotheses').disabled,true);
assert.equal(node('details').textContent,'');assert.equal(node('export-track').disabled,true);
activePreview.close();
console.info('Live upload preserves the reviewed frame and alternative, supports seeking, and clears unsupported-time evidence.');
