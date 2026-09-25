import assert from 'node:assert/strict';
import {mapViewChanged} from '../webapp/map-check-motion.js';
import {TemporalSearch} from '../webapp/temporal-search.js';
import {LocalizationPipeline} from '../webapp/localization.js';
const camera={width:960,height:544,fx:689,fy:694},context={camera,agl_m:110,fraction:.1};
const pose=(x=0,angle=0,track_id='a')=>({candidate_id:0,accepted:true,track_id,map_manifest_sha256:'map',position_enu_m:[x,0,120],eye_to_enu_xyzw:[Math.sin(angle/2),0,0,Math.cos(angle/2)]});
const oblique=pose(0,.8),saved=JSON.stringify(oblique);
assert.equal(mapViewChanged([oblique],[pose(0,.8)],context),false,'an oblique absolute attitude does not trigger a check');
assert.equal(mapViewChanged([oblique],[pose(0,.9)],context),true,'a pitch change triggers a check without constraining camera tilt');
assert.equal(mapViewChanged([pose()],[{...pose(),eye_to_enu_xyzw:[0,0,Math.sin(.05),Math.cos(.05)]}],context),true,'rotation around the viewing axis also triggers a check');
assert.equal(mapViewChanged([pose()],[{...pose(),eye_to_enu_xyzw:[0,0,0,-1]}],context),false,'quaternion sign does not mean camera motion');
assert.equal(mapViewChanged([pose()],[pose(10)],context),true);
assert.equal(mapViewChanged([pose()],[pose(10)],{...context,agl_m:1000}),false,'the nominal image scale changes check cadence');
assert.equal(mapViewChanged([pose()],[pose(0,0,'new-branch')],context),true);
assert.equal(mapViewChanged([pose(),pose(100,0,'b')],[pose(1),pose(100,0,'b')],context),false,'distant geographic alternatives are compared only with their own lineage');
assert.equal(JSON.stringify(oblique),saved,'scheduling does not alter poses');
for(const fraction of [-1,NaN,Infinity,1.1])assert.throws(()=>mapViewChanged([pose()],[pose()],{...context,fraction}),/fraction/);
assert.equal(mapViewChanged([pose()],[pose(100)],{...context,agl_m:NaN}),false,'unknown scale retains the timed fallback');

const temporal=new TemporalSearch(),anchor={observation_sha256:'a',capture_time_ns:0,candidate_hypotheses:[pose()]};
temporal.remember(anchor,undefined,{regional:true});
const motion={...context,candidates:[pose(20)]};
assert.equal(temporal.mapDue(.2e9,5,motion),false,'large image motion cannot run a map search on every sample');
assert.equal(temporal.mapDue(1e9,5,motion),true);
assert.equal(temporal.mapDue(1e9,5,{...context,candidates:[pose()]}),false);
assert.equal(temporal.mapDue(5e9,5,{...context,candidates:[pose()]}),true,'stationary cameras retain the maximum interval');
temporal.remember({...anchor,observation_sha256:'b',capture_time_ns:1e9,candidate_hypotheses:[pose(20)]},undefined,{map:true});
assert.equal(temporal.mapDue(1.2e9,5,{...context,candidates:[pose(40)]}),false);
assert.equal(temporal.regionalDue(5e9,5),true,'local motion checks cannot postpone geographic searches');

let stamp,candidate,checked,localChecks=0,areaSearches=0;
const matcher={matchImages:async()=>({pairs:[],backend_identity:'replacement-adapter'}),retrievePairs:async()=>{areaSearches++;return []}};
const pipeline=new LocalizationPipeline(matcher,{refinements:1,headings:4,mapMotionFraction:.1});
pipeline.pack={anchor_lat_lon:[0,0]};pipeline.camera=camera;pipeline.references={elevation:()=>0,crops:()=>[]};
pipeline.renderer={
 begin(_pixels,_prior,sequence,capture_time_ns){stamp={sequence,capture_time_ns,observation_sha256:`frame-${sequence}`};checked=null},
 select(){return JSON.stringify({...stamp,accepted:false,decision:'unresolved',candidate_hypotheses:checked?[checked]:[pose()]})},
 async render_reference(_id,value){candidate=JSON.parse(value);return new Uint8Array(4)},
 track(){return JSON.stringify({...pose(stamp.capture_time_ns/1e9*12),accepted:false,tracking_supported:true})},
 refine(id){localChecks++;checked={...candidate,candidate_id:id,accepted:true};return JSON.stringify(checked)}
};
globalThis.OffscreenCanvas=class{getContext(){return {drawImage(){},translate(){},rotate(){},getImageData:()=>({data:new Uint8Array(this.width*this.height*4)})}}};
const estimate=(time,sequence)=>pipeline.estimate({gray:new Uint8Array(4),width:2,height:2,canvas:{},time,timing:'decoded video'}, {latitude:0,longitude:0,agl_m:110,radius_m:500},sequence,()=>{});
await estimate(0,0);await estimate(.2,1);assert.equal(localChecks,0);
const corrected=await estimate(1,2);assert.equal(localChecks,1,'the real pipeline runs a local map check before the five-second deadline');
assert.equal(areaSearches,1);assert.equal(corrected.accepted,false,'an earlier local check cannot establish a unique geographic fix');
assert.equal(corrected.local_map_check.candidate_hypotheses[0].accepted,true);
assert.equal(corrected.candidate_hypotheses[0].relative_alternative.tracking_supported,true,'the conditional motion alternative remains traceable');
await estimate(1.2,3);assert.equal(localChecks,1);
const timed=new LocalizationPipeline(matcher,{refinements:1,headings:4});
Object.assign(timed,{pack:pipeline.pack,camera:pipeline.camera,references:pipeline.references,renderer:pipeline.renderer});
localChecks=0;areaSearches=0;
const timedEstimate=(time,sequence)=>timed.estimate({gray:new Uint8Array(4),width:2,height:2,canvas:{},time,timing:'decoded video'}, {latitude:0,longitude:0,agl_m:110,radius_m:500},sequence,()=>{});
await timedEstimate(0,0);await timedEstimate(1,1);assert.equal(localChecks,0,'the default retains the timed cadence until movement scheduling is validated');
await timedEstimate(5,2);assert.equal(localChecks,1,'the default still checks the map at the deadline');
console.info('Motion-based map checks respect full camera rotation, separate alternatives, and bounded cadence');
