import assert from 'node:assert/strict';
import {refineCandidates} from '../webapp/temporal-search.js';
import {LocalMatcher} from '../webapp/inference/local.js';
import {matchingOptions} from '../webapp/matching-options.js';
const pose={position_enu_m:[0,0,100],eye_to_enu_xyzw:[0,0,0,1]},image={gray:new Uint8Array(4),width:2,height:2};
const keys=[],results=new Map();let active,expanded=false;
const matcher={async matchImages(_reference,_query,context){keys.push({...context});expanded=context.recovery;return {pairs:[],backend_identity:expanded?'detail':'normal'}}};
const renderer={
 async render_reference(id){active=id;return image.gray},
 refine(id,_pairs,backend){const report=id===0&&!expanded?{accepted:false,refinement_proposal:pose}:{accepted:true,...pose};results.set(id,{...report,candidate_id:id,backend});return JSON.stringify(report)},
 select(){return JSON.stringify({accepted:false,candidate_hypotheses:[...results.values()]})},
};
const report=await refineCandidates(renderer,matcher,image,[pose,pose],image,'same-observation',3,()=>{});
assert.deepEqual(keys.map(k=>k.recovery),[false,true,false],'a failed converged seed retries, and another candidate starts at normal cost');
assert.ok(keys.every(k=>k.query==='same-observation/query'),'a retry retains the same evidence identity');
assert.deepEqual(report.candidate_hypotheses.map(h=>h.backend),['detail','normal']);
assert.equal(report.accepted,false,'separate candidate evaluations do not imply geographic uniqueness');
assert.equal(active,1);
keys.length=0;renderer.refine=()=>JSON.stringify({accepted:false,reason:'missing reference data'});
await refineCandidates(renderer,matcher,image,[pose],image,'missing',3,()=>{});
assert.deepEqual(keys.map(k=>k.recovery),[false],'unsupported geometry does not consume a detail retry');
keys.length=0;renderer.refine=()=>JSON.stringify({accepted:false,refinement_proposal:{...pose,position_enu_m:[keys.length,0,100]}});
await refineCandidates(renderer,matcher,image,[pose],image,'bounded',3,()=>{});
assert.deepEqual(keys.map(k=>k.recovery),[false,true,true],'recovery remains within the configured pass budget');

const adapter=new LocalMatcher({refinementPatches:'on_rejection'}),calls=[];
adapter.denseAsset={sha256:'model'};adapter.dense={match:async()=>{calls.push('normal');return []},refine:async()=>{calls.push('patches');return []}};
for(const context of [{stage:'retrieval',recovery:true},{stage:'refinement',recovery:false},{stage:'refinement',recovery:true},{stage:'tracking',recovery:true}])await adapter.matchImages(image,image,context);
assert.deepEqual(calls,['normal','normal','patches','normal'],'the concrete adapter owns patch preprocessing and applies it only to requested refinement recovery');
for(const mode of ['balanced','detailed'])assert.equal(matchingOptions(mode).refinementPatches,'on_rejection');
assert.equal(matchingOptions('fast').refinementPatches,undefined);
console.info('Refinement retries keep evidence and candidate identities, bound work, preserve data rejection, and leave patch processing inside the adapter');

keys.length=0;
await refineCandidates(renderer,matcher,image,[pose],image,'broad-search',3,()=>{},{recoveryAllowed:false});
assert.deepEqual(keys.map(k=>k.recovery),[false,false,false],'broad proposals keep the normal matching budget when detail recovery is not useful');

const {LocalizationPipeline}=await import('../webapp/localization.js');
globalThis.OffscreenCanvas=class {
 constructor(width,height){Object.assign(this,{width,height})}
 getContext(){return {drawImage(){},translate(){},rotate(){},getImageData:()=>({data:new Uint8Array(this.width*this.height*4)})}}
};
const acquisitionCalls=[],acquisition=new LocalizationPipeline({
 async retrievePairs(){return [{reference_index:0,query_index:0}]},
 async matchImages(_a,_b,context){
  if(context.stage==='refinement')acquisitionCalls.push(context.recovery);
  return {pairs:Array.from({length:12},()=>({reference:[0,0],query:[0,0]})),backend_identity:'replaceable-adapter'};
 },
},{headings:4,candidates:1,refinements:3,temporal:false});
acquisition.camera={width:2,height:2};acquisition.pack={anchor_lat_lon:[0,0]};
acquisition.references={elevation:()=>0,crops:()=>[{key:'map-crop',image,world:()=>[0,0,0]}]};
acquisition.propose=()=>JSON.stringify({retrieved:true,retrieval_inliers:12,...pose});
acquisition.renderer={begin(){},render_reference:async()=>image.gray,
 refine:()=>JSON.stringify({accepted:false,refinement_proposal:{...pose,position_enu_m:[acquisitionCalls.length,0,100]}}),
 select:()=>JSON.stringify({observation_sha256:'independent-still',capture_time_ns:0,candidate_hypotheses:[],accepted:false}),
};
await acquisition.estimate({...image,canvas:{},time:0,timing:'still image'},{latitude:0,longitude:0,agl_m:100,radius_m:500},0,()=>{});
assert.deepEqual(acquisitionCalls,[false,false,false],'the geographic workflow applies the normal budget to coarse map proposals');
