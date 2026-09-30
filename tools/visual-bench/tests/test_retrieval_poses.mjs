import assert from 'node:assert/strict';
import {matchingOptions} from '../webapp/matching-options.js';
import {cropPoseCandidates} from '../webapp/retrieval-poses.js';
import {LocalizationPipeline} from '../webapp/localization.js';
import {refineCandidates} from '../webapp/temporal-search.js';
const region={crop:'map/crop',crop_center:[20,30,10],angle:255},prior={agl_m:110,radius_m:500};
const seeds=cropPoseCandidates([region,{...region,angle:260},{...region,crop:'map/other'}],[0,0,120],prior,1);
assert.equal(seeds.length,3);
for(const [i,yaw] of [255,240,270].entries()){
 assert.deepEqual(seeds[i].position_enu_m,[20,30,120]);
 assert.ok(Math.abs(seeds[i].eye_to_enu_xyzw[2]-Math.sin(yaw*Math.PI/360))<1e-12);
 assert.equal(Object.hasOwn(seeds[i],'accepted'),false);
 assert.equal(Object.hasOwn(seeds[i],'covariance'),false);
}
assert.equal(cropPoseCandidates([region,region,{...region,crop:'map/other'}],[0,0,120],prior,2).length,6);
assert.deepEqual(cropPoseCandidates([region],[0,0,120],prior,0),[]);
assert.deepEqual(cropPoseCandidates([{...region,crop_center:null},{...region,crop_center:[0,0,NaN]}],[0,0,120],prior,1),[]);
assert.deepEqual(cropPoseCandidates([region],[2000,0,120],prior,1),[]);
assert.deepEqual(cropPoseCandidates([region],[0,0,120],{...prior,agl_m:NaN},1),[]);
assert.throws(()=>cropPoseCandidates([],[],prior,9),/budget/);
await assert.rejects(refineCandidates({}, {}, {}, [], {}, 'observation',1,()=>{},{firstCandidateId:-1}),/ID range/);

globalThis.OffscreenCanvas=class {
 constructor(width,height){Object.assign(this,{width,height})}
 getContext(){return {drawImage(){},translate(){},rotate(){},getImageData:()=>({data:new Uint8Array(this.width*this.height*4)})}}
};
const pose={position_enu_m:[0,0,100],eye_to_enu_xyzw:[.1,0,0,Math.sqrt(.99)]};
const image={gray:new Uint8Array(4),width:2,height:2,canvas:{},time:0,timing:'still image'};
for(const initialAcceptance of [false,true]){
 const evaluated=new Map(),keys=[],ids=[];
 const pipeline=new LocalizationPipeline({
  async retrievePairs(){return [{reference_index:0,query_index:0}]},
  async matchImages(_reference,_query,context){keys.push(context);return {pairs:Array.from({length:12},()=>({reference:[0,0],query:[0,0]})),backend_identity:'independent-adapter'}}
 },{...matchingOptions('balanced'),temporal:false,headings:4,candidates:1,refinements:1});
 pipeline.camera={width:2,height:2};pipeline.pack={anchor_lat_lon:[0,0]};
 pipeline.references={elevation:()=>0,crops:()=>[{key:'map/crop',image,world:()=>[0,0,0]}]};
 pipeline.propose=()=>JSON.stringify({retrieved:true,retrieval_inliers:12,...pose});
 pipeline.renderer={
  begin(){evaluated.clear()},async render_reference(id,value){ids.push(id);evaluated.set(id,{candidate_id:id,reference_pose:JSON.parse(value)});return image.gray},
  refine(id){const result={...evaluated.get(id),accepted:initialAcceptance||id===1,...pose};evaluated.set(id,result);return JSON.stringify(result)},
  select:()=>JSON.stringify({accepted:false,observation_sha256:'same-still',candidate_hypotheses:[...evaluated.values()]})
 };
 const result=await pipeline.estimate(image,{latitude:0,longitude:0,agl_m:100,radius_m:500},0,()=>{});
 assert.deepEqual(ids,initialAcceptance?[0]:[0,1,2,3]);
 assert.deepEqual(result.candidate_hypotheses.map(h=>h.candidate_id),ids,'fallback does not replace existing hypotheses');
 assert.deepEqual(result.candidate_hypotheses[0].reference_pose.eye_to_enu_xyzw,pose.eye_to_enu_xyzw,'the initial arbitrary camera orientation stays intact');
 assert.equal(result.retrieval.crop_pose_candidates,initialAcceptance?0:3);
 assert.equal(result.retrieval.evaluated_candidates,ids.length);
 const refinement=keys.filter(k=>k.stage==='refinement');
 assert.ok(refinement.every(k=>k.query==='same-still/query'&&!k.recovery),'extra poses use the same observation and the normal adapter budget');
}
console.info('Bounded crop poses retain arbitrary original poses, evidence identity, prior limits, and separate acceptance');
