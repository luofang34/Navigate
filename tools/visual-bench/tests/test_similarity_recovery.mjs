import {matchingOptions} from '../webapp/matching-options.js';
import assert from 'node:assert/strict';
import {similarityCandidates} from '../webapp/retrieval-recovery.js';
import {LocalizationPipeline} from '../webapp/localization.js';
import {refineReferenceCandidates} from '../webapp/candidate-refinement.js';

const pose=x=>({position_enu_m:[x,0,100],eye_to_enu_xyzw:[0,0,0,1]});
const records=Array.from({length:40},(_,source_index)=>({source_index,ground:[source_index]}));
const propose=(_camera,ground,threshold)=>JSON.stringify({retrieved:true,retrieval_inliers:20,retrieval_support:JSON.parse(ground)[0],...pose(threshold===4?10:20)});
const seeds=similarityCandidates(records,propose,{},[0,0,100],500);
assert.equal(seeds.length,64);
assert.deepEqual(seeds.slice(0,4).map(s=>[s.source_index,s.seed_threshold_px]),[[39,4],[39,12],[38,4],[38,12]]);
assert.equal(similarityCandidates(records,()=>JSON.stringify({retrieved:false}),{},[0,0,100],500).length,0);
assert.equal(similarityCandidates(records,propose,{},[1000,0,100],10).length,0);

globalThis.OffscreenCanvas=class {
 constructor(width,height){Object.assign(this,{width,height})}
 getContext(){return {drawImage(){},translate(){},rotate(){},getImageData:()=>({data:new Uint8Array(this.width*this.height*4)})}}
};
const recoveredOrientation=[.1,-.2,0,Math.sqrt(.95)];
const image={gray:new Uint8Array(4),width:2,height:2,canvas:{},time:0,timing:'still image'};
for(const [enabled,initialAcceptance] of [[true,false],[true,true],[false,false]]){
 const reports=new Map(),ids=[],keys=[],seedCalls=[];
 const matcher={
  async retrievePairs(){return [{reference_index:0,query_index:0}]},
  async matchImages(_r,_q,context){keys.push(context);return {pairs:Array.from({length:12},()=>({reference:[0,0],query:[0,0]})),backend_identity:'replacement-adapter'}}
 };
 const pipeline=new LocalizationPipeline(matcher,{temporal:false,headings:4,candidates:1,refinements:1,similarityCandidates:enabled?1:0});
 pipeline.camera=image;pipeline.pack={anchor_lat_lon:[0,0]};pipeline.references={elevation:()=>0,crops:()=>[{key:'map/crop',image,world:()=>[0,0,0]}]};
 pipeline.propose=()=>JSON.stringify({retrieved:true,retrieval_inliers:12,...pose(0)});
 pipeline.proposeNadir=(camera,pairs,threshold)=>{seedCalls.push(threshold);return JSON.stringify({retrieved:true,retrieval_inliers:9,retrieval_support:8,...pose(threshold)})};
 pipeline.renderer={
  begin(){reports.clear()},
  async render_reference(id,value){ids.push(id);reports.set(id,{candidate_id:id,reference_pose:JSON.parse(value)});return image.gray},
  refine(id){const report={...reports.get(id),accepted:initialAcceptance||id===1,...pose(id),eye_to_enu_xyzw:recoveredOrientation,map_manifest_sha256:'same-map'};reports.set(id,report);return JSON.stringify(report)},
  select:()=>JSON.stringify({accepted:false,decision:'unresolved',observation_sha256:'same-image',candidate_hypotheses:[...reports.values()]})
 };
 const report=await pipeline.estimate(image,{latitude:0,longitude:0,agl_m:100,radius_m:500},0,()=>{});
 const recovery=enabled&&!initialAcceptance;
 assert.deepEqual(seedCalls,recovery?[4,12]:[],'skip optional work when normal geometry accepts');
 assert.deepEqual([...new Set(ids)],recovery?[0,1]:[0],'extra candidates cannot overwrite earlier hypotheses');
 assert.ok(keys.filter(k=>k.stage==='refinement').every(k=>k.query==='same-image/query'&&!k.recovery));
 assert.equal(report.retrieval.stage,'retrieval_only');assert.equal(report.accepted,false);assert.equal(report.decision,'unresolved');
 if(recovery){
  assert.deepEqual(report.seed_verification_work.evaluated_candidate_ids,[1]);
  assert.equal(report.seed_verification_work.unexamined_candidates[0].candidate_id,2);
  assert.equal(report.candidate_hypotheses[0].accepted,false);
  assert.equal(report.candidate_hypotheses[1].accepted,true);
  assert.deepEqual(report.candidate_hypotheses[1].eye_to_enu_xyzw,recoveredOrientation,'auxiliary nadir seeds do not constrain final attitude');
  assert.equal(report.retrieval.evaluated_candidates,2);
 }
}
await assert.rejects(refineReferenceCandidates({}, {}, {}, [pose(0)], {}, '',()=>{},{firstCandidateId:2**32}),/ID range/);
console.info('Similarity fallback preserves evidence, earlier hypotheses, deferred alternatives, bounded work, and independent geometric decisions');

for(const mode of ['balanced','detailed'])assert.equal(matchingOptions(mode).similarityCandidates,16);
assert.equal(matchingOptions('fast').similarityCandidates,undefined);
