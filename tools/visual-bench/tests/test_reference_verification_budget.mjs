import assert from 'node:assert/strict';
import {refineReferenceCandidates} from '../webapp/candidate-refinement.js';
import {LocalizationPipeline} from '../webapp/localization.js';
import {TemporalSearch} from '../webapp/temporal-search.js';

const pose=x=>({position_enu_m:[x,0,100],eye_to_enu_xyzw:[0,0,0,1]});
const candidates=Array.from({length:6},(_,id)=>pose(id));
const image={width:2,height:2,gray:new Uint8Array(4),time:0,timing:'still image'};
function fixture(){
 const calls=[],results=new Map(),counts=new Map();let active,stamp={observation_sha256:'observation'};
 const renderer={
  begin(_pixels,_prior,sequence,capture_time_ns){results.clear();stamp={...stamp,sequence,capture_time_ns}},
  async render_reference(id,value){active={id,pose:JSON.parse(value)};return image.gray},
  refine(id,pairs,backend){
   assert.equal(id,active.id);assert.deepEqual(JSON.parse(pairs),[]);assert.equal(backend,'replacement-adapter');
   const count=(counts.get(id)??0)+1;counts.set(id,count);
   const report={candidate_id:id,accepted:count===3,...(count===3?active.pose:{refinement_proposal:{...pose(active.pose.position_enu_m[0]+1),spatial_support:id+1}})};
   results.set(id,report);return JSON.stringify(report);
  },
  select(){return JSON.stringify({...stamp,accepted:false,decision:'unresolved',candidate_hypotheses:[...results.values()]})},
 };
 const matcher={async matchImages(_reference,query,keys){assert.equal(query,image);calls.push({id:active.id,...keys});return {pairs:[],backend_identity:'replacement-adapter'}}};
 return {renderer,matcher,calls};
}
const f=fixture(),original=structuredClone(candidates);
const limited=await refineReferenceCandidates(f.renderer,f.matcher,image,candidates,image,'observation',()=>{},{candidateLimit:2,refineLimit:1});
assert.deepEqual(f.calls.map(c=>c.id),[0,1,1,1],'candidate and refinement budgets independently bound adapter calls');
assert.ok(f.calls.every(c=>c.recovery===false&&c.stage==='refinement'));
assert.deepEqual(limited.verification_work.evaluated_candidate_ids,[0,1]);
assert.equal(limited.verification_work.attempts,4);
assert.deepEqual(limited.verification_work.unexamined_candidates,candidates.slice(2).map((pose,index)=>({candidate_id:index+2,pose})));
assert.deepEqual(candidates,original,'search input and skipped poses stay intact');
assert.equal(limited.accepted,false);assert.equal(limited.decision,'unresolved','the scheduler preserves the geometry selection decision');
assert.equal(limited.candidate_hypotheses.find(h=>h.candidate_id===1).accepted,true);
for(const options of [{candidateLimit:2,passes:1},{candidateLimit:2,refineLimit:0},{passes:1}]){
 const next=fixture(),result=await refineReferenceCandidates(next.renderer,next.matcher,image,candidates,image,'observation',()=>{},options);
 assert.equal(next.calls.length,options.candidateLimit??6);
 assert.equal(result.verification_work.unexamined_candidates.length,6-(options.candidateLimit??6));
}
const empty=fixture(),emptyResult=await refineReferenceCandidates(empty.renderer,empty.matcher,image,[],image,'observation',()=>{});
assert.equal(empty.calls.length,0);assert.deepEqual(emptyResult.verification_work.evaluated_candidate_ids,[]);
for(const options of [{candidateLimit:0},{candidateLimit:-1},{candidateLimit:129},{candidateLimit:NaN},{candidateLimit:1.5},{refineLimit:-1},{refineLimit:129},{passes:0},{passes:4}]){
 const invalid=fixture();await assert.rejects(refineReferenceCandidates(invalid.renderer,invalid.matcher,image,candidates,image,'observation',()=>{},options),/budget/);assert.equal(invalid.calls.length,0);
}
const oversized=fixture();await assert.rejects(refineReferenceCandidates(oversized.renderer,oversized.matcher,image,Array(129).fill(pose(0)),image,'observation',()=>{}),/budget/);assert.equal(oversized.calls.length,0);

const integration=fixture(),search={async propose(){return {candidates,scope:'bounded retrieval; alternatives remain',backend_identity:'retrieval-adapter',reference_ids:[1,2,3],unsupported_references:[9]}}};
const pipeline=new LocalizationPipeline(integration.matcher,{referenceCandidates:2,temporal:false},search);
pipeline.renderer=integration.renderer;pipeline.camera=image;pipeline.pack={anchor_lat_lon:[0,0]};pipeline.references={elevation:()=>0};
const result=await pipeline.estimate(image,{latitude:0,longitude:0,agl_m:100,radius_m:500},0,()=>{});
assert.equal(result.retrieval.pose_candidates,6);assert.equal(result.retrieval.evaluated_candidates,2);
assert.deepEqual(result.retrieval.unexamined_candidate_ids,[2,3,4,5]);
assert.deepEqual(result.retrieval.unsupported_references,[9]);assert.equal(result.retrieval.stage,'retrieval_only');
assert.equal(result.verification_work.unexamined_candidates.length,4);
assert.ok(integration.calls.every(c=>c.id<2),'pipeline passes the caller budget through to verification');
console.info('Reference verification caps work, preserves IDs and skipped poses, and reports incomplete search without changing geometric decisions');

const rejected=fixture(),checked=await refineReferenceCandidates(rejected.renderer,rejected.matcher,image,candidates,image,'observation',()=>{},{candidateLimit:2,passes:1});
const relative={decision:'relative_tracking',candidate_hypotheses:[{candidate_id:7,tracking_supported:true,...pose(7)}]};
const fallback=new TemporalSearch().reacquired(checked,relative,null);
assert.equal(fallback.decision,'relative_tracking');
assert.deepEqual(fallback.regional_search.verification_work.unexamined_candidates,checked.verification_work.unexamined_candidates,'continuing relative tracking retains the incomplete geographic search');
assert.deepEqual(fallback.candidate_hypotheses,relative.candidate_hypotheses,'a limited rejected map search does not erase supported relative motion');
