import assert from 'node:assert/strict';
import {recoverRejectedCandidates} from '../webapp/candidate-refinement.js';
import {matchingOptions} from '../webapp/matching-options.js';
import {LocalizationPipeline} from '../webapp/localization.js';
const pose=x=>({position_enu_m:[x,0,100],eye_to_enu_xyzw:[0,0,0,1]});
const image={width:2,height:2,gray:new Uint8Array(4)};
function fixture(){
 const initial=[
  {candidate_id:30,accepted:false,refinement_proposal:{...pose(30),spatial_support:30}},
  {candidate_id:9,accepted:false,refinement_proposal:{...pose(9),spatial_support:25}},
  {candidate_id:8,accepted:false,refinement_proposal:{...pose(8),spatial_support:25}},
  {candidate_id:20,accepted:false,refinement_proposal:{...pose(20),spatial_support:8}},
  {candidate_id:12,accepted:false,reason:'Reference depth is unavailable'},
 ];
 const results=new Map(initial.map(h=>[h.candidate_id,h])),calls=[],counts=new Map();let active;
 const renderer={
  async render_reference(id,value){active={id,pose:JSON.parse(value)};return image.gray},
  refine(id,_pairs,backend){
   assert.equal(id,active.id);const count=(counts.get(id)??0)+1;counts.set(id,count);
   const success=id===8||(id===30&&count===2),next=pose(active.pose.position_enu_m[0]+1);
   const report={candidate_id:id,backend,accepted:success,...(success?active.pose:{refinement_proposal:next}),map_manifest_sha256:'same-map'};results.set(id,report);return JSON.stringify(report);
  },
  select(){const values=[...results.values()];return JSON.stringify({observation_sha256:'same-observation',accepted:false,decision:values.filter(h=>h.accepted).length>1?'unresolved':'rejected',candidate_hypotheses:values})},
 };
 const matcher={async matchImages(_r,q,keys){assert.equal(q,image);calls.push({id:active.id,...keys});return {pairs:[],backend_identity:'replaceable-recovery-adapter'}}};
 return {renderer,matcher,results,calls};
}
const f=fixture();
const report=await recoverRejectedCandidates(f.renderer,f.matcher,image,image,'same-observation',()=>{},{limit:3,passes:2});
assert.deepEqual(report.recovery_work.candidate_ids,[30,8,9],'rank geometric support and break ties by stable candidate ID');
assert.deepEqual(f.calls.map(c=>c.id),[30,30,8,9,9]);
assert.equal(report.recovery_work.attempts,5);
assert.ok(f.calls.every(c=>c.recovery===true&&c.stage==='refinement'&&c.query==='same-observation/query'));
assert.equal(report.observation_sha256,'same-observation');
assert.equal(report.decision,'unresolved');assert.equal(report.accepted,false);
assert.deepEqual(report.candidate_hypotheses.filter(h=>h.accepted).map(h=>h.candidate_id),[30,8]);
assert.equal(report.candidate_hypotheses.find(h=>h.candidate_id===12).reason,'Reference depth is unavailable');
assert.equal(report.candidate_hypotheses.find(h=>h.candidate_id===20).refinement_proposal.spatial_support,8,'unselected alternatives remain visible');
assert.match(report.recovery_work.scope,/does not add independent confidence/);
for(const options of [{limit:0},{limit:2,passes:1}]){
 const next=fixture();await recoverRejectedCandidates(next.renderer,next.matcher,image,image,'bounded',()=>{},options);
 assert.ok(next.calls.length<=options.limit*(options.passes??2),'caller work budget bounds adapter calls');
}
const accepted=fixture();accepted.results.set(30,{candidate_id:30,accepted:true,...pose(30)});
await recoverRejectedCandidates(accepted.renderer,accepted.matcher,image,image,'accepted',()=>{},{limit:4});
assert.equal(accepted.calls.length,0,'accepted normal search does not pay recovery cost');
const unsupported=fixture();unsupported.results.clear();unsupported.results.set(12,{candidate_id:12,accepted:false,reason:'Reference depth is unavailable'});
await recoverRejectedCandidates(unsupported.renderer,unsupported.matcher,image,image,'unsupported',()=>{},{limit:4});assert.equal(unsupported.calls.length,0);
for(const options of [{limit:-1},{limit:17},{limit:NaN},{limit:1.5},{passes:0},{passes:4}])await assert.rejects(recoverRejectedCandidates(f.renderer,f.matcher,image,image,'invalid',()=>{},options),/budget/);
const replacement=fixture(),pipeline=new LocalizationPipeline(replacement.matcher,{recoveryCandidates:3});pipeline.renderer=replacement.renderer;pipeline.camera=image;
const composed=await pipeline.recoverCandidates({verification_work:{attempts:128},retrieval:{stage:'retrieval_only'}},image,'same-observation',()=>{});
assert.equal(composed.verification_work.attempts,128,'existing retrieval verification diagnostics survive recovery');assert.equal(composed.retrieval.stage,'retrieval_only');assert.equal(composed.recovery_work.attempts,5);
for(const profile of ['balanced','detailed'])assert.equal(matchingOptions(profile).recoveryCandidates,4);
assert.equal(matchingOptions('fast').recoveryCandidates,undefined);
console.info('Bounded recovery preserves alternatives and evidence, skips unsupported data and accepted results, and rechecks geometry through a replaceable matcher');
