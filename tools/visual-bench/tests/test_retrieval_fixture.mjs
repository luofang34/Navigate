import assert from 'node:assert/strict';
import {retrievalFixture,refineRetrievalFixture} from '../webapp/qa-retrieval-fixture.js';
const prior={latitude:40,longitude:-74,radius_m:500,agl_m:110},sha='a'.repeat(64),pack={pack_id:'b'.repeat(64)};
const pose={position_enu_m:[1,2,110],eye_to_enu_xyzw:[.6,0,0,.8],accepted:true,covariance:[0]};
const fixture={schema:2,map_manifest_sha256:pack.pack_id,prior,source:'test retriever',cases:{[sha]:{candidates:[pose,{...pose,position_enu_m:[100,200,110]}],reference_ids:[1,2]}}};
const selected=retrievalFixture(fixture,sha,pack,prior);
assert.equal(selected.candidates.length,2,'geographic alternatives stay separate');
assert.deepEqual(selected.candidates[0].eye_to_enu_xyzw,pose.eye_to_enu_xyzw,'arbitrary orientation survives');
assert.equal(Object.hasOwn(selected.candidates[0],'accepted'),false,'retrieval cannot supply acceptance');
assert.equal(Object.hasOwn(selected.candidates[0],'covariance'),false,'retrieval cannot supply pose confidence');
assert.throws(()=>retrievalFixture(fixture,'c'.repeat(64),pack,prior),/candidate set/);
assert.throws(()=>retrievalFixture(fixture,sha,{pack_id:'d'.repeat(64)},prior),/map identity/);
assert.throws(()=>retrievalFixture(fixture,sha,pack,{...prior,radius_m:600}),/prior/);
fixture.cases[sha].candidates[0]={...pose,eye_to_enu_xyzw:[0,0,0,2]};
assert.throws(()=>retrievalFixture(fixture,sha,pack,prior),/pose/);
console.info('Retrieval replay binds source and map identities, preserves alternatives and carries no acceptance');

const reports=new Map(),calls=new Map(),contexts=[];let active;
const seeds={candidates:Array.from({length:16},(_,id)=>({position_enu_m:[id,0,100],eye_to_enu_xyzw:[0,0,0,1]}))};
const renderer={
 async render_reference(id,value){active={id,pose:JSON.parse(value)};calls.set(id,(calls.get(id)??0)+1);return new Uint8Array(4)},
 refine(id){assert.equal(id,active.id);const next={position_enu_m:[active.pose.position_enu_m[0]+.2,0,100],eye_to_enu_xyzw:[0,0,0,1],spatial_support:id};const report={candidate_id:id,accepted:id>=14,...(id>=14?next:{refinement_proposal:next})};reports.set(id,report);return JSON.stringify(report)},
 select:()=>JSON.stringify({candidate_hypotheses:[...reports.values()]})
};
const matcher={async matchImages(_reference,_query,keys){contexts.push(keys);return {pairs:[],backend_identity:'test adapter'}}};
const result=await refineRetrievalFixture({renderer,matcher,camera:{width:2,height:2}},seeds,{gray:new Uint8Array(4)},'same observation',()=>{});
assert.equal(result.candidate_hypotheses.length,16,'bounded refinement retains other geographic alternatives');
assert.equal(result.candidate_hypotheses.filter(h=>h.accepted).length,2);
for(let id=0;id<16;id++)assert.equal(calls.get(id),id<4?1:3,'only twelve geometric proposals receive extra rendering');
assert.ok(contexts.every(k=>k.query==='same observation/query'&&!k.recovery),'refinement shares evidence and does not enlarge matcher budgets');

assert.equal(result.verification_work.attempts,contexts.length,'timing counts include the work that was actually executed');
assert.equal(result.verification_work.attempts,40);
for(const field of ['render_ms','matching_ms','geometry_ms'])assert.ok(Number.isFinite(result.verification_work[field])&&result.verification_work[field]>=0);
