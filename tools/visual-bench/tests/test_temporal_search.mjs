import assert from 'node:assert/strict';
import {TemporalSearch,poseConverged,refineCandidates} from '../webapp/temporal-search.js';
const pose=(x,q=[0,0,0,1])=>({position_enu_m:[x,0,110],eye_to_enu_xyzw:q});
const history=new TemporalSearch();
const first={observation_sha256:'first',candidate_hypotheses:[{candidate_id:2,accepted:true,...pose(0)},{candidate_id:9,accepted:true,...pose(80)},{candidate_id:10,accepted:false,...pose(1)}]};
history.remember(first);first.candidate_hypotheses[0].position_enu_m[0]=999;
assert.deepEqual(history.seeds('second',[0,0,110],100).map(x=>x.position_enu_m[0]),[0,80],'separate alternatives survive without averaging');
assert.deepEqual(history.seeds('first',[0,0,110],100),[],'reprocessing does not seed itself');
assert.deepEqual(history.seeds('second',[1000,0,110],100),[],'out-of-prior seeds do not replace an area search');
const seeds=history.seeds('second',[0,0,110],100);
const tracked=history.label({observation_sha256:'second',accepted:true,decision:'unique_among_evaluated',candidate_hypotheses:[{candidate_id:0,accepted:true,...pose(2)}]},seeds);
assert.equal(tracked.accepted,false);assert.equal(tracked.decision,'unresolved');assert.equal(tracked.search_seeds.observation_sha256,'first');assert.deepEqual(tracked.search_seeds.candidate_ids,[2,9]);
assert.match(tracked.evidence_correlation,/unknown/);
assert.equal(poseConverged(pose(0),pose(.01,[0,0,0,-1])),true);
assert.equal(poseConverged(pose(0),pose(.2)),false);
assert.equal(poseConverged(pose(0),pose(0,[0,0,Math.sin(.1),Math.cos(.1)])),false);
const rendered=[],matched=[],reports=new Map();
const renderer={
 render_reference:async(id,json)=>{rendered.push({id,pose:JSON.parse(json)});return new Uint8Array([1])},
 refine:(id,_pairs,backend)=>{assert.equal(backend,'replacement-matcher');const latest=id===0?{candidate_id:id,accepted:true,...pose(.01)}:{candidate_id:id,accepted:false,reason:'current image has insufficient geometry'};reports.set(id,latest);return JSON.stringify(latest)},
 select:()=>JSON.stringify({candidate_hypotheses:[...reports.values()]})
};
const matcher={matchImages:async(_reference,query,keys)=>{matched.push({query,keys});return {pairs:[],backend_identity:'replacement-matcher'}}};
const query={gray:new Uint8Array([7])};
const result=await refineCandidates(renderer,matcher,{width:1,height:1},seeds,query,'second',3,()=>{});
assert.equal(rendered.length,2,'converged and rejected candidates each stop without extra fits');
assert.deepEqual(result.candidate_hypotheses.map(h=>h.accepted),[true,false]);
assert.ok(matched.every(m=>m.query===query&&m.keys.query==='second/query'),'every seed is checked using the current observation');
history.remember({observation_sha256:'failed',candidate_hypotheses:[{accepted:false}]});assert.deepEqual(history.seeds('next',[0,0,110],100),[],'a failed observation clears stale tracking seeds');
console.info('Temporal alternatives, seed provenance, current-image verification and convergence passed');
let evaluations=0;const attempts=[];
const refining={render_reference:async(id,json)=>{attempts.push(JSON.parse(json));return new Uint8Array([1])},refine:()=>JSON.stringify(++evaluations===1?{accepted:false,refinement_proposal:pose(3)}:{accepted:true,...pose(3.01)}),select:()=>JSON.stringify({candidate_hypotheses:[{accepted:evaluations===2}]})};
const recovered=await refineCandidates(refining,matcher,{width:1,height:1},[pose(0)],query,'same-observation',3,()=>{});
assert.equal(evaluations,2);assert.deepEqual(attempts.map(p=>p.position_enu_m[0]),[0,3]);assert.equal(recovered.candidate_hypotheses[0].accepted,true);
evaluations=0;const stillRejected=await refineCandidates(refining,matcher,{width:1,height:1},[pose(0)],query,'same-observation',1,()=>{});assert.equal(stillRejected.candidate_hypotheses[0].accepted,false,'a pending refinement proposal is never accepted by itself');
console.info('Rejected surface seeds require a new render and a new full acceptance check');
const tracking=new TemporalSearch(),pixels={gray:new Uint8Array([1,2,3,4]),width:2,height:2};
tracking.remember({...first,sequence:4,capture_time_ns:100},pixels);
pixels.gray.fill(0);
const tracks=[];
const tracker={render_reference:async(id,json)=>{tracks.push({id,pose:JSON.parse(json)});return new Uint8Array(4)},
 track:(previous,json,_pairs,backend)=>{assert.equal(previous[0],1,'retain the previous camera pixels');assert.equal(backend,'replacement-matcher');const metadata=JSON.parse(json);assert.equal(metadata.sequence,4);assert.equal(metadata.capture_time_ns,100);return JSON.stringify({candidate_id:metadata.candidate_id,accepted:false,tracking_supported:true,...pose(metadata.candidate_id*80+3)})},
 select:()=>JSON.stringify({observation_sha256:'next',sequence:5,capture_time_ns:200,accepted:false,decision:'rejected'})};
const allSeeds=tracking.seeds('next',[500,0,110],1000);
const relative=await tracking.track(tracker,matcher,{width:2,height:2},allSeeds,query,'next',()=>{});
assert.equal(relative.accepted,false);assert.equal(relative.decision,'relative_tracking');
assert.equal(relative.candidate_hypotheses.length,2,'do not collapse initial map alternatives');
assert.ok(relative.candidate_hypotheses.every(h=>!h.accepted&&h.tracking_supported&&!h.geometry_covariance));
assert.deepEqual(relative.candidate_hypotheses.map(h=>h.tracking_anchor.candidate_id),[2,9]);
tracking.remember(relative,pixels);assert.equal(tracking.previous.steps,1);
assert.equal(tracking.previous.candidates[0].tracking_anchor.observation_sha256,'first','relative tracking retains its original map anchor');
tracker.track=()=>JSON.stringify({accepted:false,tracking_supported:false});
assert.equal(await tracking.track(tracker,matcher,{width:2,height:2},allSeeds,query,'failed',()=>{}),null,'failed relative geometry requires map verification or a new search');
console.info('Relative tracking retains alternatives and anchor identity without producing independent map acceptance');
let variants=0;
const alternateMatcher={...matcher,async *matchAlternatives(){variants++;yield {pairs:[],backend_identity:'rotated-camera-adapter'}}};
tracker.track=(_pixels,metadata,_pairs,backend)=>JSON.stringify({candidate_id:JSON.parse(metadata).candidate_id,accepted:false,tracking_supported:backend==='rotated-camera-adapter',...pose(3)});
const turned=await tracking.track(tracker,alternateMatcher,{width:2,height:2},allSeeds,query,'turned',()=>{});
assert.equal(variants,1);assert.equal(turned.decision,'relative_tracking');assert.equal(turned.accepted,false);
assert.ok(turned.candidate_hypotheses.every(h=>h.tracking_supported));
assert.equal(turned.candidate_hypotheses[0].tracking_anchor.observation_sha256,'first');
console.info('Adapter alternatives require a fresh geometric check and retain the same anchor');
const progressEvents=[];
const loadingMatcher={matchImages:async(_reference,_query,keys)=>{keys.progress('Loading model for this observation');return {pairs:[],backend_identity:'replacement-matcher'}}};
await refineCandidates(renderer,loadingMatcher,{width:1,height:1},[pose(0)],query,'new-request',1,text=>progressEvents.push(text));
assert.ok(progressEvents.includes('Loading model for this observation'),'lazy model progress belongs to the active observation, not the completed initialization request');
