import assert from 'node:assert/strict';
import {refineSequenceGaps} from '../webapp/sequence-refinement.js';
import {LocalizationPipeline} from '../webapp/localization.js';
const hypothesis=id=>({accepted:true,candidate_id:id,map_manifest_sha256:'map',position_enu_m:[id,0,100],eye_to_enu_xyzw:[0,0,0,1]});
const frame=(sequence,candidates=[])=>({sequence,capture_time_ns:sequence*1e9,observation_sha256:`observation-${sequence}-7`,accepted:false,decision:candidates.length?'unresolved':'rejected',candidate_hypotheses:candidates,query:`frame-${sequence}.png`});
const pixels=()=>({gray:new Uint8Array([7,7,7,7]),width:2,height:2});
const frames=[frame(0),frame(1),frame(2,[hypothesis(0),hypothesis(1)]),frame(3)],initial=structuredClone(frames),calls=[];
const options={image:async index=>({...pixels(),time:index}),track:async(reference,image,sequence)=>{
 calls.push([reference.report.sequence,sequence]);return {...frame(sequence),decision:'relative_tracking',candidate_hypotheses:reference.report.candidate_hypotheses.map(h=>({...h,accepted:false,tracking_supported:true,tracking_anchor:h.tracking_anchor??{observation_sha256:reference.report.observation_sha256,candidate_id:h.candidate_id}}))};
}};
assert.equal(await refineSequenceGaps(frames,options),2);assert.deepEqual(calls,[[2,1],[1,0]]);
assert.deepEqual(frames[2],initial[2]);assert.deepEqual(frames[3],initial[3],'a trailing gap with no later anchor stays unresolved');
for(const i of [0,1]){assert.equal(frames[i].candidate_hypotheses.length,2);assert.equal(frames[i].accepted,false);assert.deepEqual(frames[i].sequence_refinement.previous_attempt,initial[i]);assert.equal(frames[i].candidate_hypotheses[0].tracking_anchor.observation_sha256,initial[2].observation_sha256)}
assert.equal(await refineSequenceGaps(frames,options),0);assert.equal(calls.length,2,'repeating the pass does not create another observation or add support');
const wrong=()=>[frame(0),frame(1,[hypothesis(0)])];
await assert.rejects(refineSequenceGaps(wrong(),{...options,track:async()=>({...frame(0,[hypothesis(0)]),observation_sha256:'other'})}),/identity/);
await assert.rejects(refineSequenceGaps(wrong(),{...options,track:async()=>({...frame(0,[hypothesis(0)]),accepted:true})}),/conditional/);
const duplicate=wrong();duplicate[0].observation_sha256=duplicate[1].observation_sha256;assert.equal(await refineSequenceGaps(duplicate,options),0);
const unknown=wrong();delete unknown[0].observation_sha256;assert.equal(await refineSequenceGaps(unknown,options),0);

// The adapter test checks evidence boundaries. It does not measure pose accuracy.
let stamp,active;const matched=[];
const pipeline=new LocalizationPipeline({matchImages:async(a,b)=>{matched.push([a.gray[0],b.gray[0]]);return {pairs:[{reference:[0,0],query:[0,0]}],backend_identity:'test-matcher'}}},{});
pipeline.pack={pack_id:'map',anchor_lat_lon:[0,0]};pipeline.camera={width:2,height:2};pipeline.references={elevation:()=>0};
pipeline.renderer={begin(p,_prior,sequence,capture_time_ns){stamp={sequence,capture_time_ns,observation_sha256:`observation-${sequence}-${p[0]}`}},select(){return JSON.stringify({...stamp,candidate_hypotheses:[]})},async render_reference(id){active=id;return new Uint8Array([7,7,7,7])},track(){return JSON.stringify({...hypothesis(active),accepted:false,tracking_supported:true})}};
const reference={report:frame(2,[hypothesis(0),hypothesis(1)]),image:pixels()},prior={latitude:0,longitude:0,radius_m:500,agl_m:100};
pipeline.temporal.previous={forward:'preserved'};const forward=pipeline.temporal.previous;
const result=await pipeline.trackFrom(reference,{...pixels(),time:1,timing:'decoded'},prior,1,()=>{});
assert.equal(result.decision,'relative_tracking');assert.equal(result.accepted,false);assert.equal(result.candidate_hypotheses.length,2);assert.equal(result.retrieval.searched_pairs,0);assert.equal(result.observation_sha256,frame(1).observation_sha256);assert.strictEqual(pipeline.temporal.previous,forward);assert.deepEqual(matched,[[7,7]]);
assert.equal(result.candidate_hypotheses[1].tracking_anchor.candidate_id,1);
await assert.rejects(pipeline.trackFrom({...reference,image:{...pixels(),gray:new Uint8Array([8,8,8,8])}},{...pixels(),time:1},prior,1,()=>{}),/pixels/);
const other=structuredClone(reference);other.report.candidate_hypotheses[0].map_manifest_sha256='different';await assert.rejects(pipeline.trackFrom(other,{...pixels(),time:1},prior,1,()=>{}),/map data/);
await assert.rejects(pipeline.trackFrom(reference,{...pixels(),width:1},prior,1,()=>{}),/calibration/);
console.info('Backward gap refinement preserves alternatives, evidence identity, map versions, and conditional status');

const {refineSequenceBackward}=await import('../webapp/sequence-refinement.js');
const {trackBranches,branchKey}=await import('../webapp/track-preview.js');
const {playbackPose}=await import('../webapp/pose-playback.js');
const conditional=(id,anchor)=>({...hypothesis(id),accepted:false,tracking_supported:true,tracking_anchor:anchor});
const anchor={observation_sha256:'older',candidate_id:0,map_manifest_sha256:'map'};
const series=[frame(0),frame(1,[conditional(0,anchor)]),frame(2,[hypothesis(0)]),frame(3,[conditional(0,anchor)]),frame(4,[hypothesis(0)])],baseline=structuredClone(series),checks=[];
const backward={image:async index=>({...pixels(),time:index}),intervalSeconds:2,checkMotion:async(reference,observation,_image,candidates)=>({observation_sha256:observation.observation_sha256,reference_observation_sha256:reference.report.observation_sha256,checks:candidates.map(h=>({candidate_id:h.candidate_id,reference_candidate_id:0,consistent:true,inliers:100}))}),track:async(reference,_image,sequence)=>{
 const h=reference.report.candidate_hypotheses[0];return {...frame(sequence),decision:'relative_tracking',candidate_hypotheses:[conditional(0,h.tracking_anchor??{observation_sha256:reference.report.observation_sha256,candidate_id:h.candidate_id,map_manifest_sha256:h.map_manifest_sha256})]};
},verify:async(observation,_image,candidates)=>{checks.push(observation.sequence);assert.ok(candidates.length);return {...observation,candidate_hypotheses:[{...hypothesis(0),position_enu_m:[20,0,100]}],decision:'unresolved'}}};
const counts=await refineSequenceBackward(series,backward);
assert.deepEqual(counts,{frames_updated:4,map_accepted:2,recovered:1});assert.deepEqual(checks,[2,0]);
assert.deepEqual(series[4],baseline[4]);
for(let i=0;i<4;i++){
 assert.equal(series[i].accepted,false,'local rechecks do not establish a unique geographic fix');
 assert.deepEqual(series[i].sequence_refinement.previous_attempt,baseline[i]);
 assert.equal(new Set(series[i].candidate_hypotheses.map(h=>h.candidate_id)).size,series[i].candidate_hypotheses.length,'forward and backward alternatives have distinct observation-local IDs');
 assert.deepEqual(series[i].candidate_hypotheses.slice(1),baseline[i].candidate_hypotheses,'forward alternatives are retained');
}
assert.equal(series[2].candidate_hypotheses[0].tracking_anchor.candidate_id,series[2].candidate_hypotheses[0].candidate_id);
const key=branchKey(series[1],series[1].candidate_hypotheses[0]),paths=trackBranches(series);
assert.ok(playbackPose(paths.get(key),1.5,{maxGap:1.1}),'a backward segment reaches its shared map anchor without a false interpolation break');
const firstPass=structuredClone(series);await refineSequenceBackward(series,backward);
assert.deepEqual(series,firstPass,'a repeated pass replaces its own result without accumulating duplicate alternatives or confidence');
const failed=[frame(0),frame(1,[hypothesis(0)])];const beforeFailed=structuredClone(failed);
await refineSequenceBackward(failed,{...backward,track:async()=>null,verify:async observation=>({...observation,candidate_hypotheses:[]})});assert.deepEqual(failed[0].sequence_refinement.previous_attempt,beforeFailed[0]);assert.deepEqual(failed[0].candidate_hypotheses,beforeFailed[0].candidate_hypotheses,'unsupported reverse checks retain forward evidence');assert.equal(failed[0].sequence_refinement.map_attempt.candidate_hypotheses.length,0);
console.info('Sequence map checks preserve forward alternatives, remap identities, avoid false anchor gaps, and remain idempotent');

pipeline.renderer.refine=id=>JSON.stringify({...hypothesis(id),accepted:true});
const oldSelect=pipeline.renderer.select;pipeline.renderer.select=()=>JSON.stringify({...JSON.parse(oldSelect()),candidate_hypotheses:[hypothesis(0)]});
const checked=await pipeline.refineAt(frame(1),{...pixels(),time:1},prior,[hypothesis(0)],()=>{});
assert.equal(checked.accepted,false);assert.equal(checked.decision,'unresolved');assert.equal(checked.candidate_hypotheses[0].accepted,true);assert.match(checked.evidence_correlation,/repeated checks add no independent confidence/);
await assert.rejects(pipeline.refineAt({...frame(1),observation_sha256:'different'},{...pixels(),time:1},prior,[hypothesis(0)],()=>{}),/pixels/);

const snapshot=structuredClone(reference.report),poses=reference.report.candidate_hypotheses;
pipeline.renderer.check_tracking_pose=(_pixels,stamp,pose)=>JSON.stringify({consistent:JSON.parse(pose).candidate_id===0,reference_id:JSON.parse(stamp).candidate_id});
const motion=await pipeline.checkMotion(reference,frame(1),{...pixels(),time:1},prior,poses,()=>{});
assert.equal(motion.checks.length,4);assert.deepEqual(motion.checks.map(c=>[c.reference_candidate_id,c.candidate_id,c.consistent]),[[0,0,true],[0,1,false],[1,0,true],[1,1,false]]);
assert.deepEqual(reference.report,snapshot,'motion checks never move or merge the candidate poses');
assert.match(motion.evidence_correlation,/not independent/);
await assert.rejects(pipeline.checkMotion(reference,frame(1),{...pixels(),gray:new Uint8Array([9,9,9,9]),time:1},prior,poses,()=>{}),/pixels/);
await assert.rejects(pipeline.checkMotion(reference,frame(1),{...pixels(),time:1},prior,[{...hypothesis(0),map_manifest_sha256:'other'}],()=>{}),/map data/);
console.info('Fixed-pose motion checks retain candidate identities, map versions, and uncertain correlation');

const {motionCompatibleCandidates}=await import('../webapp/sequence-refinement.js');
const geometricChecks=[{candidate_id:0,reference_candidate_id:2,consistent:true,inliers:1200},{candidate_id:1,reference_candidate_id:2,consistent:true,inliers:21},{candidate_id:2,reference_candidate_id:2,consistent:true,inliers:900},{candidate_id:3,reference_candidate_id:8,consistent:true,inliers:900}];
assert.deepEqual([...motionCompatibleCandidates(geometricChecks,1)],[1],'map support is compared with the same reference and image pairs, not another geographic alternative');
assert.throws(()=>motionCompatibleCandidates(geometricChecks,1,NaN),/retention/);
const conflict=[frame(0,[conditional(0,anchor)]),frame(1,[hypothesis(0)])],conflictOriginal=structuredClone(conflict);
const conflictOptions={...backward,intervalSeconds:.5,checkMotion:async(reference,observation)=>({observation_sha256:observation.observation_sha256,reference_observation_sha256:reference.report.observation_sha256,checks:[{candidate_id:0,reference_candidate_id:0,consistent:true,inliers:100},{candidate_id:1,reference_candidate_id:0,consistent:true,inliers:21}]})};
await refineSequenceBackward(conflict,conflictOptions);
assert.equal(conflict[0].candidate_hypotheses[0].tracking_supported,true,'a conflicting map fit cannot replace the relative continuation');
assert.equal(conflict[0].candidate_hypotheses[1].accepted,true,'map geometric acceptance remains distinct from temporal association');
assert.deepEqual(conflict[0].candidate_hypotheses[1].position_enu_m,[20,0,100],'the conflicting map alternative is retained without averaging');
assert.equal(conflict[0].accepted,false);assert.equal(conflict[0].decision,'unresolved');assert.deepEqual(conflict[0].sequence_refinement.previous_attempt,conflictOriginal[0]);
const once=structuredClone(conflict);await refineSequenceBackward(conflict,conflictOptions);assert.deepEqual(conflict,once);
const diagnosticOff=structuredClone(conflictOriginal);await refineSequenceBackward(diagnosticOff,{...conflictOptions,checkMotion:undefined});assert.equal(diagnosticOff[0].candidate_hypotheses[0].accepted,true,'the optional conditional motion policy is not enabled without its callback');
console.info('Conflicting map poses remain alternatives without replacing a supported motion branch or adding confidence');
