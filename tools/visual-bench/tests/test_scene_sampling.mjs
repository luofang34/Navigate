import assert from 'node:assert/strict';
import {sceneSamplingPlans,refineSceneSampling} from '../webapp/scene-sampling.js';
const sources=Array.from({length:6},(_,sequence)=>({sequence,observation_sha256:'frame-'+sequence,capture_time_ns:sequence*2e8,requested_time_s:sequence*.2,candidate_hypotheses:[]}));
const candidate=items=>({cameras:items.map(s=>({observation_sha256:s.observation_sha256}))});
const group={source_group_sha256:'split',observations:sources,candidates:[candidate(sources.slice(0,3)),candidate(sources.slice(3))]};
const reconstruction={groups:[group],geographic_acceptance:false};
const plan=sceneSamplingPlans(reconstruction);
assert.deepEqual(plan.plans[0].times,[.25,.3,.35,.45,.5,.55,.65,.7,.75]);
assert.deepEqual(sceneSamplingPlans({groups:[{...group,candidates:[candidate(sources)]}]}).plans,[],'connected candidates need no extra inference');
assert.deepEqual(sceneSamplingPlans({groups:[{...group,candidates:[]}]}).plans,[],'no evidence is not a fabricated transition');
assert.deepEqual(sceneSamplingPlans(reconstruction,{maxExtraFrames:8}).deferred_group_sha256,['split'],'a partial sampling budget leaves an explicit deferral');
assert.equal(sceneSamplingPlans({groups:[group,group,group]}).plans.length,2);
assert.throws(()=>sceneSamplingPlans(reconstruction,{subdivisions:50}),/budget/);
const observed=[],saved=[],commits=[],messages=[],frames=structuredClone(sources);let decodeCount=0;
const pipeline={
 beginSequence:async options=>assert.equal(options.maxFrames,129),
 observeScene:async(image,prior,sequence,expected,progress)=>{progress('Following image features…');const f={sequence,capture_time_ns:Math.round(image.time*1e9),observation_sha256:expected??'new-'+sequence,requested_time_s:image.requested_time_s,candidate_hypotheses:[],accepted:false};observed.push(f);return f},
 finishSequence:async()=>[{sha256:'denser'}],
 refineScenes:async(records,original)=>{assert.equal(records[0].sha256,'denser');assert.equal(original,reconstruction);assert.equal(commits.length,1,'new source frames persist before reconstruction');return {groups:[{source_group_sha256:'denser',observations:observed,candidates:[candidate(observed)]}],geographic_acceptance:false}}
};
const image=async f=>({time:f.capture_time_ns/1e9,requested_time_s:f.requested_time_s});
const options={pipeline,frames,image,progress:text=>messages.push(text),decode:async time=>{decodeCount++;return {time:time===.25?.2:time,requested_time_s:time,blob:new Blob([String(time)])}},
 save:async samples=>saved.push(...samples),commit:async()=>commits.push(frames.map(f=>f.observation_sha256))};
const result=await refineSceneSampling(reconstruction,options);
assert.equal(decodeCount,9);assert.equal(saved.length,8,'two seek requests for the same decoded frame are one observation');
assert.ok(messages.some(text=>text.includes('sample 15/15 · Following image features')),'matcher progress retains the current frame and total');
assert.equal(observed.length,14);assert.equal(frames.length,14);assert.equal(new Set(frames.map(f=>f.sequence)).size,14);
assert.deepEqual(observed.filter(f=>f.observation_sha256.startsWith('frame')).map(f=>f.sequence),sources.map(f=>f.sequence));
assert.equal(result.groups.length,2);assert.equal(result.groups[0],group,'the original alternatives remain unchanged');
assert.equal(result.geographic_acceptance,false);assert.match(result.sampling_refinement.evidence_correlation,/unknown/);
assert.equal(result.sampling_refinement.attempts[0].skipped.length,1);
assert.ok(frames.every((f,i)=>i===0||f.capture_time_ns>frames[i-1].capture_time_ns));
assert.ok(saved.every(({frame})=>frame.accepted===false));
await assert.rejects(refineSceneSampling(reconstruction,{...options,frames:structuredClone(sources),pipeline:{...pipeline,observeScene:async()=>({observation_sha256:'changed'})}}),/existing observation/);
const failureFrames=structuredClone(sources);
await assert.rejects(refineSceneSampling(reconstruction,{...options,frames:failureFrames,save:async()=>{throw Error('Disk full')}}),/Disk full/);
assert.deepEqual(failureFrames,sources,'storage failure does not publish observations without saved pixels');
console.info('Bounded resampling preserves alternatives and identities, skips duplicate decoded frames, and commits pixels before reconstruction.');

const repeated=await refineSceneSampling(result,{...options,pipeline:{beginSequence(){throw Error('Repeated inference')}}});
assert.deepEqual(repeated.groups,result.groups);assert.deepEqual(repeated.sampling_refinement.attempts,result.sampling_refinement.attempts,'reprocessing cannot manufacture new observations or confidence');

const boundarySources=Array.from({length:176},(_,sequence)=>({sequence,observation_sha256:'boundary-'+sequence,capture_time_ns:sequence*2e8,requested_time_s:sequence*.2,candidate_hypotheses:[]}));
const completeGroup=(name,start,end,parent)=>({source_group_sha256:name,observations:boundarySources.slice(start,end),candidates:[{...candidate(boundarySources.slice(start,end)),parent_scene_sha256:parent}]});
const boundaryGroups=[completeGroup('before',0,64,null),completeGroup('separate',56,120,null),completeGroup('after',112,176,'separate')];
const boundaryReconstruction={groups:boundaryGroups,geographic_acceptance:false},boundaryBefore=JSON.stringify(boundaryReconstruction);
const bridge=sceneSamplingPlans(boundaryReconstruction);
assert.equal(bridge.plans.length,1,'complete groups with a failed coordinate link still receive a bounded retry');
assert.equal(bridge.plans[0].scope,'overlapping_group_retry');
assert.equal(bridge.plans[0].observations.length,129);
assert.deepEqual(bridge.plans[0].observations,boundarySources.slice(47));
assert.deepEqual(bridge.plans[0].times,[],'reuse source observations instead of fabricating new evidence');
assert.equal(JSON.stringify(boundaryReconstruction),boundaryBefore);
assert.equal(sceneSamplingPlans({groups:[boundaryGroups[0],{...boundaryGroups[1],candidates:[{...boundaryGroups[1].candidates[0],parent_scene_sha256:'before'}]},boundaryGroups[2]]}).plans.length,0,'an existing parent link requires no overlap retry');
assert.equal(sceneSamplingPlans({groups:[{...boundaryGroups[0],candidates:[]},...boundaryGroups.slice(1)]}).plans.length,0,'absent predecessor geometry does not create a shared-camera constraint');
const observedBoundary=[];
const boundaryResult=await refineSceneSampling(boundaryReconstruction,{frames:structuredClone(boundarySources),image,decode:()=>{throw Error('Existing overlap requires no new video samples')},save:async samples=>assert.equal(samples.length,0),pipeline:{
 beginSequence:async()=>{},observeScene:async(pixels,prior,sequence,expected)=>{const f={...boundarySources[sequence],observation_sha256:expected};observedBoundary.push(f);return f},
 finishSequence:async()=>[{sha256:'bridge'}],refineScenes:async()=>({groups:[{source_group_sha256:'bridge',observations:observedBoundary,candidates:[]}],geographic_acceptance:false})}});
assert.equal(observedBoundary.length,129);
assert.equal(boundaryResult.groups[0],boundaryGroups[0]);
assert.equal(boundaryResult.groups.at(-1).candidates.length,0,'scheduling a retry does not force an unsupported scene alignment');
assert.equal(sceneSamplingPlans(boundaryResult).plans.length,0,'a failed bounded retry cannot consume repeated work or add confidence');
assert.equal(boundaryResult.geographic_acceptance,false);
console.info('A disconnected full scene group retries shared source images within a 129-observation window; failed geometry remains unresolved.');
