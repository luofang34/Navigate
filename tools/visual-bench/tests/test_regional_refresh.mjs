import assert from 'node:assert/strict';
import {LocalizationPipeline} from '../webapp/localization.js';

// The workflow test replaces image and geometry adapters. It does not test pose accuracy.
globalThis.OffscreenCanvas=class {
 constructor(width,height){Object.assign(this,{width,height})}
 getContext(){return {drawImage(){},translate(){},rotate(){},getImageData:()=>({data:new Uint8Array(this.width*this.height*4)})}}
};
class ImageTracks {constructor(){this.sources=[]}push(id){this.sources.push(id)}free(){this.freed=true}}
const pose=x=>({position_enu_m:[x,0,100],eye_to_enu_xyzw:[0,0,0,1]});
let searches=0,stamp,refinement;
const matcher={matchImages:async()=>({pairs:[],backend_identity:'test-adapter'}),retrievePairs:async()=>{searches++;return []}};
const pipeline=new LocalizationPipeline(matcher,{headings:4,refinements:1,regionalIntervalSeconds:5});
pipeline.SceneTracks=ImageTracks;
pipeline.pack={anchor_lat_lon:[0,0]};pipeline.camera={width:2,height:2};pipeline.references={elevation:()=>0,crops:()=>[]};
pipeline.renderer={
 begin(_pixels,_prior,sequence,capture_time_ns){stamp={observation_sha256:`observation-${sequence}`,sequence,capture_time_ns};refinement=null},
 select(){return JSON.stringify({...stamp,accepted:true,decision:'unique_among_evaluated',candidate_hypotheses:[refinement??{accepted:true,candidate_id:0,map_manifest_sha256:'map',...pose(searches*80)}]})},
 async render_reference(){return new Uint8Array(4)},
 track(){return JSON.stringify({accepted:false,tracking_supported:true,candidate_id:0,...pose(2)})},
 refine(id){refinement={accepted:true,candidate_id:id,map_manifest_sha256:'map',...pose(3)};return JSON.stringify(refinement)},
};
const prior={latitude:0,longitude:0,radius_m:500,agl_m:100};
const estimate=(time,sequence)=>pipeline.estimate({gray:new Uint8Array(4),width:2,height:2,canvas:{},time},prior,sequence,()=>{});
const initial=await estimate(0,0);assert.equal(searches,1);assert.equal(initial.accepted,true);
const relative=await estimate(1,1);assert.equal(searches,1);assert.equal(relative.decision,'relative_tracking');
pipeline.temporal.previous.steps=4;
const local=await estimate(4.8,2);assert.equal(searches,1);assert.equal(local.decision,'relative_tracking','raising the video sample rate does not multiply scheduled map checks');
const refreshed=await estimate(5,3);assert.equal(searches,2,'the live pipeline must execute a new area search while relative tracking remains supported');
assert.deepEqual(refreshed.candidate_hypotheses[0].position_enu_m,[160,0,100]);
assert.equal(refreshed.candidate_hypotheses[0].continuity_break,true);
assert.equal(refreshed.candidate_hypotheses[0].track_id,relative.candidate_hypotheses[0].track_id);
assert.deepEqual(refreshed.regional_search.relative_alternatives[0].position_enu_m,[2,0,100]);
assert.equal(refreshed.retrieval.algorithm,'descriptor_shortlist_then_regional_planar_proposals');
console.info('The localization workflow performs timed area searches and preserves a separate relative alternative');

const waiting=new LocalizationPipeline(matcher,{headings:4,regionalIntervalSeconds:5});
waiting.SceneTracks=ImageTracks;
Object.assign(waiting,{pack:pipeline.pack,camera:pipeline.camera,references:pipeline.references});let waitingStamp;
waiting.renderer={begin(_pixels,_prior,sequence,capture_time_ns){waitingStamp={sequence,capture_time_ns,observation_sha256:'waiting-'+sequence}},select:()=>JSON.stringify({...waitingStamp,accepted:false,decision:'rejected',candidate_hypotheses:[]})};
const check=(time,sequence,timing='browser decoded frame presentation timestamp')=>waiting.estimate({gray:new Uint8Array(4),width:2,height:2,canvas:{},time,timing},prior,sequence,()=>{});
const before=searches;await check(0,0);assert.equal(searches,before+1);
const deferred=await check(.2,1);assert.equal(searches,before+1,'unsupported adjacent video frames do not each repeat the full geographic search');
assert.equal(deferred.decision,'search_deferred');assert.equal(deferred.accepted,false);assert.equal(deferred.retrieval.stage,'not_run');assert.equal(deferred.candidate_hypotheses.length,0);assert.equal(deferred.observation_sha256,'waiting-1');
assert.equal(waiting.temporal.previous,null,'a deferred frame does not preserve stale pose seeds');
await check(5,2);assert.equal(searches,before+2,'search resumes at the configured capture-time interval');
await check(0,3,'still image');await check(0,4,'still image');assert.equal(searches,before+4,'independent still images with unknown times each receive their own geographic search');
const {resultSummary}=await import('../webapp/result-summary.js');const summary=resultSummary(deferred);assert.equal(summary.title,'No pose for this frame');assert.match(summary.explanation,/search was not run/);assert.deepEqual(summary.metrics,[]);
console.info('Acquisition cadence bounds repeated failed video searches without inventing poses or suppressing independent still-image searches');

const retainedMatcher=waiting.matcher;waiting.beginSequence();assert.strictEqual(waiting.matcher,retainedMatcher,'new capture streams keep the loaded matcher');assert.equal(waiting.temporal.previous,null);assert.equal(waiting.temporal.lastRegionalSearchNs,null);const beforeRestart=searches;await check(0,0);assert.equal(searches,beforeRestart+1,'a new video starts a geographic search even when its first timestamp repeats');

searches=0;pipeline.beginSequence();pipeline.options.regionalIntervalSeconds=30;pipeline.options.mapIntervalSeconds=5;
const startSearches=searches;await estimate(0,0);assert.equal(searches,startSearches+1);
const localRefresh=await estimate(5,1);assert.equal(searches,startSearches+1,'local map checks do not execute area retrieval');
assert.equal(localRefresh.accepted,false,'local support cannot establish a unique geographic fix');
assert.equal(localRefresh.decision,'unresolved');assert.equal(localRefresh.retrieval.algorithm,'previous_pose_seeds_then_current_image_geometry');
assert.deepEqual(localRefresh.candidate_hypotheses[0].position_enu_m,[3,0,100]);
assert.deepEqual(localRefresh.candidate_hypotheses[0].relative_alternative.position_enu_m,[2,0,100],'map refresh retains the separate relative proposal');
assert.ok(localRefresh.local_map_check.candidate_hypotheses[0].accepted);
assert.equal(pipeline.temporal.lastRegionalSearchNs,0);assert.equal(pipeline.temporal.lastMapSearchNs,5e9);
const adjacent=await estimate(5.2,2);assert.equal(adjacent.decision,'relative_tracking');assert.equal(adjacent.local_map_check,undefined);
await estimate(10,3);await estimate(20,4);assert.equal(searches,startSearches+1);
await estimate(30,5);assert.equal(searches,startSearches+2,'successful local checks never postpone the scheduled geographic refresh');
console.info('Local map checks use their own cadence, preserve relative alternatives, and do not postpone geographic searches');

waiting.options.regionalIntervalSeconds=30;waiting.options.recoveryIntervalSeconds=5;waiting.beginSequence();
const recoverySearches=searches;await check(0,0);await check(.2,1);await check(5,2);
assert.equal(searches,recoverySearches+2,'failed acquisition keeps the shorter retry interval even with a longer supported-track refresh');
const originalTrack=pipeline.renderer.track,originalRefine=pipeline.renderer.refine,originalSelect=pipeline.renderer.select;
pipeline.renderer.track=()=>JSON.stringify({accepted:false,tracking_supported:false});
pipeline.renderer.refine=()=>JSON.stringify({accepted:false,reason:'No reference geometry'});
pipeline.renderer.select=()=>JSON.stringify({...stamp,accepted:false,decision:'rejected',candidate_hypotheses:[]});
const beforeLoss=searches,loss=await estimate(36,6);
assert.equal(searches,beforeLoss+1,'loss of supported motion and local map geometry triggers bounded area recovery before the normal area interval');
assert.equal(loss.candidate_hypotheses.length,0);assert.equal(pipeline.temporal.previous,null);
await estimate(36.2,7);assert.equal(searches,beforeLoss+1,'adjacent unsupported frames do not repeat the recovery search');
Object.assign(pipeline.renderer,{track:originalTrack,refine:originalRefine,select:originalSelect});
console.info('Lost tracks enter bounded area recovery without inventing a pose or reusing stale seeds');

searches=0;pipeline.beginSequence();await estimate(0,0);
let localAttempts=0;
pipeline.renderer.refine=()=>{localAttempts++;return JSON.stringify({accepted:false,reason:'No current map support'})};
pipeline.renderer.select=()=>JSON.stringify({...stamp,accepted:false,decision:'rejected',candidate_hypotheses:[]});
const unsupportedMap=await estimate(5,1);
assert.equal(unsupportedMap.decision,'relative_tracking','a failed map check does not promote or discard supported relative geometry');
assert.equal(unsupportedMap.accepted,false);assert.equal(unsupportedMap.local_map_check.decision,'rejected');
assert.equal(pipeline.temporal.lastMapSearchNs,5e9);assert.equal(pipeline.temporal.lastRegionalSearchNs,0);
await estimate(5.2,2);assert.equal(localAttempts,1,'a failed local check does not repeat on every adjacent sample');
Object.assign(pipeline.renderer,{refine:originalRefine,select:originalSelect});
console.info('Rejected local checks retain conditional tracking and preserve both retry clocks');

assert.ok(pipeline.imageSequence.current.sources.includes('observation-2'),'map rejection does not discard uploaded image observations');
