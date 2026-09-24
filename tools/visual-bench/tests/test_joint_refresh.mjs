import assert from 'node:assert/strict';
import {LocalizationPipeline} from '../webapp/localization.js';
globalThis.OffscreenCanvas=class {
 constructor(width,height){Object.assign(this,{width,height})}
 getContext(){return {drawImage(){},translate(){},rotate(){},getImageData:()=>({data:new Uint8Array(this.width*this.height*4)})}}
};
const pose=x=>({position_enu_m:[x,0,100],eye_to_enu_xyzw:[0,0,0,1],map_manifest_sha256:'map'});
let searches=0,checks=0,stamp,refinement,regionalSuccess=false,localSuccess=true;
const matcher={matchImages:async()=>({pairs:[],backend_identity:'test-matcher'}),retrievePairs:async()=>{searches++;return []}};
const pipeline=new LocalizationPipeline(matcher,{headings:4,refinements:1,regionalIntervalSeconds:5,mapIntervalSeconds:5});
pipeline.pack={anchor_lat_lon:[0,0]};pipeline.camera={width:2,height:2};pipeline.references={elevation:()=>0,crops:()=>[]};
pipeline.renderer={
 begin(_pixels,_prior,sequence,capture_time_ns){stamp={observation_sha256:`frame-${sequence}`,sequence,capture_time_ns};refinement=null},
 select(){const candidates=refinement?[refinement]:searches===1||regionalSuccess?[{candidate_id:0,accepted:true,...pose(80)}]:[{candidate_id:0,accepted:false,reason:'No area match'}];return JSON.stringify({...stamp,accepted:candidates[0].accepted,decision:candidates[0].accepted?'unique_among_evaluated':'rejected',candidate_hypotheses:candidates})},
 async render_reference(){return new Uint8Array(4)},
 track(){return JSON.stringify({candidate_id:0,accepted:false,tracking_supported:true,...pose(100)})},
 refine(id){checks++;refinement={candidate_id:id,accepted:localSuccess,...pose(20),...(!localSuccess?{reason:"No local map support"}:{})};return JSON.stringify(refinement)},
};
const prior={latitude:0,longitude:0,radius_m:500,agl_m:100};
const estimate=(time,sequence)=>pipeline.estimate({gray:new Uint8Array(4),width:2,height:2,canvas:{},time,timing:'browser decoded frame presentation timestamp'},prior,sequence,()=>{});
await estimate(0,0);
const recovered=await estimate(5,1);
assert.equal(searches,2,'the scheduled regional search still runs');
assert.equal(checks,1,'a regional search cannot suppress the scheduled local map check');
assert.equal(recovered.accepted,false,'local map support does not establish a unique geographic fix');
assert.equal(recovered.decision,'unresolved');
assert.deepEqual(recovered.candidate_hypotheses[0].position_enu_m,[20,0,100]);
assert.equal(recovered.candidate_hypotheses[0].accepted,true);
assert.deepEqual(recovered.candidate_hypotheses[0].relative_alternative.position_enu_m,[100,0,100]);
assert.equal(recovered.local_map_check.candidate_hypotheses[0].accepted,true);
assert.equal(recovered.regional_search.decision,'rejected');
assert.equal(recovered.regional_search.candidate_hypotheses[0].reason,'No area match');
assert.deepEqual(pipeline.temporal.seeds('next',[0,0,100],500)[0].position_enu_m,[20,0,100]);
assert.equal(pipeline.temporal.lastMapSearchNs,5e9);assert.equal(pipeline.temporal.lastRegionalSearchNs,5e9);
regionalSuccess=true;
const alternatives=await estimate(10,2),supported=alternatives.candidate_hypotheses.filter(h=>h.accepted||h.tracking_supported);
assert.equal(searches,3);assert.equal(checks,2);
assert.equal(alternatives.accepted,false);assert.equal(alternatives.decision,'unresolved');
assert.deepEqual(supported.map(h=>h.position_enu_m[0]),[80],'regional reacquisition keeps a bounded active candidate set');
assert.deepEqual(alternatives.regional_search.local_map_alternatives.map(h=>h.position_enu_m[0]),[20],'current local alternatives remain explicit and separate');
assert.equal(alternatives.regional_search.local_map_alternatives[0].accepted,true);
assert.deepEqual(alternatives.regional_search.local_map_alternatives[0].relative_alternative.position_enu_m,[100,0,100]);
assert.match(alternatives.evidence_correlation,/unknown/);
assert.deepEqual(pipeline.temporal.seeds('later',[0,0,100],500).map(h=>h.position_enu_m[0]),[80]);
console.info('Scheduled local and regional checks both run; failed searches retain map-supported poses and successful searches retain separate alternatives');

regionalSuccess=false;localSuccess=false;
const failedChecks=await estimate(15,3);
assert.equal(searches,4);assert.equal(checks,3);
assert.equal(failedChecks.accepted,false);assert.equal(failedChecks.decision,'relative_tracking');
assert.deepEqual(failedChecks.candidate_hypotheses[0].position_enu_m,[100,0,100]);
assert.equal(failedChecks.local_map_check.candidate_hypotheses[0].accepted,false);
assert.equal(failedChecks.regional_search.decision,'rejected');
await estimate(15.2,4);assert.equal(searches,4);assert.equal(checks,3);
console.info('Two failed map checks retain only conditional motion and do not repeat on an adjacent sample');
