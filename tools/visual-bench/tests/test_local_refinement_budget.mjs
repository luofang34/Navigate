import assert from 'node:assert/strict';
import {LocalizationPipeline} from '../webapp/localization.js';
const pose=x=>({position_enu_m:[x,0,100],eye_to_enu_xyzw:[0,0,0,1],map_manifest_sha256:'map'});
globalThis.OffscreenCanvas=class{getContext(){return {drawImage(){},translate(){},rotate(){},getImageData:()=>({data:new Uint8Array(this.width*this.height*4)})}}};
async function run(acceptAt,passes=3){
 let stamp,checked,count=0;const rendered=[];
 const matcher={matchImages:async()=>({pairs:[],backend_identity:'test-backend'}),retrievePairs:async()=>[]};
 const pipeline=new LocalizationPipeline(matcher,{headings:4,refinements:passes,regionalIntervalSeconds:60,mapIntervalSeconds:5,mapMotionFraction:0});
 pipeline.camera={width:2,height:2};pipeline.pack={anchor_lat_lon:[0,0]};pipeline.references={elevation:()=>0,crops:()=>[]};
 pipeline.renderer={
  begin(_pixels,_prior,sequence,capture_time_ns){stamp={sequence,capture_time_ns,observation_sha256:`frame-${sequence}`};checked=null},
  select(){return JSON.stringify({...stamp,accepted:false,decision:'unresolved',candidate_hypotheses:[checked??{candidate_id:0,accepted:true,...pose(0)}]})},
  async render_reference(_id,value){rendered.push(JSON.parse(value));return new Uint8Array(4)},
  track(){return JSON.stringify({candidate_id:0,accepted:false,tracking_supported:true,...pose(10)})},
  refine(id){count++;checked=count>=acceptAt?{candidate_id:id,accepted:true,...pose(count)}:{candidate_id:id,accepted:false,refinement_proposal:pose(count),reason:'Not enough geometric support'};return JSON.stringify(checked)}
 };
 const estimate=(time,sequence)=>pipeline.estimate({gray:new Uint8Array(4),width:2,height:2,canvas:{},time,timing:'decoded video'}, {latitude:0,longitude:0,agl_m:100,radius_m:500},sequence,()=>{});
 await estimate(0,0);const result=await estimate(5,1);return {count,rendered,result};
}
const recovered=await run(3);
assert.equal(recovered.count,3,'a rejected second pass can use the profile’s third pass');
assert.deepEqual(recovered.rendered.map(p=>p.position_enu_m[0]),[0,10,1,2],'each refinement proposal needs a new reference render');
assert.equal(recovered.result.local_map_check.candidate_hypotheses[0].accepted,true);
assert.equal(recovered.result.accepted,false,'local geometry does not establish a unique geographic fix');
assert.deepEqual(recovered.result.candidate_hypotheses[0].relative_alternative.position_enu_m,[10,0,100]);
const stable=await run(1);assert.equal(stable.count,2,'accepted local candidates keep the bounded two-pass cost');
const unresolved=await run(4);assert.equal(unresolved.count,3);assert.equal(unresolved.result.local_map_check.candidate_hypotheses[0].accepted,false);assert.equal(unresolved.result.decision,'relative_tracking');
const limited=await run(3,2);assert.equal(limited.count,2,'an explicit smaller profile budget is respected');assert.equal(limited.result.decision,'relative_tracking');
console.info('Local candidates use remaining refinement budget only after a rejected geometric check');
