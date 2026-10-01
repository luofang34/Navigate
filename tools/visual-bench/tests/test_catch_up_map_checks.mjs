import assert from 'node:assert/strict';
import {LocalizationPipeline} from '../webapp/localization.js';
globalThis.OffscreenCanvas=class{getContext(){return {drawImage(){},translate(){},rotate(){},getImageData:()=>({data:new Uint8Array(this.width*this.height*4)})}}};
let alternatives=0;
const pose=x=>({position_enu_m:[x,0,100],eye_to_enu_xyzw:[0,0,0,1],map_manifest_sha256:'map'});

async function mapChecks(catchUp,{tracking=true}={}){
  let stamp,checks=0;
  const pipeline=new LocalizationPipeline({matchImages:async()=>({pairs:[],backend_identity:'test'}),retrievePairs:async()=>[],async *matchAlternatives(){alternatives++;yield {pairs:[],backend_identity:'alternative'}}},{headings:4,refinements:1,mapMotionFraction:0});
  pipeline.camera={width:2,height:2};pipeline.pack={anchor_lat_lon:[0,0]};
  pipeline.references={elevation:()=>0,crops:()=>[]};
  pipeline.renderer={
    begin(_p,_prior,sequence,capture_time_ns){stamp={sequence,capture_time_ns,observation_sha256:`frame-${sequence}`}},
    select(){return JSON.stringify({...stamp,accepted:false,decision:'unresolved',candidate_hypotheses:[{candidate_id:0,accepted:stamp.sequence===0,...pose(0)}]})},
    async render_reference(){return new Uint8Array(4)},
    track(){return JSON.stringify({candidate_id:0,accepted:false,tracking_supported:tracking,...pose(1)})},
    refine(id){checks++;return JSON.stringify({candidate_id:id,accepted:true,...pose(1)})}
  };
  for(let sequence=0;sequence<=(tracking?15:2);sequence++)
    await pipeline.estimate({gray:new Uint8Array(4),width:2,height:2,canvas:{},time:sequence*2,timing:'decoded video',catch_up:sequence>0?catchUp:undefined},{latitude:0,longitude:0,agl_m:100,radius_m:500},sequence,()=>{});
  return checks;
}
const live=await mapChecks(false),catching=await mapChecks('hop');
assert.ok(catching*2<live,`catch-up frames check the map less often: ${catching} against ${live}`);
assert.ok(catching>=1,'catch-up frames still receive map checks');
alternatives=0;assert.equal(await mapChecks('hop',{tracking:false}),0,'a failed catch-up hop defers without a map check');
assert.equal(alternatives,0,'a failed catch-up hop does not try alternative matchers');
await mapChecks(false,{tracking:false});assert.ok(alternatives>0,'a failed live frame still tries alternative matchers');
assert.ok(await mapChecks('nearest',{tracking:false})>0,'the nearest buffered frame still checks the map when tracking fails');
assert.ok(await mapChecks(false,{tracking:false})>0,'a failed live frame still checks the map');
console.info('Catch-up frames space map checks so tracking gains on the source');
