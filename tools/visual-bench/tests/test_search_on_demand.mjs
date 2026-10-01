import assert from 'node:assert/strict';
import {LocalizationPipeline} from '../webapp/localization.js';
const pose=x=>({position_enu_m:[x,0,100],eye_to_enu_xyzw:[0,0,0,1],map_manifest_sha256:'map'});
globalThis.OffscreenCanvas=class{getContext(){return {drawImage(){},translate(){},rotate(){},getImageData:()=>({data:new Uint8Array(this.width*this.height*4)})}}};

async function run(options,{trackingSupported=true,times=[0,5,10]}={}){
  let stamp,searches=0;
  const matcher={matchImages:async()=>({pairs:[],backend_identity:'test'}),retrievePairs:async()=>[]};
  const pipeline=new LocalizationPipeline(matcher,{headings:4,refinements:2,mapMotionFraction:0,...options});
  pipeline.camera={width:2,height:2};pipeline.pack={anchor_lat_lon:[0,0]};
  pipeline.references={elevation:()=>0,crops:()=>{searches++;return []}};
  pipeline.renderer={
    begin(_p,_prior,sequence,capture_time_ns){stamp={sequence,capture_time_ns,observation_sha256:`frame-${sequence}`}},
    select(){return JSON.stringify({...stamp,accepted:false,decision:'unresolved',candidate_hypotheses:[{candidate_id:0,accepted:trackingSupported||stamp.sequence===0,...pose(0)}]})},
    async render_reference(){return new Uint8Array(4)},
    track(){return JSON.stringify({candidate_id:0,accepted:false,tracking_supported:trackingSupported,...pose(1)})},
    refine(id){return JSON.stringify(trackingSupported?{candidate_id:id,accepted:true,...pose(1)}:{candidate_id:id,accepted:false,reason:'Not enough geometric support'})}
  };
  const estimate=(time,sequence)=>pipeline.estimate({gray:new Uint8Array(4),width:2,height:2,canvas:{},time,timing:'decoded video'},{latitude:0,longitude:0,agl_m:100,radius_m:500},sequence,()=>{});
  for(const [sequence,time] of times.entries())await estimate(time,sequence);
  return searches;
}

assert.equal(await run({}),1,'supported tracking needs no timed area search after the first frame');
assert.equal(await run({regionalIntervalSeconds:5}),3,'a host can still request a timed area search');
assert.ok(await run({recoveryIntervalSeconds:5},{trackingSupported:false})>1,'lost tracking restarts the area search');
assert.equal(await run({},{trackingSupported:false,times:[0,1,2,3]}),1,'a short loss of tracking retries from the last supported pose');
assert.equal(await run({},{trackingSupported:false,times:[0,6]}),1,'the first failed frame is deferred so the search starts on a held frame');
assert.equal(await run({},{trackingSupported:false,times:[0,1,6]}),2,'the area search starts once tracking stays lost for the recovery interval');
console.info('Area search runs at cold start and after lost tracking, not on a timer');
