import assert from 'node:assert/strict';
import {LocalizationPipeline} from '../webapp/localization.js';
globalThis.OffscreenCanvas=class{getContext(){return {drawImage(){},translate(){},rotate(){},getImageData:()=>({data:new Uint8Array(this.width*this.height*4)})}}};

async function search(inliers){
  let compared=0,stamp;
  const pairs=Array.from({length:20},(_,i)=>({reference:[i,i],query:[i,i]}));
  const matcher={matchImages:async()=>{compared++;return {pairs,backend_identity:'test'}},
    retrievePairs:async(crops,queries)=>crops.map((_,r)=>({reference_index:r,query_index:0}))};
  const pipeline=new LocalizationPipeline(matcher,{headings:4,shortlist:40,candidates:8,refinements:1});
  pipeline.camera={width:16,height:12};pipeline.pack={anchor_lat_lon:[0,0]};
  const crop=i=>({key:`crop-${i}`,image:{gray:new Uint8Array(4),width:2,height:2},world:p=>[p[0],p[1],0]});
  pipeline.references={elevation:()=>0,crops:()=>Array.from({length:40},(_,i)=>crop(i))};
  pipeline.propose=()=>JSON.stringify({retrieved:true,retrieval_inliers:inliers,position_enu_m:[0,0,100],eye_to_enu_xyzw:[0,0,0,1]});
  pipeline.renderer={begin(_p,_prior,sequence,capture_time_ns){stamp={sequence,capture_time_ns,observation_sha256:'frame'}},
    select(){return JSON.stringify({...stamp,accepted:false,decision:'unresolved',candidate_hypotheses:[]})},
    async render_reference(){return new Uint8Array(4)},refine(id){return JSON.stringify({candidate_id:id,accepted:false,reason:'test'})}};
  const report=await pipeline.estimate({gray:new Uint8Array(16*12),width:16,height:12,canvas:{},time:0,timing:'still image'},{latitude:0,longitude:0,agl_m:100,radius_m:500},0,()=>{});
  return {compared,retrieval:report.retrieval};
}

const strong=await search(40);
assert.equal(strong.retrieval.compared_pairs,8,'the search stops after enough strong proposals');
assert.equal(strong.retrieval.shortlist_pairs,40);
const weak=await search(12);
assert.equal(weak.retrieval.compared_pairs,40,'weak proposals do not stop the search');
assert.equal(weak.retrieval.shortlist_stop,'shortlist exhausted');
console.info('Area search stops comparing once the refined candidate set is full of strong proposals');
