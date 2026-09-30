import {similarityCandidates} from './retrieval-recovery.js';
import {LocalizationPipeline} from './localization.js';
import {LocalMatcher} from './inference/local.js';
import {refineCandidates} from './temporal-search.js';
import {refineReferenceCandidates,recoverRejectedCandidates} from './candidate-refinement.js';
import {download,sha256} from './storage.js';
import {decodeCameraImage} from './camera-image.js';
import {gray} from './observation.js';
self.onmessage=async({data})=>{
 const report={test:'failed-current-image-proposal-recovery',cases:[]};let pipeline;
 const progress=progress=>self.postMessage({progress});
 try{
  const fixtureName=data.fixture==='similarity'?'similarity-input.json':data.fixture==='retrieval'?'retrieval-recovery.json':data.fixture==='reference'?'recovery-reference.json':'recovery.json';const fixture=await(await fetch('./models/qa-pixel-parity/'+fixtureName)).json();report.scope=fixture.scope;report.fixture=fixtureName;
  const response=await fetch('/api/offline-plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({region_id:fixture.region})});if(!response.ok)throw Error('Missing local package');const pack=await response.json();await download(pack,()=>{});
  const matcher=new LocalMatcher({matcher:'dense',refinementPatches:true});pipeline=new LocalizationPipeline(matcher);await pipeline.initialize(pack,fixture.camera,progress);
  for(const item of fixture.cases){
   if(data.fixture==='similarity'){const prior=pipeline.navigationPrior(fixture.prior);item.candidates=similarityCandidates(item.ground_records,pipeline.proposeNadir,fixture.camera,prior.pose.position_enu_m,prior.position_radius_m);item.trace_indices=item.candidates.map(h=>h.source_index)}
   const response=await fetch('./models/test-frames/'+item.image),bytes=new Uint8Array(await response.arrayBuffer());if(await sha256(bytes)!==item.source_sha256)throw Error('Query source changed');
   const bitmap=await decodeCameraImage(new Blob([bytes],{type:'image/png'}));let image;try{image=gray(bitmap,fixture.camera.width,fixture.camera.height,true)}finally{bitmap.close()}
   for(const profile of ['retrieval','similarity'].includes(data.fixture)?['bounded']:['base','crops']){
    matcher.options.refinementPatches=profile==='bounded'?'on_rejection':profile==='crops';const start=performance.now();pipeline.renderer.begin(image.gray,JSON.stringify(pipeline.navigationPrior(fixture.prior)),0,0);const observation=JSON.parse(pipeline.renderer.select()).observation_sha256;
    const prior=pipeline.navigationPrior(fixture.prior),eligible=item.candidates.map((pose,index)=>({pose,index})).filter(h=>Math.hypot(...h.pose.position_enu_m.map((v,i)=>v-prior.pose.position_enu_m[i]))<=prior.position_radius_m);
    const limit=data.fixture==='similarity'?16:64,candidates=eligible.slice(0,limit).map(h=>h.pose),update=text=>progress(item.name+' · '+profile+' · '+text);
    let result;
    if(profile==='bounded'){
     const checked=await refineReferenceCandidates(pipeline.renderer,matcher,fixture.camera,candidates,image,observation,update,{candidateLimit:limit});
     result={...checked,...(data.fixture==='similarity'?{}:await recoverRejectedCandidates(pipeline.renderer,matcher,fixture.camera,image,observation,update,{limit:4,passes:2})),source_indices:eligible.slice(0,limit).map(h=>item.trace_indices[h.index]),unexamined_source_indices:eligible.slice(limit).map(h=>item.trace_indices[h.index])};
    }else result=await refineCandidates(pipeline.renderer,matcher,fixture.camera,item.candidates,image,observation,2,update,{recoveryAllowed:false});
    report.cases.push({name:item.name,negative:item.negative,profile,...(data.fixture==='similarity'?{seeds:item.candidates}:{}),elapsed_ms:performance.now()-start,result});self.postMessage({report});
   }
  }
  report.phase='complete';
 }catch(error){report.phase='failed';report.error=String(error)}finally{await pipeline?.close();self.postMessage({report,done:true})}
};
