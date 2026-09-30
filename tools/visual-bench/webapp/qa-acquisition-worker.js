import {retrievalFixture,refineRetrievalFixture} from './qa-retrieval-fixture.js';
import {refineCandidates} from './temporal-search.js';
import {createAcquisitionPipeline} from './qa-acquisition-pipeline.js';
let pipeline,records=[],current,retrievalTrace;
self.onmessage=async({data:{id,method,args}})=>{
 const progress=text=>self.postMessage({id,progress:text});
 try{
  let value;
  if(method==='initialize'){
   pipeline=await createAcquisitionPipeline(args[0],args[1],args[2]??{},progress);const matcher=pipeline.matcher;
   const match=matcher.matchImages.bind(matcher),propose=pipeline.propose;
   if(matcher.retrievePairs){const retrieve=matcher.retrievePairs.bind(matcher);
   matcher.retrievePairs=async(refs,queries,...rest)=>{
    const ranks=await retrieve(refs,queries,...rest);matcher.references=new Map(refs.map(r=>[r.key,r]));retrievalTrace={references:refs.map(r=>({key:r.key,span_m:r.span_m,center:r.world([(r.image.width-1)/2,(r.image.height-1)/2])})),queries:queries.map(q=>({angle:q.angle})),ranks};return ranks;
   };}
   matcher.matchImages=async(reference,query,keys={})=>{
    const result=await match(reference,query,keys);
    if(!keys.stage){const crop=matcher.references.get(keys.reference);current={crop:keys.reference,angle:Number(keys.query.split('/').at(-1)),center:crop.world([319.5,319.5]),span_m:crop.span_m,pairs:result.pairs.length};records.push(current)}
    return result;
   };
   if(matcher.retrievePairs)pipeline.propose=(camera,ground)=>{const result=propose(camera,ground);current.proposal=JSON.parse(result);current.ground=JSON.parse(ground);return result};value={original_pixels:Boolean(pipeline.referenceSearch)};
  }else if(method==='beginSequence'){pipeline.beginSequence(...args);records=[];retrievalTrace=undefined;value=true}
  else if(method==='estimate'){
   const image=args[0],canvas=new OffscreenCanvas(image.width,image.height),rgba=new Uint8ClampedArray(image.gray.length*4);
   for(let i=0;i<image.gray.length;i++){rgba[i*4]=rgba[i*4+1]=rgba[i*4+2]=image.gray[i];rgba[i*4+3]=255}
   canvas.getContext('2d').putImageData(new ImageData(rgba,image.width,image.height),0,0);image.canvas=canvas;
   if(pipeline.options.traceReplay){
    const fixture=await(await fetch('./models/research-lightglue/acquisition-seeds.json',{cache:'no-store'})).json(),start=performance.now();
    const seeds=fixture.schema===2?retrievalFixture(fixture,args[3]?.source_sha256,pipeline.pack,args[1]):fixture;
    pipeline.renderer.begin(image.gray,JSON.stringify(pipeline.navigationPrior(args[1])),args[2],0);
    const observation=JSON.parse(pipeline.renderer.select()).observation_sha256;
    const result=fixture.schema===2?await refineRetrievalFixture(pipeline,seeds,image,observation,progress):await refineCandidates(pipeline.renderer,pipeline.matcher,pipeline.camera,seeds.candidates,image,observation,3,progress,{recoveryAllowed:false});
    value={...result,processing_ms:performance.now()-start,execution:pipeline.matcher.diagnostics(),retrieval:{source:seeds.source,scope:seeds.scope,source_sha256:seeds.source_sha256,reference_ids:seeds.reference_ids,pose_candidates:seeds.candidates.length}};
   }else value={...await pipeline.estimate(...args,progress),acquisition_trace:records,retrieval_trace:retrievalTrace};
  }else throw Error('Unsupported acquisition trace request');
  self.postMessage({id,value});
 }catch(error){self.postMessage({id,error:String(error)})}finally{if(method==='estimate')args[0]?.original?.bitmap?.close()}
};
