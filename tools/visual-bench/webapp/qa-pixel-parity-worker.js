import {LocalMatcher} from './inference/local.js';
import {sha256} from './storage.js';
self.onmessage=async()=>{
 const report={test:'identical-pixel-loftr-parity',cases:[]};let matcher;
 const progress=progress=>self.postMessage({progress});
 try{
  const fixture=await(await fetch('./models/qa-pixel-parity/fixture.json')).json();report.scope=fixture.scope;
  matcher=new LocalMatcher({matcher:'dense',refinementPatches:true});await matcher.initialize(progress);
  for(const item of fixture.cases){
   const images={};
   for(const key of ['reference','query']){
    const response=await fetch('./models/qa-pixel-parity/'+item[key].replace(/\.png$/,'.gray'));if(!response.ok)throw Error('Missing raw pixel fixture');
    const gray=new Uint8Array(await response.arrayBuffer());if(gray.length!==fixture.width*fixture.height||await sha256(gray)!==item[key+'_sha256'])throw Error('Fixture pixel checksum differs');
    images[key]={gray,width:fixture.width,height:fixture.height};
   }
   for(const profile of ['base','crops']){
    progress(item.name+' · '+profile);const started=performance.now(),result=await matcher.matchImages(images.reference,images.query,{stage:profile==='crops'?'refinement':undefined});
    report.cases.push({name:item.name,negative:item.negative,profile,reference_image_sha256:item.reference_sha256,query_image_sha256:item.query_sha256,elapsed_ms:performance.now()-started,...result});
   }
   self.postMessage({report});
  }
  report.execution=matcher.diagnostics();report.phase='complete';
 }catch(error){report.phase='failed';report.error=String(error)}finally{await matcher?.close();self.postMessage({report,done:true})}
};
