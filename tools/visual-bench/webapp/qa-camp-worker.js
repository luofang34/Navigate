import {streamFile} from './http-file.js';
import {CampIndex} from './inference/camp-index.js';
import {CampRetriever} from './inference/camp.js';
const hex=bytes=>[...new Uint8Array(bytes)].map(v=>v.toString(16).padStart(2,'0')).join('');
async function load(file,progress){const parts=[];let count=0;for await(const part of streamFile(file.url,file.size)){parts.push(part);count+=part.length;progress(`CAMP data ${Math.round(count/1048576)} / ${Math.round(file.size/1048576)} MB`)}const bytes=await new Blob(parts).arrayBuffer();if(hex(await crypto.subtle.digest('SHA-256',bytes))!==file.sha256)throw Error('CAMP data checksum differs');return bytes}
self.onmessage=async()=>{
 const progress=progress=>self.postMessage({progress}),report={test:'browser-camp-retrieval',phase:'loading',cases:[],geographic_accuracy:'not measured; ranked references are not camera poses'};let retriever;
 try{
  const manifest=await(await fetch('./models/research-camp/manifest.json',{cache:'no-store'})).json();
  const model=await load(manifest.model,progress),descriptors=await load(manifest.descriptors,progress);progress('Compiling browser CAMP…');
  const start=performance.now();retriever=await CampRetriever.create(model,new CampIndex(manifest.index,new Float32Array(descriptors)));report.load_ms=performance.now()-start;report.backend=retriever.identity;
  for(const query of manifest.cases){
   progress('CAMP '+query.id);const bitmap=await createImageBitmap(new Blob([await load(query.image,progress)]));const canvas=new OffscreenCanvas(bitmap.width,bitmap.height),ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(bitmap,0,0);bitmap.close();const rgba=ctx.getImageData(0,0,canvas.width,canvas.height).data,rgb=new Uint8Array(canvas.width*canvas.height*3);for(let i=0;i<rgb.length/3;i++)rgb.set(rgba.subarray(i*4,i*4+3),i*3);
   const start=performance.now(),groups=[];for(const eligible of query.eligible_groups)groups.push(await retriever.rank({rgb,width:canvas.width,height:canvas.height},eligible,10));
   report.cases.push({id:query.id,groups,expected_groups:query.expected_groups,top1_equal:groups.filter((g,i)=>g[0]===query.expected_groups[i][0]).length,exact_lists:groups.filter((g,i)=>JSON.stringify(g)===JSON.stringify(query.expected_groups[i])).length,elapsed_ms:performance.now()-start});report.phase='inference';report.execution=retriever.diagnostics();self.postMessage(report);
  }
  report.phase='complete';self.postMessage(report);
 }catch(error){report.phase='failed';report.error=String(error);report.stack=error.stack;self.postMessage(report)}finally{await retriever?.close()}
};
