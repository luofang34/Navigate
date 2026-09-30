import {createAcquisitionPipeline} from './qa-acquisition-pipeline.js';
import {makeCanvas} from './observation.js';
let pipeline;
self.onmessage=async({data:{id,method,args}})=>{
 const progress=text=>self.postMessage({id,progress:text});
 try{
  let value;
  if(method==='initialize'){pipeline=await createAcquisitionPipeline(args[0],args[1],{traceMatching:'lightglue',...args[2]},progress);value={original_pixels:Boolean(pipeline.referenceSearch)}}
  else if(method==='beginSequence'){pipeline.beginSequence(...args);value=true}
  else if(method==='estimate'){
   const image=args[0],canvas=makeCanvas();canvas.width=image.width;canvas.height=image.height;
   const rgba=new Uint8ClampedArray(image.gray.length*4);
   for(let i=0;i<image.gray.length;i++){rgba[i*4]=rgba[i*4+1]=rgba[i*4+2]=image.gray[i];rgba[i*4+3]=255}
   canvas.getContext('2d').putImageData(new ImageData(rgba,image.width,image.height),0,0);image.canvas=canvas;
   value=await pipeline.estimate(...args,progress);
  }else throw Error('Unsupported research request');
  self.postMessage({id,value});
 }catch(error){self.postMessage({id,error:String(error)})}finally{if(method==='estimate')args[0]?.original?.bitmap?.close()}
};
