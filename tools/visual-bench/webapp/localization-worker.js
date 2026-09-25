import {LocalizationPipeline} from './localization.js';
import {LocalMatcher} from './inference/local.js';
let pipeline;
self.onmessage=async({data})=>{const {id,method,args}=data;try{
  const progress=text=>self.postMessage({id,progress:text});let value;
  if(method==='initialize'){const [pack,camera,options={}]=args;pipeline=new LocalizationPipeline(new LocalMatcher(options),options);await pipeline.initialize(pack,camera,progress);value=true}
  else if(method==='beginSequence'){pipeline.beginSequence(...args);value=true}
  else if(method==='finishSequence')value=await pipeline.finishSequence();
  else if(method==='reconstructSequence')value=await pipeline.reconstructSequence(...args,progress);
  else if(method==='observeScene')value=await pipeline.observeScene(...args,progress);
  else if(method==='refineScenePaths')value=await pipeline.refineScenePaths(...args,progress);
  else if(method==='refineScenes')value=await pipeline.refineScenes(...args,progress);
  else if(method==='registerScene')value=await pipeline.registerScene(...args,progress);
  else if(method==='estimate'){
    const image=args[0],canvas=new OffscreenCanvas(image.width,image.height),rgba=new Uint8ClampedArray(image.gray.length*4);
    for(let i=0;i<image.gray.length;i++){rgba[i*4]=rgba[i*4+1]=rgba[i*4+2]=image.gray[i];rgba[i*4+3]=255}
    canvas.getContext('2d').putImageData(new ImageData(rgba,image.width,image.height),0,0);image.canvas=canvas;
    value=await pipeline.estimate(...args,progress);
  }else if(method==='trackFrom')value=await pipeline.trackFrom(...args,progress);
  else if(method==='checkMotion')value=await pipeline.checkMotion(...args,progress);
  else if(method==='refineAt')value=await pipeline.refineAt(...args,progress);
  else throw Error('Unknown localization request');self.postMessage({id,value});
}catch(error){self.postMessage({id,error:String(error)})}};
