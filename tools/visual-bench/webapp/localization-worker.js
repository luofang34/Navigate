import {LocalizationPipeline} from './localization.js';
import {LocalMatcher} from './inference/local.js';
let pipeline;
self.onmessage=async({data})=>{const {id,method,args}=data;try{
  const progress=text=>self.postMessage({id,progress:text});let value;
  if(method==='initialize'){const [pack,camera,options={}]=args;pipeline=new LocalizationPipeline(new LocalMatcher(options),options);await pipeline.initialize(pack,camera,progress);value=true}
  else if(method==='estimate'){
    const image=args[0],canvas=new OffscreenCanvas(image.width,image.height),rgba=new Uint8ClampedArray(image.gray.length*4);
    for(let i=0;i<image.gray.length;i++){rgba[i*4]=rgba[i*4+1]=rgba[i*4+2]=image.gray[i];rgba[i*4+3]=255}
    canvas.getContext('2d').putImageData(new ImageData(rgba,image.width,image.height),0,0);image.canvas=canvas;
    value=await pipeline.estimate(...args,progress);
  }else throw Error('Unknown localization request');self.postMessage({id,value});
}catch(error){self.postMessage({id,error:String(error)})}};
