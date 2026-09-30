import {assetUrl} from '../asset-url.js';
import {instrumentDevice} from './gpu-metrics.js';
import {campPixels,validateCampImage} from './camp-pixels.js';
import {validateCampDescriptor} from './camp-index.js';
const digest=async bytes=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
export class CampRetriever {
 static async create(bytes,index,shared){
  const hash=await digest(bytes);if(hash!==index.modelSha256)throw Error('CAMP model differs from the pinned index');
  const indexHash=await digest(new TextEncoder().encode(JSON.stringify([index.catalog,[...index.rows.keys()],await digest(index.descriptors)])));
  const ort=await import('../runtime/ort.webgpu.min.mjs');ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths=assetUrl('runtime/');
  const self=new CampRetriever();self.Tensor=ort.Tensor;self.index=index;self.catalog=index.catalog;self.metrics=shared?.metrics??{};self.ownsTracker=!shared;
  if(!shared){const adapter=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('CAMP browser retrieval needs WebGPU');ort.env.webgpu.adapter=adapter;self.metrics.adapter={vendor:adapter.info?.vendor,architecture:adapter.info?.architecture}}
  self.session=await ort.InferenceSession.create(bytes,{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'});
  self.gpu=shared?.gpu??instrumentDevice(ort.env.webgpu.device,self.metrics);
  self.identity=`camp-global-rgb-linear-u8-v1/${hash}/${indexHash}/webgpu-wasm`;
  return self;
 }
 async rank(image,eligible,limit){
  const rows=this.index.eligible(eligible,limit);if(!rows.length||!limit)return [];
  validateCampImage(image);const {width,height}=image,rgb=Uint8Array.from(image.rgb),key=`${width}x${height}/${await digest(rgb)}`;
  if(this.cached?.key!==key){
   const pixels=campPixels({width,height,rgb});this.metrics.place_retrieval_preprocessing_runs=(this.metrics.place_retrieval_preprocessing_runs??0)+1;
   const tensor=new this.Tensor('float32',pixels,[1,3,384,384]);let result;
   this.gpu.phase('place_retrieval');this.metrics.place_retrieval_runs=(this.metrics.place_retrieval_runs??0)+1;
   try{
    result=await this.session.run({image:tensor});const output=result.descriptor;
    if(!output||JSON.stringify(output.dims)!=='[1,1024]')throw Error('Invalid CAMP output tensor');
    validateCampDescriptor(output.data);this.cached={key,descriptor:Float32Array.from(output.data)};
   }finally{tensor.dispose();if(result)for(const tensor of Object.values(result))tensor.dispose()}
  }
  return this.index.rank(this.cached.descriptor,rows,limit);
 }
 diagnostics(){return {...this.metrics,execution:'WebGPU with WASM fallback; ranked reference IDs only'}}
 async close(){this.cached=null;if(this.ownsTracker)this.gpu?.restore();await this.session?.release()}
}
