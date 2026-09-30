import {streamFile} from './http-file.js';
import {detectorInput,decodeDetector,decodeAssignments} from './qa-superpoint-decode.js';
import {instrumentDevice} from './inference/gpu-metrics.js';
export class ResearchLightGlue {
 async initialize(progress,shared){
  const ort=await import('./runtime/ort.webgpu.min.mjs');this.Tensor=ort.Tensor;
  ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths=new URL('./runtime/',import.meta.url).href;
  let adapterInfo;
  if(!shared){const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('No WebGPU adapter');ort.env.webgpu.adapter=adapter;adapterInfo={vendor:adapter.info?.vendor,architecture:adapter.info?.architecture,description:adapter.info?.description}}
  const manifest=await(await fetch('/models/research-lightglue/manifest.json')).json(),models={};
  for(const [name,file] of Object.entries(manifest)){
   progress('Loading local research '+name);const response=await fetch(file.url,{method:'HEAD',cache:'no-store'});if(!response.ok)throw Error('Missing local research model');const parts=[];for await(const part of streamFile(file.url,Number(response.headers.get('Content-Length'))))parts.push(part);const bytes=await new Blob(parts).arrayBuffer();
   const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');if(hash!==file.sha256)throw Error('Research model hash mismatch');models[name]=bytes;
  }
  this.detector=await ort.InferenceSession.create(models.detector,{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'});
  this.metrics=shared?.metrics??{feature_runs:0,match_runs:0,adapter:adapterInfo};this.ownsTracker=!shared;this.gpu=shared?.gpu??instrumentDevice(ort.env.webgpu.device,this.metrics);
  this.matcher=await ort.InferenceSession.create(models.matcher,{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'});
  this.identity='research-superpoint-lightglue-full/webgpu-wasm/'+manifest.detector.sha256+'/'+manifest.matcher.sha256;this.cache=new Map();
 }
 async features(image){
  const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',image.gray))].map(v=>v.toString(16).padStart(2,'0')).join(''),key=image.width+'x'+image.height+'/'+hash;
  if(!image.valid&&this.cache.has(key)){const value=this.cache.get(key);this.cache.delete(key);this.cache.set(key,value);return value}
  const input=detectorInput(image),tensor=new this.Tensor('float32',input.data,[1,1,360,640]);let output;
  this.gpu.phase('feature');this.metrics.feature_runs++;
  try{output=await this.detector.run({image:tensor});const value=decodeDetector(output,image,input);if(!image.valid){this.cache.set(key,value);if(this.cache.size>2)this.cache.delete(this.cache.keys().next().value)}return value}
  finally{tensor.dispose();if(output)for(const tensor of Object.values(output))tensor.dispose()}
 }
 async matchImages(reference,query){
  const a=await this.features(reference),b=await this.features(query);
  if(Math.min(a.pixels.length,b.pixels.length)<2)return {pairs:[],backend_identity:this.identity};
  const feeds={};for(const [i,f] of [a,b].entries()){feeds['keypoints'+i]=new this.Tensor('float32',f.points,[1,f.pixels.length,2]);feeds['descriptors'+i]=new this.Tensor('float32',f.descriptors,[1,f.pixels.length,256])}
  let output;this.gpu.phase('matching');this.metrics.match_runs++;
  try{output=await this.matcher.run(feeds);return {pairs:decodeAssignments(output.matches0.data,a,b),backend_identity:this.identity}}
  finally{for(const tensor of Object.values(feeds))tensor.dispose();if(output)for(const tensor of Object.values(output))tensor.dispose()}
 }
 diagnostics(){return {...this.metrics,execution:'WebGPU with WASM fallback; local research weights; no exclusive GPU placement claim'}}
 async close(){this.cache?.clear();if(this.ownsTracker)this.gpu?.restore();await this.matcher?.release();await this.detector?.release()}
}
