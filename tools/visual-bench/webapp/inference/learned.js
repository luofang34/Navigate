import {assetUrl} from '../asset-url.js';
import * as ort from '../runtime/ort.webgpu.min.mjs';
import {instrumentDevice} from './gpu-metrics.js';
import {DescriptorRetrieval} from './retrieval-gpu.js';
import {decode} from './superpoint.js';
export class LearnedMatcher {
  static async create(models){
    ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths=assetUrl('runtime/');
    const self=new LearnedMatcher();self.cache=new Map();self.metrics={feature_runs:0,match_runs:0};
    const adapter=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('A WebGPU adapter is required for browser matching');
    self.metrics.adapter={vendor:adapter.info?.vendor,architecture:adapter.info?.architecture,description:adapter.info?.description};
    ort.env.webgpu.adapter=adapter;
    self.point=await ort.InferenceSession.create(models.superpoint,{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'});
    self.glue=await ort.InferenceSession.create(models.superglue,{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'});
    self.gpu=instrumentDevice(ort.env.webgpu.device,self.metrics);
    self.retrieval=await DescriptorRetrieval.create(ort.env.webgpu.device);self.identity='browser-superpoint-superglue/onnxruntime-webgpu-wasm';return self;
  }
  async features(image,key,limit=512){
    if(image.gray.length!==image.width*image.height||image.width%8||image.height%8||Math.min(image.width,image.height)<64||Math.max(image.width,image.height)>1920)throw Error("Invalid matcher image dimensions");
    if(key)key=`${key}/keypoints/${limit}`;
    if(key&&this.cache.has(key))return this.cache.get(key);
    const input=new Float32Array(image.gray.length);for(let i=0;i<input.length;i++)input[i]=image.gray[i]/255;
    this.gpu.phase("superpoint");this.metrics.feature_runs++;const result=await this.point.run({image:new ort.Tensor('float32',input,[1,1,image.height,image.width])});
    const value=decode(result.scores,result.descriptors,image.width,image.height,limit,image.valid);
    for(const tensor of Object.values(result))tensor.dispose();
    if(key){this.cache.set(key,value);if(this.cache.size>192)this.cache.delete(this.cache.keys().next().value)}return value;
  }
  async pairs(first,second){
    if(first.count<6||second.count<6)return [];
    const feeds={};for(const [index,f] of [first,second].entries()){
      const coords=new Float32Array(f.pixels.length),scale=640/Math.max(f.width,f.height);
      for(let i=0;i<f.count;i++){coords[i*2]=(f.pixels[i*2]-f.width/2)*scale+320;coords[i*2+1]=(f.pixels[i*2+1]-f.height/2)*scale+180}
      feeds['keypoints'+index]=new ort.Tensor('float32',coords,[1,f.count,2]);feeds['scores'+index]=new ort.Tensor('float32',f.scores,[1,f.count]);feeds['descriptors'+index]=new ort.Tensor('float32',f.descriptors,[1,256,f.count]);
    }
    this.gpu.phase("superglue");this.metrics.match_runs++;const result=await this.glue.run(feeds),indices=result.matches0.data,pairs=[];
    for(let i=0;i<indices.length;i++){const j=Number(indices[i]);if(j>=0)pairs.push({reference:[first.pixels[i*2],first.pixels[i*2+1]],query:[second.pixels[j*2],second.pixels[j*2+1]]})}
    for(const tensor of [...Object.values(result),...Object.values(feeds)])tensor.dispose();return pairs;
  }
  async matchImages(first,second){return {pairs:await this.pairs(await this.features(first),await this.features(second)),backend_identity:this.identity}}
  async close(){this.gpu.restore();this.retrieval.close();await this.point.release();await this.glue.release()}
}
