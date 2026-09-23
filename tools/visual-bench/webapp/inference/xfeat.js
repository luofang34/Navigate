import {assetUrl} from '../asset-url.js';
import * as ort from '../runtime/ort.webgpu.min.mjs';
import {learnedPairs} from './lighterglue.js';
import {instrumentDevice} from './gpu-metrics.js';
import {DescriptorRetrieval} from './retrieval-gpu.js';
import {decodeXFeat,xfeatInput,emptyXFeat,isEmptyXFeatOutput} from './xfeat-decode.js';
export class XFeatMatcher {
  static async create(models){
    ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths=assetUrl('runtime/');
    const self=new XFeatMatcher();self.cache=new Map();self.metrics={feature_runs:0,match_runs:0};
    const adapter=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});
    if(!adapter)throw Error('A WebGPU adapter is required for browser matching');
    self.metrics.adapter={vendor:adapter.info?.vendor,architecture:adapter.info?.architecture,description:adapter.info?.description};ort.env.webgpu.adapter=adapter;
    self.point=await ort.InferenceSession.create(models.xfeat,{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'});
    self.gpu=instrumentDevice(ort.env.webgpu.device,self.metrics);
    self.retrieval=await DescriptorRetrieval.create(ort.env.webgpu.device,64,{minimumSimilarity:.82,maximumDistanceRatio:.9});
    if(models.lighterglue)self.glue=await ort.InferenceSession.create(models.lighterglue,{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'});
    self.identity=`browser-xfeat/${self.glue?'lighterglue':'mutual-nearest'}-webgpu/onnxruntime-webgpu-wasm`;return self;
  }
  async features(image,key,limit=512){
    if(image.gray.length!==image.width*image.height||Math.min(image.width,image.height)<64||Math.max(image.width,image.height)>1920||limit>4096)throw Error('Invalid matcher image dimensions or feature limit');
    if(key)key=key+'/keypoints/'+limit;if(key&&this.cache.has(key))return this.cache.get(key);
    let min=255,max=0;for(const value of image.gray){min=Math.min(min,value);max=Math.max(max,value)}
    if(max-min<2)return emptyXFeat(image.width,image.height);
    const {data,width,height,transform}=xfeatInput(image);
    const tensor=new ort.Tensor('float32',data,[1,3,height,width]);let result;
    this.gpu.phase('feature');this.metrics.feature_runs++;
    try {
      result=await this.point.run({images:tensor});const value=decodeXFeat(result,image,limit,transform);this.metrics.feature_count_max=Math.max(this.metrics.feature_count_max||0,value.count);
      if(key){this.cache.set(key,value);if(this.cache.size>192)this.cache.delete(this.cache.keys().next().value)}return value;
    } catch(error){
      // The sparse export has an invalid broadcast only when its detector returns zero points.
      if(isEmptyXFeatOutput(error)){this.metrics.empty_feature_runs=(this.metrics.empty_feature_runs||0)+1;return emptyXFeat(image.width,image.height)}throw error;
    } finally {tensor.dispose();if(result)for(const value of Object.values(result))value.dispose()}
  }
  async pairs(first,second){
    if(first.count<6||second.count<6)return [];
    this.gpu.phase('matching');this.metrics.match_runs++;
    const indices=this.glue?await learnedPairs(this.glue,first,second):await this.retrieval.matchPairs(first,second);this.metrics.correspondences_max=Math.max(this.metrics.correspondences_max||0,indices.length);return indices.map(([r,q])=>({reference:[first.pixels[r*2],first.pixels[r*2+1]],query:[second.pixels[q*2],second.pixels[q*2+1]]}));
  }
  async close(){this.gpu.restore();this.retrieval.close();await this.point.release();await this.glue?.release()}
}
