import {refinementPairs} from './refinement-patches.js';
import {denseInput,supported,quarterTurn} from './loftr-input.js';
export class DenseMatcher {
 static async create(bytes,tracker,metrics){const ort=await import('../runtime/ort.webgpu.min.mjs');const self=new DenseMatcher();self.Tensor=ort.Tensor;self.tracker=tracker;self.metrics=metrics;self.session=await ort.InferenceSession.create(bytes,{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'});return self}
 async match(reference,query){
  for(const image of [reference,query])if(image.gray.length!==image.width*image.height||Math.min(image.width,image.height)<64||Math.max(image.width,image.height)>1920)throw Error('Invalid dense matcher image dimensions');
  const a=denseInput(reference),b=denseInput(query);if(a.flat||b.flat)return [];
  const feeds={image0:new this.Tensor('float32',a.data,[1,1,480,640]),image1:new this.Tensor('float32',b.data,[1,1,480,640])};let result;
  this.tracker.phase('matching');this.metrics.match_runs++;
  try{
   result=await this.session.run(feeds);const {keypoints0,keypoints1,confidence}=result,pairs=[];
   if(keypoints0.data.length!==confidence.data.length*2||keypoints1.data.length!==keypoints0.data.length)throw Error('Invalid dense matcher output shape');
   for(const tensor of [keypoints0,keypoints1,confidence])if(tensor.data.some(value=>!Number.isFinite(value)))throw Error('Nonfinite dense matcher output');
   for(let i=0;i<confidence.data.length&&pairs.length<4096;i++){
    const r=a.pixel(keypoints0.data.subarray(i*2,i*2+2)),q=b.pixel(keypoints1.data.subarray(i*2,i*2+2));
    if(confidence.data[i]>.2&&supported(reference,r)&&supported(query,q))pairs.push({reference:r,query:q});
   }
   this.metrics.correspondences_max=Math.max(this.metrics.correspondences_max||0,pairs.length);return pairs;
  }finally{for(const t of Object.values(feeds))t.dispose();if(result)for(const t of Object.values(result))t.dispose()}
 }
 refine(reference,query){return refinementPairs(reference,query,(a,b)=>this.match(a,b))}
 async *alternatives(reference,query){
  for(const turn of [1,2,3]){const rotated=quarterTurn(query,turn),pairs=await this.match(reference,rotated);yield pairs.map(p=>({reference:p.reference,query:rotated.unrotate(p.query)}))}
 }
 close(){return this.session.release()}
}
