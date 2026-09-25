import {TrackingPairCache} from './tracking-pair-cache.js';
import {assetUrl} from '../asset-url.js';
import {downloadFiles,read,get,put} from '../storage.js';
export class LocalMatcher {
  constructor(options={}){this.options=options;this.trackingPairs=new TrackingPairCache()}
  async initialize(progress){this.trackingPairs.clear();let manifest;try{const response=await fetch(assetUrl('models/manifest.json'));if(!response.ok)throw Error('Browser models are not prepared');manifest=await response.json();await put('state','public-xfeat-models-v2',manifest)}catch(e){manifest=await get('state','public-xfeat-models-v2')??await get('state','public-xfeat-models-v1');if(!manifest)throw e}
    if(!manifest.xfeat||Object.keys(manifest).some(name=>!['xfeat','lighterglue','loftr'].includes(name)))throw Error('Unsupported public matcher manifest');this.progress=progress;this.denseAsset=this.options.matcher==='dense'?manifest.loftr:null;const selected={xfeat:manifest.xfeat,...(!this.denseAsset&&manifest.lighterglue?{lighterglue:manifest.lighterglue}:{})};const files=Object.values(selected);await downloadFiles(files,(n,total)=>progress(`Matching model ${(n/1048576).toFixed(1)} / ${(total/1048576).toFixed(1)} MB`));
    const models={};for(const [name,f] of Object.entries(selected))models[name]=new Uint8Array(await read(`pilotage://chunks/${f.sha256}.bin`,0,f.size));
    progress('Loading image matcher…');const {XFeatMatcher}=await import('./xfeat.js');
    progress('Preparing image matcher…');this.matcher=await XFeatMatcher.create(models);this.identity=this.matcher.identity+'/'+files.map(f=>f.sha256).join('/');
  }
  async matchImages(reference,query,keys={}){
    if(this.denseAsset){
      const progress=keys.progress??this.progress;
      if(!this.dense){const f=this.denseAsset;await downloadFiles([f],(n,t)=>progress(`Image matching model ${(n/1048576).toFixed(1)} / ${(t/1048576).toFixed(1)} MB`));const {DenseMatcher}=await import('./loftr.js');this.dense=await DenseMatcher.create(new Uint8Array(await read(`pilotage://chunks/${f.sha256}.bin`,0,f.size)),this.matcher.gpu,this.matcher.metrics)}
      const patches=this.options.refinementPatches&&keys.stage==='refinement';let pairs=this.trackingPairs.get(keys);if(pairs===undefined){pairs=await (patches?this.dense.refine(reference,query):this.dense.match(reference,query));this.trackingPairs.put(keys,pairs)}return {pairs,backend_identity:`browser-loftr-ds-640x480/${this.denseAsset.sha256}/webgpu-wasm${patches?'/overlapping-refinement':''}`};
    }
const base=this.options.keypoints??1024,limit=this.matcher.glue?(['refinement','tracking'].includes(keys.stage)?Math.min(2048,Math.max(1024,base*2)):Math.min(2048,Math.max(512,base))):(['refinement','tracking'].includes(keys.stage)?Math.min(4096,Math.max(2048,base*3)):2048);const q=await this.matcher.features(query,keys.query,limit);return {pairs:q.count<6?[]:await this.matcher.pairs(await this.matcher.features(reference,keys.reference,limit),q),backend_identity:this.identity}}
  async *matchAlternatives(reference,query){
    if(this.dense)for await(const pairs of this.dense.alternatives(reference,query))yield {pairs,backend_identity:`browser-loftr-ds-640x480/${this.denseAsset.sha256}/rotation-search-webgpu-wasm`};
  }
  async retrievePairs(references,queries,limit,progress){
    const features=[],queryFeatures=[];
    for(const [i,reference] of references.entries()){progress(`Preparing reference images ${i+1}/${references.length}`);features.push(await this.matcher.features(reference.image,reference.key,512,'reference'))}
    for(const [q,query] of queries.entries()){
      progress(`Preparing camera views ${q+1}/${queries.length}`);queryFeatures.push(await this.matcher.features(query.image,query.key));
    }
    this.matcher.gpu.phase('retrieval');return this.matcher.retrieval.rankGrid(features,queryFeatures,limit,progress);
  }
  diagnostics(){return {counter_scope:"worker lifetime; feature outputs can be cached",...this.matcher.metrics,tracking_pair_cache_hits:this.trackingPairs.hits,tracking_pair_cache_misses:this.trackingPairs.misses,tracking_pair_cache_bytes:this.trackingPairs.bytes,reference_feature_cache_hits:this.matcher.referenceCache?.hits,reference_feature_cache_misses:this.matcher.referenceCache?.misses,reference_feature_cache_bytes:this.matcher.referenceCache?.bytes,frame_feature_cache_bytes:this.matcher.cache?.bytes,retrieval_gpu_dispatches:this.matcher.retrieval.dispatches,descriptor_upload_count:this.matcher.retrieval.uploadCount,descriptor_upload_bytes:this.matcher.retrieval.uploadBytes,execution:'WebGPU kernels with WASM fallback for unsupported model operators'}}
  async close(){this.trackingPairs.clear();await this.dense?.close();await this.matcher.close()}
}
