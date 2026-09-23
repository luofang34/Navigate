import {assetUrl} from '../asset-url.js';
import {downloadFiles,read,get,put} from '../storage.js';
export class LocalMatcher {
  constructor(options={}){this.options=options}
  async initialize(progress){let manifest;try{const response=await fetch(assetUrl('models/manifest.json'));if(!response.ok)throw Error('Browser models are not prepared');manifest=await response.json();await put('state','public-xfeat-models-v2',manifest)}catch(e){manifest=await get('state','public-xfeat-models-v2')??await get('state','public-xfeat-models-v1');if(!manifest)throw e}
    if(!manifest.xfeat||Object.keys(manifest).some(name=>!['xfeat','lighterglue'].includes(name)))throw Error('Unsupported public matcher manifest');const files=Object.values(manifest);await downloadFiles(files,(n,total)=>progress(`Matching model ${(n/1048576).toFixed(1)} / ${(total/1048576).toFixed(1)} MB`));
    const models={};for(const [name,f] of Object.entries(manifest))models[name]=new Uint8Array(await read(`pilotage://chunks/${f.sha256}.bin`,0,f.size));
    progress('Loading image matcher…');const {XFeatMatcher}=await import('./xfeat.js');
    progress('Preparing image matcher…');this.matcher=await XFeatMatcher.create(models);this.identity=this.matcher.identity+'/'+files.map(f=>f.sha256).join('/');
  }
  async matchImages(reference,query,keys={}){const base=this.options.keypoints??1024,limit=this.matcher.glue?(keys.stage==='refinement'?Math.min(2048,Math.max(1024,base*2)):Math.min(2048,Math.max(512,base))):(keys.stage==='refinement'?Math.min(4096,Math.max(2048,base*3)):2048);const q=await this.matcher.features(query,keys.query,limit);return {pairs:q.count<6?[]:await this.matcher.pairs(await this.matcher.features(reference,keys.reference,limit),q),backend_identity:this.identity}}
  async retrievePairs(references,queries,limit,progress){
    const features=[],queryFeatures=[];
    for(const [i,reference] of references.entries()){progress(`Preparing reference images ${i+1}/${references.length}`);features.push(await this.matcher.features(reference.image,reference.key,512,'reference'))}
    for(const [q,query] of queries.entries()){
      progress(`Preparing camera views ${q+1}/${queries.length}`);queryFeatures.push(await this.matcher.features(query.image,query.key));
    }
    this.matcher.gpu.phase('retrieval');return this.matcher.retrieval.rankGrid(features,queryFeatures,limit,progress);
  }
  diagnostics(){return {counter_scope:"worker lifetime; feature outputs can be cached",...this.matcher.metrics,reference_feature_cache_hits:this.matcher.referenceCache?.hits,reference_feature_cache_misses:this.matcher.referenceCache?.misses,reference_feature_cache_bytes:this.matcher.referenceCache?.bytes,frame_feature_cache_bytes:this.matcher.cache?.bytes,retrieval_gpu_dispatches:this.matcher.retrieval.dispatches,descriptor_upload_count:this.matcher.retrieval.uploadCount,descriptor_upload_bytes:this.matcher.retrieval.uploadBytes,execution:'WebGPU kernels with WASM fallback for unsupported model operators'}}
  close(){return this.matcher.close()}
}
