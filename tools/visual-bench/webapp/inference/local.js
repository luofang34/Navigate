import {assetUrl} from '../asset-url.js';
import {downloadFiles,read,get,put} from '../storage.js';
export class LocalMatcher {
  constructor(options={}){this.options=options}
  async initialize(progress){let manifest;try{const response=await fetch(assetUrl('models/manifest.json'));if(!response.ok)throw Error('Browser models are not prepared');manifest=await response.json();await put('state','public-xfeat-models-v1',manifest)}catch(e){manifest=await get('state','public-xfeat-models-v1');if(!manifest)throw e}
    if(!manifest.xfeat||Object.keys(manifest).length!==1)throw Error('The public demo requires the XFeat model manifest');const files=Object.values(manifest);await downloadFiles(files,(n,total)=>progress(`Matching model ${(n/1048576).toFixed(1)} / ${(total/1048576).toFixed(1)} MB`));
    const models={};for(const [name,f] of Object.entries(manifest))models[name]=new Uint8Array(await read(`pilotage://chunks/${f.sha256}.bin`,0,f.size));
    progress('Loading image matcher…');const {XFeatMatcher}=await import('./xfeat.js');
    progress('Preparing image matcher…');this.matcher=await XFeatMatcher.create(models);this.identity=this.matcher.identity+'/'+files.map(f=>f.sha256).join('/');
  }
  async matchImages(reference,query,keys={}){const limit=keys.stage==='refinement'?Math.min(4096,Math.max(2048,(this.options.keypoints??1024)*3)):2048;const q=await this.matcher.features(query,keys.query,limit);return {pairs:q.count<6?[]:await this.matcher.pairs(await this.matcher.features(reference,keys.reference,limit),q),backend_identity:this.identity}}
  async retrievePairs(references,queries,limit,progress){
    const ranked=[],features=[];
    for(const [i,reference] of references.entries()){progress(`Preparing reference images ${i+1}/${references.length}`);features.push(await this.matcher.features(reference.image,reference.key))}
    for(const [q,query] of queries.entries()){
      progress(`Searching camera orientations ${q+1}/${queries.length}`);const f=await this.matcher.features(query.image,query.key);this.matcher.gpu.phase('retrieval');const scores=[];for(let start=0;start<features.length;start+=160)scores.push(...await this.matcher.retrieval.rankMany(features.slice(start,start+160),f));
      for(const [r,score] of scores.entries())ranked.push({reference_index:r,query_index:q,score});
    }
    ranked.sort((a,b)=>b.score-a.score);return ranked.slice(0,limit).map(({reference_index,query_index})=>({reference_index,query_index}));
  }
  diagnostics(){return {counter_scope:"worker lifetime; feature outputs can be cached",...this.matcher.metrics,retrieval_gpu_dispatches:this.matcher.retrieval.dispatches,execution:'WebGPU kernels with WASM fallback for unsupported model operators'}}
  close(){return this.matcher.close()}
}
