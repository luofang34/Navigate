import {downloadFiles,read,get,put} from '../storage.js';
export class LocalMatcher {
  constructor(options={}){this.options=options}
  async initialize(progress){let manifest;try{const response=await fetch('/models/manifest.json');if(!response.ok)throw Error('Browser models are not prepared');manifest=await response.json();await put('state','models',manifest)}catch(e){manifest=await get('state','models');if(!manifest)throw e}
    const files=Object.values(manifest);await downloadFiles(files,(n,total)=>progress(`Browser model data ${(n/1048576).toFixed(1)} / ${(total/1048576).toFixed(1)} MB`));
    const models={};for(const [name,f] of Object.entries(manifest))models[name]=new Uint8Array(await read(`pilotage://chunks/${f.sha256}.bin`,0,f.size));
    progress('Loading browser inference runtime…');const {LearnedMatcher}=await import('./learned.js');
    progress('Initializing browser WebGPU / WASM models…');this.matcher=await LearnedMatcher.create(models);this.identity=this.matcher.identity+'/'+files.map(f=>f.sha256).join('/');
  }
  async matchImages(reference,query,keys={}){const limit=keys.stage==='refinement'?(this.options.keypoints??512):512;const q=await this.matcher.features(query,keys.query,limit);return {pairs:q.count<6?[]:await this.matcher.pairs(await this.matcher.features(reference,keys.reference,limit),q),backend_identity:this.identity}}
  async retrievePairs(references,queries,limit,progress){
    const ranked=[],features=[];
    for(const [i,reference] of references.entries()){progress(`GPU reference features ${i+1}/${references.length}`);features.push(await this.matcher.features(reference.image,reference.key))}
    for(const [q,query] of queries.entries()){
      progress(`GPU retrieval orientation ${q+1}/${queries.length}`);const f=await this.matcher.features(query.image,query.key);this.matcher.gpu.phase('retrieval');const scores=await this.matcher.retrieval.rankMany(features,f);
      for(const [r,score] of scores.entries())ranked.push({reference_index:r,query_index:q,score});
    }
    ranked.sort((a,b)=>b.score-a.score);return ranked.slice(0,limit).map(({reference_index,query_index})=>({reference_index,query_index}));
  }
  diagnostics(){return {counter_scope:"worker lifetime; feature outputs can be cached",...this.matcher.metrics,retrieval_gpu_dispatches:this.matcher.retrieval.dispatches,execution:'WebGPU kernels with WASM fallback for unsupported model operators'}}
  close(){return this.matcher.close()}
}
