import {assetUrl} from '../asset-url.js';
import {downloadFiles,read,get,put} from '../storage.js';
export class BrowserMatcher {
  constructor(){this.worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});this.pending=new Map();this.next=0;
    this.worker.onmessage=({data})=>{const p=this.pending.get(data.id);if(!p)return;this.pending.delete(data.id);data.error?p.reject(Error(data.error)):p.resolve(data.value)};
    this.worker.onerror=event=>{for(const p of this.pending.values())p.reject(Error(event.message));this.pending.clear()};
  }
  call(method,...args){return new Promise((resolve,reject)=>{const id=this.next=(this.next+1)>>>0;this.pending.set(id,{resolve,reject});this.worker.postMessage({id,method,args})})}
  async initialize(progress){let manifest;try{const response=await fetch(assetUrl('models/manifest.json'));if(!response.ok)throw Error('Browser models are not prepared');manifest=await response.json();await put('state','models',manifest)}catch(e){manifest=await get('state','models');if(!manifest)throw e}
    const files=Object.values(manifest);await downloadFiles(files,(n,total)=>progress(`Browser model data ${(n/1048576).toFixed(1)} / ${(total/1048576).toFixed(1)} MB`));
    const models={};for(const [name,f] of Object.entries(manifest))models[name]=new Uint8Array(await read(`pilotage://chunks/${f.sha256}.bin`,0,f.size));
    progress('Initializing browser WebGPU / WASM models…');const value=await this.call('initialize',models);this.identity=value.identity+'/'+files.map(f=>f.sha256).join('/');
  }
  async matchImages(reference,query,keys={}){const pixels=({gray,width,height})=>({gray,width,height});return {pairs:await this.call('match',pixels(reference),pixels(query),keys),backend_identity:this.identity}}
  close(){this.worker.terminate();for(const p of this.pending.values())p.reject(Error('Matcher closed'));this.pending.clear()}
}
