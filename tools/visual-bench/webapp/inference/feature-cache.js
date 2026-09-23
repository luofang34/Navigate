export class FeatureCache {
  constructor(maxBytes,maxEntries){
    if(!Number.isSafeInteger(maxBytes)||maxBytes<0||!Number.isSafeInteger(maxEntries)||maxEntries<0)throw Error('Invalid feature cache budget');
    this.maxBytes=maxBytes;this.maxEntries=maxEntries;this.bytes=0;this.entries=new Map();this.hits=0;this.misses=0;
  }
  get_cached(key){
    if(!key)return undefined;
    const entry=this.entries.get(key);
    if(!entry){this.misses++;return undefined}
    this.entries.delete(key);this.entries.set(key,entry);this.hits++;return entry.features;
  }
  insert(key,features){
    if(!key)return;
    const buffers=new Set([features.pixels,features.modelPixels,features.scores,features.descriptors].map(value=>value.buffer));
    const bytes=[...buffers].reduce((sum,buffer)=>sum+buffer.byteLength,0);
    const previous=this.entries.get(key);if(previous){this.bytes-=previous.bytes;this.entries.delete(key)}
    if(bytes>this.maxBytes||!this.maxEntries)return;
    while(this.entries.size>=this.maxEntries||this.bytes+bytes>this.maxBytes){const [oldest,entry]=this.entries.entries().next().value;this.entries.delete(oldest);this.bytes-=entry.bytes}
    this.entries.set(key,{features,bytes});this.bytes+=bytes;
  }
  clear(){this.entries.clear();this.bytes=0}
}
