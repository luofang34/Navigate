// Frame digests include pixels and calibration. Only exact tracking pairs can be reused.
export class TrackingPairCache {
 constructor({maxBytes=8*1024*1024,maxEntries=128}={}){this.maxBytes=maxBytes;this.maxEntries=maxEntries;this.entries=new Map();this.bytes=0;this.hits=0;this.misses=0}
 key({stage,reference,query}={}){return stage==='tracking'&&[reference,query].every(v=>typeof v==='string'&&/^[a-f0-9]{64}\/query$/.test(v))?reference+'|'+query:null}
 get(keys){
  const key=this.key(keys);if(!key)return undefined;
  const packed=this.entries.get(key);if(!packed){this.misses++;return undefined}
  this.entries.delete(key);this.entries.set(key,packed);this.hits++;
  const pairs=[];for(let i=0;i<packed.length;i+=4)pairs.push({reference:[packed[i],packed[i+1]],query:[packed[i+2],packed[i+3]]});return pairs;
 }
 put(keys,pairs){
  const key=this.key(keys);if(!key)return;
  const packed=new Float64Array(pairs.length*4);for(const [i,p] of pairs.entries())packed.set([...p.reference,...p.query],i*4);
  if(packed.byteLength>this.maxBytes||this.maxEntries<1)return;
  const prior=this.entries.get(key);if(prior){this.bytes-=prior.byteLength;this.entries.delete(key)}
  while(this.entries.size&&(this.bytes+packed.byteLength>this.maxBytes||this.entries.size>=this.maxEntries)){const oldest=this.entries.keys().next().value;this.bytes-=this.entries.get(oldest).byteLength;this.entries.delete(oldest)}
  this.entries.set(key,packed);this.bytes+=packed.byteLength;
 }
 clear(){this.entries.clear();this.bytes=0}
}
