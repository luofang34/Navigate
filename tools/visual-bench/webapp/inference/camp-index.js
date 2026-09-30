export class CampIndex {
 constructor(metadata,descriptors){
  const hash=value=>typeof value==='string'&&/^[a-f0-9]{64}$/i.test(value);
  if(!metadata.map_release||![metadata.map_manifest_sha256,metadata.catalog_manifest_sha256,metadata.model_sha256].every(hash)||!Array.isArray(metadata.ids)||!metadata.ids.length||!(descriptors instanceof Float32Array)||descriptors.length!==metadata.ids.length*1024)throw Error('Invalid CAMP index identity or shape');
  this.rows=new Map();this.descriptors=Float32Array.from(descriptors);
  for(const [row,id] of metadata.ids.entries()){
   if(!Number.isSafeInteger(id)||id<0||this.rows.has(id))throw Error('Invalid or repeated CAMP reference ID');
   validateCampDescriptor(descriptors.subarray(row*1024,(row+1)*1024));this.rows.set(id,row);
  }
  this.catalog=Object.freeze({map_release:metadata.map_release,map_manifest_sha256:metadata.map_manifest_sha256,catalog_manifest_sha256:metadata.catalog_manifest_sha256});
  this.modelSha256=metadata.model_sha256.toLowerCase();
 }
 eligible(ids,limit){
  if(!Array.isArray(ids)||!Number.isInteger(limit)||limit<0||limit>4096)throw Error('Invalid CAMP retrieval limit');
  const seen=new Set();return ids.map(id=>{if(!this.rows.has(id)||seen.has(id))throw Error('Unknown or repeated eligible CAMP reference ID');seen.add(id);return [id,this.rows.get(id)]});
 }
 rank(descriptor,rows,limit){
  validateCampDescriptor(descriptor);
  const ranked=rows.map(([id,row])=>{let score=0;for(let c=0;c<1024;c++)score+=this.descriptors[row*1024+c]*descriptor[c];return {id,score}});
  ranked.sort((a,b)=>b.score-a.score||a.id-b.id);return ranked.slice(0,limit).map(r=>r.id);
 }
}
export function validateCampDescriptor(values){
 if(values.length!==1024)throw Error('Invalid CAMP descriptor shape');
 let norm=0;for(const v of values){if(!Number.isFinite(v))throw Error('Nonfinite CAMP descriptor');norm+=v*v}
 if(Math.abs(norm-1)>.001)throw Error('CAMP descriptor is not normalized');
}
