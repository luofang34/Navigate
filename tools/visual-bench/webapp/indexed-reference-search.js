import {localPosition} from './geography.js';
export class IndexedReferenceSearch {
 constructor(retriever,entries,scaleOrder){
  this.retriever=retriever;this.entries=new Map();this.scales=scaleOrder;
  if(!Array.isArray(scaleOrder)||!scaleOrder.length||new Set(scaleOrder).size!==scaleOrder.length||scaleOrder.some(v=>!Number.isFinite(v)||v<=0))throw Error('Invalid reference scale order');
  for(const entry of entries){if(!Number.isSafeInteger(entry.id)||entry.id<0||this.entries.has(entry.id)||!Number.isFinite(entry.width_m)||entry.width_m<=0||!Array.isArray(entry.center_enu_m)||entry.center_enu_m.length!==2||entry.center_enu_m.some(v=>!Number.isFinite(v)))throw Error('Invalid indexed reference entry');this.entries.set(entry.id,entry)}
 }
 async propose(image,prior,camera,reference,progress,{limit=8,orientations}={}){
  if(this.retriever.catalog.map_manifest_sha256!==reference.pack.pack_id||this.retriever.catalog.map_release!==reference.pack.release_id)throw Error('Reference search uses different map data');
  if(!Number.isInteger(limit)||limit<1||limit>8||!Array.isArray(orientations)||!orientations.length||orientations.length>8||orientations.some(q=>!Array.isArray(q)||q.length!==4||q.some(v=>!Number.isFinite(v))||Math.abs(Math.hypot(...q)-1)>1e-6))throw Error('Invalid indexed search budget or orientations');
  if(![prior.latitude,prior.longitude,prior.radius_m,camera.width,camera.fx].every(Number.isFinite)||prior.radius_m<=0||Math.abs(prior.latitude)>=90||Math.abs(prior.longitude)>180||camera.width<=0||camera.fx<=0)throw Error('Invalid reference search prior or calibration');
  const center=localPosition(reference.pack,prior.latitude,prior.longitude,0),groups=this.scales.map(scale=>[...this.entries.values()].filter(e=>Math.round(e.width_m)===scale&&Math.hypot(e.center_enu_m[0]-center[0],e.center_enu_m[1]-center[1])<=prior.radius_m+e.width_m/Math.sqrt(2)).map(e=>e.id));
  const source=image.original??image,{width,height}=source;
  if(!Number.isInteger(width)||!Number.isInteger(height)||Math.min(width,height)<1||Math.max(width,height)>8192||(source.rgb&&(!(source.rgb instanceof Uint8Array)||source.rgb.length!==width*height*3))||(source.gray&&(!(source.gray instanceof Uint8Array)||source.gray.length!==width*height))||(!source.rgb&&!source.gray))throw Error('Invalid retrieval observation pixels');
  const grayRgb=new Uint8Array(width*height*3);for(let i=0;i<width*height;i++){const value=source.gray?.[i]??((77*source.rgb[i*3]+150*source.rgb[i*3+1]+29*source.rgb[i*3+2])>>>8);grayRgb.fill(value,i*3,i*3+3)}
  const inputs=[grayRgb,...(source.rgb?[source.rgb]:[])],rankings=[];
  for(const rgb of inputs){const ranks=[];for(const eligible of groups){progress('Finding reference areas…');const ranked=await this.retriever.rank({rgb,width,height},eligible,limit);if(!Array.isArray(ranked)||ranked.length>limit||new Set(ranked).size!==ranked.length||ranked.some(id=>!eligible.includes(id)))throw Error('Retriever returned ineligible reference IDs');ranks.push(ranked)}rankings.push(ranks)}
  const ids=[];for(let rank=0;rank<limit*groups.length&&ids.length<limit*2;rank++){const group=rank%groups.length,within=Math.floor(rank/groups.length);for(const ranks of rankings){const id=ranks[group][within];if(id!==undefined&&!ids.includes(id)&&ids.length<limit*2)ids.push(id)}}
  const candidates=[],unsupported=[];
  for(const id of ids){const entry=this.entries.get(id);if(!entry)throw Error('Retriever returned an unknown reference ID');const [east,north]=entry.center_enu_m,[lat0,lon0]=reference.pack.anchor_lat_lon.map(v=>v*Math.PI/180),scale=6371008.8*Math.cos(lat0),lat=Math.atan(Math.sinh(Math.asinh(Math.tan(lat0))+north/scale))*180/Math.PI,lon=(lon0+east/scale)*180/Math.PI;
   let elevation;try{elevation=reference.elevation(lat,lon)}catch(error){unsupported.push({reference_id:id,reason:String(error)});continue}
   const position=[east,north,elevation+entry.width_m*camera.fx/camera.width];for(const q of orientations)candidates.push({position_enu_m:position.slice(),eye_to_enu_xyzw:q.slice()});
  }
  return {candidates,input_image:{width,height,scope:image.original?'original decoded pixels':'working-resolution pixels'},reference_ids:ids,unsupported_references:unsupported,backend_identity:this.retriever.identity,stage:'retrieval_only',scope:'Indexed reference crops initialize rendering; pose and geographic acceptance remain unverified'};
 }
 diagnostics(){return this.retriever.diagnostics?.()}
 close(){return this.retriever.close()}
}
