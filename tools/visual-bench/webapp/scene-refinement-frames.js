// Camera selection controls a bounded solve and does not rank geographic truth.
export function refinementFrames(observations,required,{limit=112}={}){
 if(!Number.isInteger(limit)||limit<2||limit>129)throw Error('Invalid scene camera budget');
 const ids=new Set();for(const o of observations){if(ids.has(o.observation_sha256)||!Number.isFinite(o.capture_time_ns))throw Error('Invalid or repeated scene observation');ids.add(o.observation_sha256)}
 const ordered=[...observations].sort((a,b)=>a.capture_time_ns-b.capture_time_ns||a.sequence-b.sequence||a.observation_sha256.localeCompare(b.observation_sha256));
 if(ordered.length<2)throw Error('Scene refinement needs two source observations');
 const byId=new Map(ordered.map((o,index)=>[o.observation_sha256,index])),chosen=new Set([0,ordered.length-1]);
 for(const id of required){const index=byId.get(id);if(index===undefined)throw Error('Required map anchor is absent from the scene path');chosen.add(index)}
 if(chosen.size>limit)throw Error('Required scene cameras exceed the work budget');
 const duration=Math.max(1,ordered.at(-1).capture_time_ns-ordered[0].capture_time_ns);
 while(chosen.size<Math.min(limit,ordered.length)){
  let best=-1,score=-1;
  for(let i=0;i<ordered.length;i++){
   if(chosen.has(i))continue;
   let time=Infinity,index=Infinity;
   for(const j of chosen){time=Math.min(time,Math.abs(ordered[i].capture_time_ns-ordered[j].capture_time_ns));index=Math.min(index,Math.abs(i-j))}
   const distance=time/duration+index/ordered.length*1e-9;
   if(distance>score){score=distance;best=i}
  }
  if(best<0)break;chosen.add(best);
 }
 return [...chosen].sort((a,b)=>a-b).map(i=>ordered[i]);
}

export function addUnresolvedFrames(selected,observations,unresolved,{limit=129}={}){
 if(!Number.isInteger(limit)||limit<selected.length||limit>129)throw Error('Invalid camera retry budget');
 const known=new Set(observations.map(o=>o.observation_sha256)),ids=new Set(selected.map(o=>o.observation_sha256));
 for(const id of unresolved)if(!known.has(id))throw Error('Unresolved camera has an unknown source');
 const missing=observations.filter(o=>unresolved.includes(o.observation_sha256)&&!ids.has(o.observation_sha256));
 if(!missing.length||ids.size===limit)return selected;
 // Retry frames spread work across gaps. No camera pose is interpolated.
 const available=limit-ids.size;
 const extra=missing.length<=available?missing:refinementFrames(missing,[],{limit:Math.max(2,available)}).slice(0,available);
 return [...selected,...extra].sort((a,b)=>a.capture_time_ns-b.capture_time_ns||a.sequence-b.sequence||a.observation_sha256.localeCompare(b.observation_sha256));
}

export function connectRefinementFrames(selected,observations,points,{limit=129}={}){
 if(!Number.isInteger(limit)||limit<selected.length||limit>129)throw Error('Invalid camera connection budget');
 const indices=new Map(observations.map((o,i)=>[o.observation_sha256,i]));
 const source=selected.map(o=>indices.get(o.observation_sha256));
 if(source.some((index,i)=>index===undefined||(i>0&&index<=source[i-1])))throw Error('Connection cameras must be distinct ordered source observations');
 const parents=selected.map((_,i)=>i),root=index=>{while(parents[index]!==index)index=parents[index];return index};
 for(const point of points){
  const cameras=point.observations.map(o=>o.camera_index);
  if(cameras.some(i=>!Number.isInteger(i)||i<0||i>=selected.length))throw Error('Connection point has an unknown camera');
  for(const i of cameras.slice(1))parents[root(i)]=root(cameras[0]);
 }
 const ids=new Set(selected.map(o=>o.observation_sha256)),disconnected=selected.filter((_,i)=>root(i)!==root(0)).map(o=>o.observation_sha256);
 // Bisect real source intervals across disconnected image-support components.
 for(let i=1;i<selected.length&&ids.size<limit;i++)if(root(i)!==root(i-1)&&source[i]-source[i-1]>1)ids.add(observations[Math.floor((source[i]+source[i-1])/2)].observation_sha256);
 return {frames:observations.filter(o=>ids.has(o.observation_sha256)),disconnected_observation_sha256:disconnected};
}
