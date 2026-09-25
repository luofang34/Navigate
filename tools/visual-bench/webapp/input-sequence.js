import {branchKey} from './hypotheses.js';

export function selectedInputs(files){
 const values=Array.from(files);
 if(values.length>120)throw Error('Select at most 120 images.');
 if(values.length>1&&values.some(f=>!f.type.startsWith('image/')))throw Error('Select one video or a sequence of images.');
 if(values.some(f=>f.type.startsWith('image/')&&f.size>256*1024*1024))throw Error('Each image must be at most 256 MB.');
 return values.sort((a,b)=>a.name.localeCompare(b.name,'en',{numeric:true}));
}
export function sequenceTrack(frames){
 const features=[],gaps=[],segments=new Map();
 const end=key=>{
  const segment=segments.get(key);if(!segment)return;
  if(segment.samples.length>1)features.push({type:'Feature',geometry:{type:'LineString',coordinates:segment.samples.map(p=>p.coordinate)},properties:{kind:'illustrative hypothesis path',track_id:key,observations:segment.samples.map(p=>p.observation),candidate_ids:segment.samples.map(p=>p.id),map_manifest_sha256:segment.map,accuracy:'not independently measured',correlation:'unknown; shared map evidence',selection:'one explicit hypothesis identity per line; no branch selection or fusion'}});
  segments.delete(key);
 };
 for(const [index,frame] of frames.entries()){
  const candidates=(frame.candidate_hypotheses??[]).filter(h=>(h.accepted||h.tracking_supported||h.scene_supported)&&Number.isFinite(h.longitude_deg)&&Number.isFinite(h.latitude_deg));
  if(!candidates.length)gaps.push({index,observation_sha256:frame.observation_sha256,reason:frame.reason});
  const present=new Set(candidates.map(h=>branchKey(frame,h)));
  for(const key of segments.keys())if(!present.has(key))end(key);
  for(const candidate of candidates){
   features.push({type:'Feature',geometry:{type:'Point',coordinates:[candidate.longitude_deg,candidate.latitude_deg]},properties:{...candidate,observation_sha256:frame.observation_sha256,frame_index:index,input_name:frame.input_name,decision:frame.decision,timing_scope:frame.timing_scope,capture_time_ns:frame.capture_time_ns,accuracy:'not independently measured'}});
   const key=branchKey(frame,candidate),previous=segments.get(key),last=previous?.samples.at(-1);
   const unordered=last&&!(frame.timing_scope==='still image'&&last.scope==='still image')&&(!Number.isFinite(frame.capture_time_ns)||!Number.isFinite(last.time)||frame.capture_time_ns<=last.time);
   if(!frame.observation_sha256||!candidate.map_manifest_sha256||previous?.map!==candidate.map_manifest_sha256||candidate.continuity_break||last?.observation===frame.observation_sha256||unordered)end(key);
   if(!frame.observation_sha256||!candidate.map_manifest_sha256)continue;
   if(!segments.has(key))segments.set(key,{map:candidate.map_manifest_sha256,samples:[]});
   segments.get(key).samples.push({coordinate:[candidate.longitude_deg,candidate.latitude_deg],observation:frame.observation_sha256,id:candidate.candidate_id,time:frame.capture_time_ns,scope:frame.timing_scope});
  }
 }
 for(const key of segments.keys())end(key);
 return {type:'FeatureCollection',features,gaps,sequence_scope:'Input order. Still-image flight times are unknown. Lines retain separate hypothesis identities; they are not a fused navigation track.'};
}

export function videoTimes(duration,start,{mode='whole',period=.2,maxFrames=120}={}){
 if(!Number.isFinite(duration)||duration<=0||!Number.isFinite(start)||start<0||start>=duration)throw Error('Invalid video time');
 if(mode==='frame')return [start];
 if(!['whole','sequence'].includes(mode)||!Number.isFinite(period)||period<.1)throw Error('Invalid video sampling interval');
 if(mode==='sequence'&&(!Number.isInteger(maxFrames)||maxFrames<1||maxFrames>120))throw Error('Invalid frame limit');
 const first=mode==='whole'?0:start,count=Math.max(1,Math.ceil((duration-.02-first)/period));
 if(mode==='whole'&&count>5000)throw Error('This interval selects more than 5000 frames. Increase the interval or select a shorter range.');
 return Array.from({length:mode==='whole'?count:Math.min(count,maxFrames)},(_,i)=>first+i*period);
}
