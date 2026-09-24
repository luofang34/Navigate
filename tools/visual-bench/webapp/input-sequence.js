export function selectedInputs(files){
 const values=Array.from(files);
 if(values.length>120)throw Error('Select at most 120 images.');
 if(values.length>1&&values.some(f=>!f.type.startsWith('image/')))throw Error('Select one video or a sequence of images.');
 if(values.some(f=>f.type.startsWith('image/')&&f.size>256*1024*1024))throw Error('Each image must be at most 256 MB.');
 return values.sort((a,b)=>a.name.localeCompare(b.name,'en',{numeric:true}));
}
export function sequenceTrack(frames){
 const features=[],gaps=[];let segment=[],context;
 const end=()=>{if(segment.length>1)features.push({type:'Feature',geometry:{type:'LineString',coordinates:segment.map(p=>p.coordinate)},properties:{kind:'illustrative hypothesis path',observations:segment.map(p=>p.observation),candidate_ids:segment.map(p=>p.id),map_manifest_sha256:context,accuracy:'not independently measured',correlation:'unknown; shared map evidence',selection:'first supported hypothesis per frame; map fixes and conditional tracking remain distinct in points'}});segment=[]};
 for(const [index,frame] of frames.entries()){
  const candidates=(frame.candidate_hypotheses??[]).filter(h=>(h.accepted||h.tracking_supported)&&Number.isFinite(h.longitude_deg)&&Number.isFinite(h.latitude_deg));
  const first=candidates[0];
  if(!first){end();gaps.push({index,observation_sha256:frame.observation_sha256,reason:frame.reason});continue}
  if(!first.map_manifest_sha256||context!==first.map_manifest_sha256)end();context=first.map_manifest_sha256;
  for(const candidate of candidates)features.push({type:'Feature',geometry:{type:'Point',coordinates:[candidate.longitude_deg,candidate.latitude_deg]},properties:{...candidate,frame_index:index,input_name:frame.input_name,decision:frame.decision,timing_scope:frame.timing_scope,capture_time_ns:frame.capture_time_ns,accuracy:'not independently measured'}});
  segment.push({coordinate:[first.longitude_deg,first.latitude_deg],observation:frame.observation_sha256,id:first.candidate_id});
 }
 end();return {type:'FeatureCollection',features,gaps,sequence_scope:'Input order. Still-image flight times are unknown. Lines illustrate one possible path; they are not a fused navigation track.'};
}
