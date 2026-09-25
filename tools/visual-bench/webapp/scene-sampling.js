// Extra samples test a disconnected image interval. They do not accept a map pose.
export function sceneSamplingPlans(reconstruction,{maxGroups=2,maxExtraFrames=24,subdivisions=4}={}){
 for(const n of [maxGroups,maxExtraFrames,subdivisions])if(!Number.isInteger(n)||n<1)throw Error('Invalid scene sampling budget');
 if(maxGroups>8||maxExtraFrames>64||subdivisions>8)throw Error('Scene sampling budget is too large');
 const plans=[],deferred=[],attempted=new Set(reconstruction.sampling_refinement?.attempts.map(a=>a.source_group_sha256)??[]);let remaining=maxExtraFrames;
 for(const [groupIndex,group] of reconstruction.groups.entries()){
  if(attempted.has(group.source_group_sha256))continue;
  const sources=group.observations,coverage=group.candidates.map(c=>new Set(c.cameras.map(c=>c.observation_sha256))),gaps=[];
  for(let i=1;i<sources.length;i++){
   const a=sources[i-1].observation_sha256,b=sources[i].observation_sha256;
   if(coverage.some(c=>c.has(a))&&coverage.some(c=>c.has(b))&&!coverage.some(c=>c.has(a)&&c.has(b)))gaps.push(i);
  }
  const boundary=gaps.length?null:boundaryObservations(reconstruction.groups,groupIndex);
  if(!gaps.length&&!boundary)continue;
  const times=new Set(),originals=new Set(sources.map(s=>s.requested_time_s));
  for(const gap of gaps)for(let i=Math.max(1,gap-1);i<=Math.min(sources.length-1,gap+1);i++){
   const a=sources[i-1].requested_time_s,b=sources[i].requested_time_s;
   if(!Number.isFinite(a)||!Number.isFinite(b)||b<=a)continue;
   for(let j=1;j<subdivisions;j++){const t=Math.round((a+(b-a)*j/subdivisions)*1e9)/1e9;if(!originals.has(t))times.add(t)}
  }
  const selected=[...times].sort((a,b)=>a-b),observations=boundary??sources,budget=Math.min(remaining,129-observations.length);
  if(plans.length>=maxGroups||(!selected.length&&!boundary)||selected.length>budget){deferred.push(group.source_group_sha256);continue}
  plans.push({source_group_sha256:group.source_group_sha256,observations,times:selected,scope:boundary?'overlapping_group_retry':'denser_interval_retry'});remaining-=selected.length;
 }
 return {plans,deferred_group_sha256:deferred,max_extra_frames:maxExtraFrames,geographic_acceptance:false};
}


function boundaryObservations(groups,index){
 const group=groups[index],previous=groups[index-1];
 if(!previous||previous.source_group_sha256===group.source_group_sha256||!group.candidates.length||!group.candidates.every(c=>c.parent_scene_sha256===null))return null;
 const supported=new Set(previous.candidates.flatMap(c=>c.cameras.map(c=>c.observation_sha256)));
 const shared=group.observations.filter(o=>supported.has(o.observation_sha256));
 if(shared.length<3)return null;
 const own=new Set(group.observations.map(o=>o.observation_sha256)),next=groups[index+1];
 const neighbors=[previous,group];
 if(next&&next.observations.filter(o=>own.has(o.observation_sha256)).length>=3)neighbors.push(next);
 const observations=[...new Map(neighbors.flatMap(g=>g.observations).map(o=>[o.observation_sha256,o])).values()]
  .sort((a,b)=>a.capture_time_ns-b.capture_time_ns).slice(-129);
 return observations.filter(o=>supported.has(o.observation_sha256)).length>=3?observations:null;
}

export async function refineSceneSampling(reconstruction,{pipeline,frames,prior,image,decode,save,commit=async()=>{},progress=()=>{},options}){
 const schedule=sceneSamplingPlans(reconstruction,options),attempts=[...(reconstruction.sampling_refinement?.attempts??[])];let result=reconstruction;
 const used=new Set(frames.map(f=>f.sequence));let next=frames.reduce((n,f)=>Math.max(n,f.sequence),0);
 const sequence=()=>{do{next=(next+1)>>>0}while(used.has(next));used.add(next);return next};
 for(const plan of schedule.plans){
  const originals=new Map(frames.map(f=>[f.observation_sha256,f]));
  const ordered=plan.observations.map(source=>{const frame=originals.get(source.observation_sha256);if(!frame||frame.sequence!==source.sequence||frame.capture_time_ns!==source.capture_time_ns)throw Error('Scene sampling source identity changed');return {frame,time:frame.requested_time_s}});
  ordered.push(...plan.times.map(time=>({time,frame:frames.find(f=>Math.abs(f.requested_time_s-time)<1e-9)})));ordered.sort((a,b)=>a.time-b.time);
  const captures=new Set(frames.map(s=>s.capture_time_ns)),added=[],skipped=[];
  await pipeline.beginSequence({maxFrames:129});
  for(const [index,item] of ordered.entries()){
   progress(`Refining image interval · sample ${index+1}/${ordered.length}…`);
   const pixels=item.frame?await image(item.frame):await decode(item.time),capture=Math.round(pixels.time*1e9);
   if(!item.frame&&captures.has(capture)){skipped.push({requested_time_s:item.time,capture_time_ns:capture,reason:'same decoded source frame'});continue}
   const frame=await pipeline.observeScene(pixels,prior,item.frame?.sequence??sequence(),item.frame?.observation_sha256,text=>progress(`Refining image interval · sample ${index+1}/${ordered.length} · ${text}`));
   if(item.frame&&(frame.observation_sha256!==item.frame.observation_sha256||frame.capture_time_ns!==item.frame.capture_time_ns))throw Error('Scene sampling changed an existing observation');
   captures.add(capture);
   if(!item.frame){frame.input_name=ordered.find(i=>i.frame)?.frame.input_name;added.push({frame,blob:pixels.blob})}
  }
  const records=await pipeline.finishSequence();
  await save(added);for(const {frame} of added)frames.push(frame);frames.sort((a,b)=>a.capture_time_ns-b.capture_time_ns);await commit();
  const extension=await pipeline.refineScenes(records,result,progress);
  result={...result,groups:[...result.groups,...extension.groups]};
  attempts.push({source_group_sha256:plan.source_group_sha256,image_groups:records,added_observation_sha256:added.map(v=>v.frame.observation_sha256),skipped,geographic_acceptance:false});
 }
 return {...result,sampling_refinement:{...schedule,plans:undefined,attempts,evidence_correlation:'unknown; reused frames retain their identities; new adjacent samples share scene and calibration errors'}};
}
