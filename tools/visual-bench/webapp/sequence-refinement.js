import {hypotheses} from './hypotheses.js';

export async function refineSequenceGaps(frames,{image,track,progress=()=>{}}){
 let recovered=0;
 for(let index=frames.length-2;index>=0;index--){
  const original=frames[index],reference=frames[index+1];
  if(hypotheses(original).length||!hypotheses(reference).length)continue;
  if(!original.observation_sha256||!reference.observation_sha256)continue;
  if(!(reference.capture_time_ns>original.capture_time_ns)||reference.observation_sha256===original.observation_sha256)continue;
  progress(`Refining gap · frame ${index+1}/${frames.length}`);
  const result=await track({report:reference,image:await image(index+1)},await image(index),original.sequence,progress);
  if(!result||!hypotheses(result).length)continue;
  if(result.observation_sha256!==original.observation_sha256||result.sequence!==original.sequence||result.capture_time_ns!==original.capture_time_ns)throw Error('Gap refinement changed the observation identity');
  if(result.accepted||result.decision!=='relative_tracking'||hypotheses(result).some(h=>h.accepted))throw Error('Gap refinement must remain conditional tracking');
  frames[index]={...original,...result,sequence_refinement:{direction:'from later observation',reference_observation_sha256:reference.observation_sha256,previous_attempt:original}};
  recovered++;
 }
 return recovered;
}

function sameObservation(original,result){
 if(!original.observation_sha256||result.observation_sha256!==original.observation_sha256||result.sequence!==original.sequence||result.capture_time_ns!==original.capture_time_ns)throw Error('Sequence refinement changed the observation identity');
}
function mergeRefinement(original,result,reference,tracking,{alternatives=[],motion_check}={}){
 sameObservation(original,result);
 const used=new Set(original.candidate_hypotheses.map(h=>h.candidate_id));let next=0;
 const primary=hypotheses(result),mapped=[...primary,...alternatives].map(h=>{
  while(used.has(next))next=(next+1)>>>0;const candidate_id=next;used.add(candidate_id);next=(next+1)>>>0;
  const anchor=h.accepted?{observation_sha256:original.observation_sha256,candidate_id,map_manifest_sha256:h.map_manifest_sha256}:h.tracking_anchor;
  return {...h,candidate_id,refinement_candidate_id:h.candidate_id,tracking_anchor:anchor,track_id:JSON.stringify(anchor),continuity_break:h.accepted?false:h.continuity_break};
 });
 const refined=mapped.slice(0,primary.length),candidate_hypotheses=[...mapped,...original.candidate_hypotheses];
 const merged={...original,...result,accepted:false,decision:candidate_hypotheses.some(h=>h.accepted)?'unresolved':'relative_tracking',candidate_hypotheses,
  sequence_refinement:{method:'backward_map',reference_observation_sha256:reference.observation_sha256,previous_attempt:original,tracking_attempt:tracking,...(motion_check?{motion_check}:{})},
  evidence_correlation:'unknown; forward and backward estimates reuse observations and map data; no fusion or independent confidence'};
 return {merged,reference:{...merged,candidate_hypotheses:refined}};
}
export function motionCompatibleCandidates(checks,relativeCount,minimumRetention=.5){
 if(!Number.isFinite(minimumRetention)||minimumRetention<=0||minimumRetention>1)throw Error('Invalid motion support retention');
 const baseline=new Map();
 for(const c of checks)if(c.consistent&&c.candidate_id<relativeCount&&Number.isFinite(c.inliers)&&c.inliers>0)baseline.set(c.reference_candidate_id,Math.max(baseline.get(c.reference_candidate_id)??0,c.inliers));
 return new Set(checks.filter(c=>c.consistent&&c.candidate_id>=relativeCount&&Number.isFinite(c.inliers)&&baseline.has(c.reference_candidate_id)&&c.inliers>=minimumRetention*baseline.get(c.reference_candidate_id)).map(c=>c.candidate_id-relativeCount));
}
async function connectMapCandidates(reference,original,current,relative,mapped,checkMotion){
 const tracked=hypotheses(relative),candidates=[...tracked,...mapped].map((h,candidate_id)=>({...h,candidate_id}));
 const check=await checkMotion(reference,original,current,candidates);
 if(check.observation_sha256!==original.observation_sha256||check.reference_observation_sha256!==reference.report.observation_sha256)throw Error('Motion association changed observation identity');
 const connected=motionCompatibleCandidates(check.checks,tracked.length);
 const motion_check={...check,minimum_support_retention:.5,policy:'continue map candidates with comparable support on the same camera pairs and reference; retain conflicting alternatives',candidate_sources:candidates.map((h,i)=>({check_candidate_id:i,source:i<tracked.length?'relative':'map',source_candidate_id:(i<tracked.length?tracked[i]:mapped[i-tracked.length]).candidate_id}))};
 return {connected:mapped.filter((_,i)=>connected.has(i)),alternatives:mapped.filter((_,i)=>!connected.has(i)),motion_check};
}
export async function refineSequenceBackward(frames,{image,track,verify,checkMotion,intervalSeconds=5,progress=()=>{}}){
 if(!Number.isFinite(intervalSeconds)||intervalSeconds<=0)throw Error('Invalid sequence map-check interval');
 const originals=frames.map(f=>f.sequence_refinement?.method==='backward_map'?f.sequence_refinement.previous_attempt:f);
 const counts={frames_updated:0,map_accepted:0,recovered:0};let reference=null,lastMapTime=null;
 for(let index=originals.length-1;index>=0;index--){
  const original=originals[index];
  if(reference&&original.observation_sha256&&reference.report.capture_time_ns>original.capture_time_ns&&reference.report.observation_sha256!==original.observation_sha256){
   progress(`Checking sequence · frame ${index+1}/${frames.length}`);
   const current=await image(index),relative=await track(reference,current,original.sequence,progress);
   if(relative)sameObservation(original,relative);
   const supported=relative&&hypotheses(relative).length;let result=relative,checked,association;
   if(!supported||original.candidate_hypotheses.some(h=>h.accepted)||(lastMapTime-original.capture_time_ns)>=intervalSeconds*1e9){
    checked=await verify(original,current,hypotheses(supported?relative:reference.report),progress);sameObservation(original,checked);
    const mapped=checked.candidate_hypotheses.filter(h=>h.accepted);
    if(mapped.length){
     counts.map_accepted++;lastMapTime=original.capture_time_ns;
     if(supported&&typeof checkMotion==='function'){
      association=await connectMapCandidates(reference,original,current,relative,mapped,checkMotion);
      if(association.connected.length)result={...checked,candidate_hypotheses:association.connected};
     }else result={...checked,candidate_hypotheses:mapped};
    }
   }
   if(result&&hypotheses(result).length){
    const merged=mergeRefinement(original,result,reference.report,relative,association);
    frames[index]=merged.merged;reference={report:merged.reference,image:current};counts.frames_updated++;if(!hypotheses(original).length)counts.recovered++;continue;
   }
   frames[index]={...original,sequence_refinement:{method:'backward_map',reference_observation_sha256:reference.report.observation_sha256,previous_attempt:original,tracking_attempt:relative,map_attempt:checked}};
   reference=null;
  }
  const anchors=original.candidate_hypotheses.filter(h=>h.accepted);
  if(anchors.length&&original.observation_sha256){reference={report:{...original,candidate_hypotheses:anchors},image:await image(index)};lastMapTime=original.capture_time_ns}
 }
 return counts;
}
