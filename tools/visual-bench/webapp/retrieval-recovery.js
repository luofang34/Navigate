// Similarity seeds do not establish geographic acceptance or confidence.
export function similarityCandidates(records,propose,camera,position,radius){
 const ranked=[],cameraJson=JSON.stringify(camera);
 for(const threshold of [4,12]){
  const proposals=[];
  for(const {ground,source_index} of records){
   const p=JSON.parse(propose(cameraJson,JSON.stringify(ground),threshold));
   if(!p.retrieved||Math.hypot(...p.position_enu_m.map((v,i)=>v-position[i]))>radius)continue;
   proposals.push({...p,source_index,seed_threshold_px:threshold});
  }
  proposals.sort(compare);ranked.push(...proposals.slice(0,32));
 }
 return ranked.sort(compare);
}
function compare(a,b){return b.retrieval_support-a.retrieval_support||b.retrieval_inliers-a.retrieval_inliers||a.source_index-b.source_index}
