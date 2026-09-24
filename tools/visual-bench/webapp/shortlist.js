export function diverseShortlist(ranked,references,queries,limit){
 if(!Number.isInteger(limit)||limit<0)throw Error('Invalid shortlist limit');
 const centers=references.map(r=>r.world([(r.image.width-1)/2,(r.image.height-1)/2])),chosen=[];
 for(const candidate of ranked){
  if(chosen.length===limit)break;
  const c=centers[candidate.reference_index],a=queries[candidate.query_index]?.angle;
  if(!c||!Number.isFinite(a))continue;
  if(chosen.some(p=>{const d=centers[p.reference_index],b=queries[p.query_index].angle,difference=Math.abs(a-b)%360;return Math.hypot(c[0]-d[0],c[1]-d[1])<50&&Math.min(difference,360-difference)<15}))continue;
  chosen.push(candidate);
 }
 return chosen;
}
