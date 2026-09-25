function similarSpan(first,second){
 return Number.isFinite(first)&&Number.isFinite(second)&&Math.min(first,second)>0
  &&Math.max(first,second)/Math.min(first,second)<1.25;
}

export function diverseShortlist(ranked,references,queries,limit){
 if(!Number.isInteger(limit)||limit<0)throw Error('Invalid shortlist limit');
 const centers=references.map(r=>r.world([(r.image.width-1)/2,(r.image.height-1)/2])),chosen=[];
 for(const candidate of ranked){
  if(chosen.length===limit)break;
  const c=centers[candidate.reference_index],a=queries[candidate.query_index]?.angle;
  if(!c||!Number.isFinite(a))continue;
  const duplicate=chosen.some(p=>{
   const d=centers[p.reference_index],b=queries[p.query_index].angle,difference=Math.abs(a-b)%360;
   return similarSpan(references[candidate.reference_index].span_m,references[p.reference_index].span_m)
    &&Math.hypot(c[0]-d[0],c[1]-d[1])<50&&Math.min(difference,360-difference)<15;
  });
  if(!duplicate)chosen.push(candidate);
 }
 return chosen;
}
