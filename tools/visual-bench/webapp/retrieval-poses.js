// These poses broaden retrieval. They do not constrain the final camera attitude.
export function cropPoseCandidates(ranked,position,prior,regionLimit){
 if(!Number.isInteger(regionLimit)||regionLimit<0||regionLimit>8)throw Error('Invalid crop pose budget');
 if(!Number.isFinite(prior.agl_m)||prior.agl_m<=0)return [];
 const chosen=[],seen=new Set();
 for(const candidate of ranked){
  if(chosen.length===regionLimit*3)break;
  const center=candidate.crop_center;
  if(!Array.isArray(center)||center.length!==3||center.some(v=>!Number.isFinite(v))||!Number.isFinite(candidate.angle)||seen.has(candidate.crop))continue;
  const point=[center[0],center[1],center[2]+prior.agl_m];
  if(Math.hypot(...point.map((v,i)=>v-position[i]))>prior.radius_m)continue;
  seen.add(candidate.crop);
  for(const offset of [0,-15,15]){
   const yaw=(candidate.angle+offset)*Math.PI/180;
   chosen.push({position_enu_m:point.slice(),eye_to_enu_xyzw:[0,0,Math.sin(yaw/2),Math.cos(yaw/2)]});
  }
 }
 return chosen;
}
