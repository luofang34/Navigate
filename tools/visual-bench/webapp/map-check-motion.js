export function mapViewChanged(previous,current,{camera,agl_m,fraction}){
 if(!Number.isFinite(fraction)||fraction<=0||fraction>1)throw Error('Invalid map motion fraction');
 const {width,height,fx,fy}=camera;
 if(![width,height,fx,fy,agl_m].every(v=>Number.isFinite(v)&&v>0))return false;
 const limit=Math.min(width,height)*fraction,focal=Math.max(fx,fy);
 return current.some(pose=>{
  if(pose.track_id==null)return true;
  const before=previous.find(p=>p.track_id===pose.track_id);
  if(!before)return true;
  const distance=Math.hypot(...pose.position_enu_m.map((v,i)=>v-before.position_enu_m[i]));
  const dot=Math.abs(pose.eye_to_enu_xyzw.reduce((sum,v,i)=>sum+v*before.eye_to_enu_xyzw[i],0));
  const angle=2*Math.acos(Math.min(1,dot));
  // The nominal height sets a search cadence. It does not constrain pose fitting or uncertainty.
  return focal*(distance/agl_m+angle)>=limit;
 });
}
