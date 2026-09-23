export const EARTH_RADIUS=6371008.8;
export function globeLocation(pack,pose){
  const [lat,lon]=pack.anchor_lat_lon.map(v=>v*Math.PI/180),r=EARTH_RADIUS;
  const axes=[[Math.cos(lon),0,-Math.sin(lon)],[-Math.sin(lon)*Math.sin(lat),Math.cos(lat),-Math.cos(lon)*Math.sin(lat)],[Math.sin(lon)*Math.cos(lat),Math.sin(lat),Math.cos(lon)*Math.cos(lat)]];
  const p=pose.position_enu_m,v=[p[0],p[1],p[2]+r],radius=Math.hypot(...v);
  const world=axes[0].map((_,i)=>axes.reduce((sum,a,j)=>sum+a[i]*v[j],0));
  return {latitude:Math.asin(world[1]/radius)*180/Math.PI,longitude:Math.atan2(world[0],world[2])*180/Math.PI,altitude:radius-r};
}
export function minimumClearance(agl){return Math.max(20,Math.min(100,Number.isFinite(agl)?agl*.1:20))}
export function constrainCamera(pack,pose,elevationAt,minimum=20){
  const location=globeLocation(pack,pose),elevation=elevationAt(location.latitude,location.longitude);
  const known=Number.isFinite(elevation),floor=Math.max(20,known?elevation+minimum:10000);
  const result=structuredClone(pose),lift=Math.max(0,floor-location.altitude);
  if(lift>0){const p=result.position_enu_m,v=[p[0],p[1],p[2]+EARTH_RADIUS],radius=Math.hypot(...v);result.position_enu_m=v.map((x,i)=>x*(radius+lift)/radius-(i===2?EARTH_RADIUS:0));}
  return {pose:result,known,lift,minimum,altitude:location.altitude+lift,agl:known?location.altitude+lift-elevation:null};
}
