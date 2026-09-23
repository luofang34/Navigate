export function localPosition(pack,lat,lon,height){const [lat0,lon0]=pack.anchor_lat_lon;const radians=Math.PI/180,scale=6371008.8*Math.cos(lat0*radians);return [scale*(lon-lon0)*radians,scale*(Math.asinh(Math.tan(lat*radians))-Math.asinh(Math.tan(lat0*radians))),height]}

export function toGlobePose(pack,pose){
  const r=6371008.8,[lat0,lon0]=pack.anchor_lat_lon.map(v=>v*Math.PI/180),[x,y,h]=pose.position_enu_m;
  const lon=lon0+x/(r*Math.cos(lat0)),lat=Math.atan(Math.sinh(Math.asinh(Math.tan(lat0))+y/(r*Math.cos(lat0))));
  const axes=(a,b)=>[[Math.cos(b),0,-Math.sin(b)],[-Math.sin(b)*Math.sin(a),Math.cos(a),-Math.cos(b)*Math.sin(a)],[Math.sin(b)*Math.cos(a),Math.sin(a),Math.cos(b)*Math.cos(a)]];
  const anchor=axes(lat0,lon0),local=axes(lat,lon),dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0),world=local[2].map((v,i)=>v*(r+h)-anchor[2][i]*r);
  const m=anchor.map(a=>local.map(b=>dot(a,b))),trace=m[0][0]+m[1][1]+m[2][2];
  // Regional pose transport has a positive trace; globe browsing starts in anchor axes.
  const w=Math.sqrt(Math.max(0,1+trace))/2;
  if(w<.1)throw Error('Pose is outside this regional frame');
  const a=[(m[2][1]-m[1][2])/(4*w),(m[0][2]-m[2][0])/(4*w),(m[1][0]-m[0][1])/(4*w),w],b=pose.eye_to_enu_xyzw;
  const q=[a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]];
  const norm=Math.hypot(...q);return {position_enu_m:anchor.map(a=>dot(a,world)),eye_to_enu_xyzw:q.map(v=>v/norm)};
}
