import {toGlobePose} from './geography.js';
import {supportedPose,connectedSamples} from './pose-playback.js';

export function projectPoint(point,pose,camera){
 const q=pose.eye_to_enu_xyzw,[x,y,z,w]=[-q[0],-q[1],-q[2],q[3]],v=point.map((p,i)=>p-pose.position_enu_m[i]);
 const [a,b,c]=v,tx=2*(y*c-z*b),ty=2*(z*a-x*c),tz=2*(x*b-y*a);
 const eye=[a+w*tx+y*tz-z*ty,b+w*ty+z*tx-x*tz,c+w*tz+x*ty-y*tx],depth=-eye[2];
 return depth>.1?[camera.fx*eye[0]/depth+camera.cx,camera.cy-camera.fy*eye[1]/depth]:null;
}

export class TrackProjection {
 project(branches,pack,presented,maxGap){
  // Frame updates replace the branch map. Render events carry immutable camera snapshots.
  if(this.branches!==branches||this.pack!==pack||this.maxGap!==maxGap){
   this.branches=branches;this.pack=pack;this.maxGap=maxGap;this.presented=null;
   this.world=new Map([...branches].map(([key,samples])=>[key,worldPath(samples,pack,maxGap)]));
  }
  if(this.presented!==presented){
   this.presented=presented;
   this.paths=new Map([...this.world].map(([key,points])=>[key,screenPath(points,presented)]));
  }
  return this.paths;
 }
}

function worldPath(samples,pack,maxGap){
 const points=[];let previous=null;
 for(const sample of samples){
  if(!supportedPose(sample.h)){previous=null;continue}
  points.push({position:toGlobePose(pack,sample.h).position_enu_m,connect:connectedSamples(previous,sample,{maxGap})});previous=sample;
 }
 return points;
}

function screenPath(points,presented){
 const commands=[];let visible=false;
 for(const {position,connect} of points){
  const point=projectPoint(position,presented.pose,presented.camera);
  if(point)commands.push({point,connect:visible&&connect});
  visible=Boolean(point);
 }
 return commands;
}
