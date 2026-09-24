import {toGlobePose} from './geography.js';
import {playbackPose,supportedPose,connectedSamples,VideoPoseClock} from './pose-playback.js';

import {branchKey} from './hypotheses.js';
import {TrackProjection,projectPoint} from './track-projection.js';
import {TrackSelection} from './track-selection.js';
export {projectPoint} from './track-projection.js';
export {branchKey} from './hypotheses.js';
export function trackBranches(frames){
 const branches=new Map();
 for(const frame of frames){
  const time=frame.capture_time_ns/1e9,found=new Map();
  for(const h of frame.candidate_hypotheses??[])if(supportedPose(h))found.set(branchKey(frame,h),h);
  for(const key of found.keys())if(!branches.has(key))branches.set(key,[]);
  for(const [key,samples] of branches)samples.push({time,h:found.get(key)??null,frame});
 }
 attachMapAnchors(branches,frames);return branches;
}
function anchorKey(anchor){return anchor?.observation_sha256&&anchor?.map_manifest_sha256?JSON.stringify([anchor.observation_sha256,anchor.candidate_id,anchor.map_manifest_sha256]):null}
function attachMapAnchors(branches,frames){
 const anchors=new Map();
 for(const frame of frames)for(const h of frame.candidate_hypotheses??[])if(h.accepted){const key=anchorKey({observation_sha256:frame.observation_sha256,candidate_id:h.candidate_id,map_manifest_sha256:h.map_manifest_sha256});if(key)anchors.set(key,{frame,h,time:frame.capture_time_ns/1e9})}
 for(const samples of branches.values()){
  const links=new Map(samples.filter(s=>anchorKey(s.h?.tracking_anchor)).map(s=>[anchorKey(s.h.tracking_anchor),s.h]));
  for(const [key,linked] of links){
   const source=anchors.get(key);if(!source)continue;const index=samples.findIndex(s=>s.frame===source.frame);
   if(index>=0&&samples[index].h&&samples[index].h!==source.h)continue;
   // The display uses the exact parent pose. This creates no observation or confidence.
   const sample={...source,source_h:source.h,h:{...source.h,track_id:linked.track_id,candidate_id:linked.candidate_id,tracking_anchor:linked.tracking_anchor,anchor_observation_sha256:linked.anchor_observation_sha256,continuity_break:false}};
   if(index>=0)samples[index]=sample;else samples.push(sample);
  }
  samples.sort((a,b)=>a.time-b.time);
 }
}

export class TrackPreview {
 constructor(map,video,{overlay,follow,camera,overview,status,alternatives,selection=()=>{}}){
  Object.assign(this,{map,video,overlay,follow,camera,overviewButton:overview,status,alternatives,selection});this.branches=new Map();this.key=null;this.maxGap=.55;this.presented=null;
  map.canvas.addEventListener('render',e=>{this.presented=e.detail;this.paint()});
  map.canvas.addEventListener('viewchange',e=>{if(e.detail==='free')follow.checked=false});
  follow.addEventListener('change',()=>{this.lastTime=null;this.update(this.clock.time)});
  alternatives?.addEventListener('change',()=>this.paint());
  camera.onclick=()=>{if(this.current)this.map.setPose(this.current.pose).catch(e=>this.fail(e))};
  overview.onclick=()=>this.overview().catch(e=>this.fail(e));
  this.clock=new VideoPoseClock(video,time=>this.update(time));
 }
 fail(error){this.status.textContent=String(error)}
 clear(){this.projection=null;this.selectionProjection=null;this.trackSelection=null;this.branches.clear();this.key=null;this.selectedTime=null;this.current=null;this.follow.checked=false;this.camera.disabled=true;this.status.textContent='';this.paint()}
 setFrames(frames,{period=.5}={}){this.branches=trackBranches(frames);this.maxGap=Math.max(.05,period*1.1);if(!this.branches.has(this.key))this.key=this.branches.keys().next().value??null;this.update(this.clock.time)}
 select(frame,h){this.selectedTime=frame.capture_time_ns/1e9;this.key=h?branchKey(frame,h):null;this.update(this.clock.time)}
 update(time){
  if(this.video.hidden)time=this.selectedTime??0;
  const available=[...this.branches].map(([key,samples])=>({key,current:playbackPose(samples,time,{maxGap:this.maxGap})})).filter(item=>item.current);
  // Choosing a view does not join the evidence or interpolate across a map restart.
  let selected=available.find(item=>item.key===this.key)??available[0];
  if(selected&&!sampleConnections(this.branches.get(selected.key),selected.current.sample,this.maxGap)){
   const source=selected.current.sample.source_h??selected.current.sample.h;
   selected=available.filter(item=>(item.current.sample.source_h??item.current.sample.h)===source)
    .sort((a,b)=>sampleConnections(this.branches.get(b.key),b.current.sample,this.maxGap)-sampleConnections(this.branches.get(a.key),a.current.sample,this.maxGap))[0];
  }
  if(selected)this.key=selected.key;this.current=selected?.current??null;this.camera.disabled=!this.current;
  const alternatives=new Set(available.map(({current:{sample}})=>sample.source_h??sample.h)).size;
  if(this.branches.size)this.status.textContent=this.current?`${time.toFixed(2)} s · estimated camera${alternatives>1?` · ${alternatives} alternatives`:''}`:`${time.toFixed(2)} s · no supported pose`;
  if(this.reportedSample!==this.current?.sample||this.reportedKey!==this.key){this.reportedSample=this.current?.sample;this.reportedKey=this.key;this.selection({time,current:this.current,alternatives})}
  if(this.follow.checked&&this.current&&this.map.preview&&(time!==this.lastTime||this.key!==this.lastFollowKey)){this.lastTime=time;this.lastFollowKey=this.key;this.map.setPose(this.current.pose,{immediate:true}).catch(e=>this.fail(e))}
  this.paint();
 }
 async overview(){
  this.follow.checked=false;
  this.trackSelection??=new TrackSelection();
  const shown=this.alternatives?.checked?this.branches:this.trackSelection.choose(this.branches,this.key,this.maxGap);
  const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];let count=0;
  for(const samples of shown.values())for(const {h} of samples)if(supportedPose(h)&&h.position_enu_m.every(Number.isFinite)){
   for(let i=0;i<3;i++){low[i]=Math.min(low[i],h.position_enu_m[i]);high[i]=Math.max(high[i],h.position_enu_m[i])}count=(count+1)>>>0;
  }
  if(!count)return;
  const x=(low[0]+high[0])/2,y=(low[1]+high[1])/2;
  const angle=.55,height=Math.max(350,high[2]+150,(high[0]-low[0])*2,(high[1]-low[1])*2);
  await this.map.setPose({position_enu_m:[x,y-height*Math.tan(angle),height],eye_to_enu_xyzw:[Math.sin(angle/2),0,0,Math.cos(angle/2)]},{constrain:true});
 }
 paint(){
  const canvas=this.overlay;if(canvas.width!==this.map.canvas.width)canvas.width=this.map.canvas.width;if(canvas.height!==this.map.canvas.height)canvas.height=this.map.canvas.height;const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!this.presented||!this.map.pack)return;
  const point=h=>projectPoint(toGlobePose(this.map.pack,h).position_enu_m,this.presented.pose,this.presented.camera),scale=canvas.width/this.map.canvas.clientWidth;
  this.projection??=new TrackProjection();this.selectionProjection??=new TrackProjection();this.trackSelection??=new TrackSelection();
  const selected=this.trackSelection.choose(this.branches,this.key,this.maxGap),paths=[];
  if(this.alternatives?.checked)for(const commands of this.projection.project(this.branches,this.map.pack,this.presented,this.maxGap).values())paths.push({commands,color:'#f6b16b'});
  for(const commands of this.selectionProjection.project(selected,this.map.pack,this.presented,this.maxGap).values())paths.push({commands,color:'#65bdff'});
  for(const {commands,color} of paths){
   ctx.beginPath();
   for(const {point,connect} of commands){if(connect)ctx.lineTo(...point);else ctx.moveTo(...point)}
   ctx.strokeStyle='#08111e';ctx.lineWidth=5*scale;ctx.stroke();ctx.strokeStyle=color;ctx.lineWidth=2.5*scale;ctx.stroke();
  }
  if(this.current){const p=point(this.current.pose);if(p){ctx.beginPath();ctx.arc(...p,6*scale,0,Math.PI*2);ctx.fillStyle='white';ctx.fill();ctx.strokeStyle='#65bdff';ctx.lineWidth=3*scale;ctx.stroke()}}
 }
 close(){this.clock.close()}
}

function sampleConnections(samples,sample,maxGap){
 const index=samples.indexOf(sample);
 return (connectedSamples(sample,samples[index+1],{maxGap})?2:0)+(connectedSamples(samples[index-1],sample,{maxGap})?1:0);
}
