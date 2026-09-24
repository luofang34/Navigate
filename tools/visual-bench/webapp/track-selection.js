import {supportedPose,connectedSamples} from './pose-playback.js';

// Display one path. Separate restarts and alternatives require an explicit view.
export class TrackSelection {
 choose(branches,preferred,maxGap){
  if(this.branches===branches&&this.preferred===preferred&&this.maxGap===maxGap)return this.selected;
  this.branches=branches;this.preferred=preferred;this.maxGap=maxGap;
  const supported=[...branches].filter(([,samples])=>samples.some(s=>supportedPose(s.h)));
  let chosen=supported.find(([key])=>key===preferred)??supported[0];
  if(chosen){
   const samples=chosen[1],poses=samples.filter(s=>supportedPose(s.h));
   if(poses.length===1){
    const source=poses[0].source_h??poses[0].h;
    // An anchor and its connected display copy identify the same camera estimate.
    chosen=supported.find(([,items])=>items.some((s,i)=>(s.source_h??s.h)===source&&
     (connectedSamples(items[i-1],s,{maxGap})||connectedSamples(s,items[i+1],{maxGap}))))??chosen;
   }
  }
  this.selected=new Map(chosen?[chosen]:[]);return this.selected;
 }
}
