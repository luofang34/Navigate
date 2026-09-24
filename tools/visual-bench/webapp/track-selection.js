import {supportedPose,connectedSamples} from './pose-playback.js';

// This chooses a review display. It does not fuse alternatives or create evidence.
export class TrackSelection {
 choose(branches,preferred,maxGap){
  if(this.branches===branches&&this.preferred===preferred&&this.maxGap===maxGap)return this.selected;
  this.branches=branches;this.preferred=preferred;this.maxGap=maxGap;
  const timeline=new Map();
  for(const [key,samples] of branches)for(const [i,sample] of samples.entries())if(supportedPose(sample.h)){
   const available=timeline.get(sample.time)??[];
   const connections=2*Number(connectedSamples(sample,samples[i+1],{maxGap}))+Number(connectedSamples(samples[i-1],sample,{maxGap}));
   available.push({key,sample,connections});timeline.set(sample.time,available);
  }
  let current=preferred;const choices=new Map(),used=new Set();
  for(const [time,available] of [...timeline].sort(([a],[b])=>a-b)){
   let chosen=available.find(item=>item.key===preferred)??available.find(item=>item.key===current)??available[0];
   const source=chosen.sample.source_h??chosen.sample.h;
   if(!chosen.connections)chosen=available.filter(item=>(item.sample.source_h??item.sample.h)===source).sort((a,b)=>b.connections-a.connections)[0];
   current=chosen.key;choices.set(time,current);used.add(current);
  }
  this.selected=new Map([...branches].filter(([key])=>used.has(key)).map(([key,samples])=>[key,samples.map(sample=>choices.get(sample.time)===key?sample:{...sample,h:null})]));
  return this.selected;
 }
}
