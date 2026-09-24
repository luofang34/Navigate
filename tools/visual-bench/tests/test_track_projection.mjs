import assert from 'node:assert/strict';
import {TrackProjection,projectPoint} from '../webapp/track-projection.js';
import {toGlobePose} from '../webapp/geography.js';
import {connectedSamples,supportedPose} from '../webapp/pose-playback.js';

const h=(x,z=100)=>({tracking_supported:true,map_manifest_sha256:'map',track_id:'branch',position_enu_m:[x,0,z],eye_to_enu_xyzw:[0,0,0,1]});
const samples=[{time:0,h:h(10)},{time:.2,h:h(20)},{time:.4,h:null},{time:.6,h:h(30)},{time:.8,h:h(40,1200)},{time:1,h:h(50)},{time:1.2,h:h(60)}];
const branches=new Map([['branch',samples]]),pack={anchor_lat_lon:[40.5,-74.4]},original=JSON.stringify([...branches]);
const view={pose:{position_enu_m:[0,0,1000],eye_to_enu_xyzw:[0,0,0,1]},camera:{fx:600,fy:600,cx:320,cy:180}};
function uncached(branches,pack,view,maxGap){
 return new Map([...branches].map(([key,samples])=>{
  let previous=null;const commands=[];
  for(const sample of samples){
   const point=supportedPose(sample.h)?projectPoint(toGlobePose(pack,sample.h).position_enu_m,view.pose,view.camera):null;
   if(!point){previous=null;continue}
   commands.push({point,connect:Boolean(previous&&connectedSamples(previous,sample,{maxGap}))});previous=sample;
  }
  return [key,commands];
 }));
}
const projection=new TrackProjection(),first=projection.project(branches,pack,view,.22);
assert.deepEqual(first,uncached(branches,pack,view,.22));
assert.deepEqual(first.get('branch').map(p=>p.connect),[false,true,false,false,true],'missing observations and behind-camera points split the drawn path');
assert.strictEqual(projection.project(branches,pack,view,.22),first,'stationary video playback reuses the projected path');
for(const updated of [
 {...view,pose:{...view.pose,position_enu_m:[0,0,700]}},
 {...view,pose:{...view.pose,eye_to_enu_xyzw:[0,0,Math.sin(.2),Math.cos(.2)]}},
 {...view,camera:{fx:1200,fy:1200,cx:640,cy:360}}
]){
 const result=projection.project(branches,pack,updated,.22);
 assert.deepEqual(result,uncached(branches,pack,updated,.22),'each presented camera or viewport updates the overlay immediately');
 assert.notDeepEqual(result,first);
}
const shifted={anchor_lat_lon:[65,12]};
assert.deepEqual(projection.project(branches,shifted,view,.22),uncached(branches,shifted,view,.22),'changing the map frame invalidates world coordinates');
const disconnected=projection.project(branches,pack,view,.1);
assert.ok(disconnected.get('branch').every(p=>!p.connect),'changing the gap policy invalidates cached connections');
const updated=new Map([['branch',[...samples,{time:1.4,h:h(70)}]]]);
assert.deepEqual(projection.project(updated,pack,view,.22),uncached(updated,pack,view,.22),'new frames appear without a map move');
assert.equal(projection.project(new Map(),pack,view,.22).size,0,'clearing a sequence clears projected paths');
assert.equal(JSON.stringify([...branches]),original,'projection does not alter pose evidence');
console.info('Cached paths retain exact projection, update on every camera change, and preserve unsupported gaps');
