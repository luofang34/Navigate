import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {toGlobePose} from '../webapp/geography.js';
import {constrainCamera,minimumClearance} from '../webapp/camera-clearance.js';
Object.assign(globalThis,{constrainCamera,minimumClearance,toGlobePose});
const source=(await fs.readFile(new URL('../webapp/map.js',import.meta.url),'utf8')).replace(/^(import|export \{).*\n/gm,'');
const {MapView}=await import('data:text/javascript,'+encodeURIComponent(source));
globalThis.ResizeObserver=class {observe(){}};
globalThis.devicePixelRatio=2;
let next=0;const timers=new Map();
globalThis.requestAnimationFrame=fn=>{const id=++next;timers.set(id,setTimeout(fn,0));return id};
globalThis.cancelAnimationFrame=id=>clearTimeout(timers.get(id));
class Canvas extends EventTarget {
  width=640;height=360;clientWidth=800;dataset={};
  getBoundingClientRect(){return {width:this.clientWidth,height:450}}
  focus(){this.focused=true}
  setPointerCapture(id){this.captured=id}
}
const canvas=new Canvas(),map=new MapView(canvas),frames=[];
map.calibration={width:640,height:360,fx:460,fy:460,cx:319.5,cy:179.5};
map.pack={anchor_lat_lon:[40.54,-74.45]};
map.preview={terrain_elevation_cached:()=>100,present:pose=>frames.push(JSON.parse(pose)),resize_display:value=>{map.resized=JSON.parse(value)}};
map.pose={position_enu_m:[0,0,800],eye_to_enu_xyzw:[0,0,0,1]};map.base=structuredClone(map.pose);
const fire=(type,data={})=>{const e=new Event(type,{cancelable:true});Object.assign(e,data);canvas.dispatchEvent(e)};
fire('pointerdown',{clientX:100,clientY:100,pointerId:1,button:0});
assert.equal(canvas.focused,true);assert.equal(canvas.captured,1);
fire('pointermove',{clientX:140,clientY:120});await map.idle;
assert.ok(map.pose.position_enu_m[0]<0);assert.deepEqual(map.pose.eye_to_enu_xyzw,[0,0,0,1]);
assert.equal(canvas.width,1600);assert.equal(canvas.height,900);assert.equal(map.resized.width,1600);
assert.equal(canvas.dataset.presentation,'wgpu-direct');assert.ok(frames.length>0);
const p=structuredClone(map.pose);fire('pointercancel');fire('pointermove',{clientX:150,clientY:140});assert.deepEqual(map.pose,p);
await map.reset();assert.deepEqual(map.pose,map.base);
fire('pointerdown',{clientX:100,clientY:100,pointerId:2,button:2});fire('pointermove',{clientX:140,clientY:120});await map.idle;
assert.notDeepEqual(map.pose.eye_to_enu_xyzw,[0,0,0,1]);assert.ok(Math.abs(Math.hypot(...map.pose.eye_to_enu_xyzw)-1)<1e-12);
await map.reset();fire('keydown',{key:'w'});await map.idle;assert.ok(map.pose.position_enu_m[1]>0);
const before=map.height();fire('wheel',{deltaY:100});await map.idle;assert.ok(map.height()>before);
const viewChanges=[];canvas.addEventListener('viewchange',event=>viewChanges.push(event.detail));
const selectedPose=structuredClone(map.base);await map.globe();
assert.deepEqual(viewChanges,['globe'],'the app receives a distinct globe-overview event');
assert.deepEqual(map.base,selectedPose,'globe navigation preserves the selected camera pose');
const height=map.height();assert.ok(height>10_000_000);map.rotate(.2,.1);assert.ok(Math.abs(map.height()-height)<1e-6);
await map.reset();assert.deepEqual(map.pose,selectedPose);assert.equal(viewChanges.at(-1),'reset');

await map.move(2,-100000);assert.ok(map.height()>=119.999,'wheel and zoom cannot enter terrain');assert.equal(JSON.parse(canvas.dataset.clearance).known,true);
map.preview.terrain_elevation_cached=()=>undefined;await map.move(0,20);assert.ok(map.height()>=9999.99,'missing terrain enforces a conservative browsing floor');assert.equal(canvas.dataset.projection,'globe');

const rendered=[];
canvas.addEventListener('render',({detail})=>{
 assert.deepEqual(detail.pose,frames.at(-1),'overlay uses the camera of the presented map frame');
 assert.equal(detail.camera.width,canvas.width);
 rendered.push(detail);
});
map.preview.terrain_elevation_cached=()=>100;
const exact={position_enu_m:[0,0,110],eye_to_enu_xyzw:[0,0,0,1]};
await map.setPose(exact);
assert.ok(Math.abs(map.height()-110)<1e-6,'estimated camera is not lifted by browsing clearance');
assert.equal(JSON.parse(canvas.dataset.clearance).adjusted,false);
await map.move(2,-1);
assert.ok(map.height()>=119.999,'user browsing enables terrain clearance');
await map.reset();assert.ok(Math.abs(map.height()-110)<1e-6,'reset restores the exact estimate');
await map.setPose(exact,{constrain:true});assert.ok(map.height()>=119.999,'reference overview requests retain browsing clearance');
await map.reset();assert.ok(map.height()>=119.999,'reset retains the reference-view clearance policy');
await map.setPose(exact);
const count=rendered.length,pan=map.move(0,20);
await new Promise(resolve=>canvas.addEventListener('render',resolve,{once:true}));
assert.ok(rendered.length>count,'overlay redraw arrives during camera movement');
assert.equal(map.pending,true,'overlay does not wait for the map to become idle');
await pan;

const queued=new Map();
globalThis.requestAnimationFrame=callback=>{const id=++next;queued.set(id,callback);return id};
globalThis.cancelAnimationFrame=id=>queued.delete(id);
const beforeFollow=frames.length,deferred=map.setPose(exact);
assert.equal(frames.length,beforeFollow,'ordinary input remains coalesced until a frame opportunity');
const presented=new Promise(resolve=>canvas.addEventListener('render',resolve,{once:true}));
const newest={...exact,position_enu_m:[25,0,110]},following=map.setPose(newest,{immediate:true});
await presented;
assert.deepEqual(frames.at(-1),toGlobePose(map.pack,newest),'a decoded video frame wakes a pending map draw without another animation-frame callback');
async function drain(){
 for(let i=0;map.pending&&i<8;i++){
  const entry=queued.entries().next().value;assert.ok(entry,'pending settling render has a scheduled opportunity');
  queued.delete(entry[0]);entry[1]();await Promise.resolve();
 }
 assert.equal(map.pending,false);
}
await drain();await Promise.all([deferred,following]);
const countImmediate=frames.length,firstImmediate=map.setPose(exact,{immediate:true});
assert.equal(frames.length,countImmediate+1,'an idle renderer presents a timed pose immediately');
assert.deepEqual(frames.at(-1),toGlobePose(map.pack,exact));
await drain();await firstImmediate;
console.info('Timed camera updates wake the renderer while ordinary input keeps frame coalescing');
