import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {constrainCamera,minimumClearance} from '../webapp/camera-clearance.js';
Object.assign(globalThis,{constrainCamera,minimumClearance});
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
await map.globe();const height=map.height();map.rotate(.2,.1);assert.ok(Math.abs(map.height()-height)<1e-6);await map.reset();assert.deepEqual(map.pose,map.base);

await map.move(2,-100000);assert.ok(map.height()>=119.999,'wheel and zoom cannot enter terrain');assert.equal(JSON.parse(canvas.dataset.clearance).known,true);
map.preview.terrain_elevation_cached=()=>undefined;await map.move(0,20);assert.ok(map.height()>=9999.99,'missing terrain enforces a conservative browsing floor');assert.equal(canvas.dataset.projection,'globe');
