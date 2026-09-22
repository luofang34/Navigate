import init,{Preview} from '/wasm/navigate_visual_preview.js';
import {localPosition,toGlobePose} from './geography.js';
export {localPosition,toGlobePose} from './geography.js';
import {read,sha256} from './storage.js';
let initialized;
export class MapView{
  constructor(canvas){this.canvas=canvas;this.preview=null;this.pose=null;this.base=null;this.pending=false;this.again=false;this.drag=null;this.idle=Promise.resolve();this.frames=0;
    this.resizeObserver=new ResizeObserver(()=>this.draw().catch(console.error));this.resizeObserver.observe(canvas);
    canvas.addEventListener('pointerdown',e=>{canvas.focus({preventScroll:true});this.drag=[e.clientX,e.clientY,e.button===2||e.shiftKey];canvas.setPointerCapture(e.pointerId)});
    for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>this.drag=null);
    canvas.addEventListener('contextmenu',e=>e.preventDefault());
    canvas.addEventListener('pointermove',e=>{if(!this.drag||!this.pose)return;const dx=e.clientX-this.drag[0],dy=e.clientY-this.drag[1];const look=this.drag[2];this.drag=[e.clientX,e.clientY,look];if(look||this.orbit)this.rotate(dx*.003,dy*.003);else this.pan(-dx,dy);this.draw().catch(console.error)});
    canvas.addEventListener('wheel',e=>{e.preventDefault();this.move(2,this.height()*(Math.exp(Math.max(-.3,Math.min(.3,e.deltaY*.001)))-1)).catch(console.error)},{passive:false});
    canvas.addEventListener('keydown',e=>{const move={w:[1,20],s:[1,-20],a:[0,-20],d:[0,20]}[e.key.toLowerCase()];if(move){e.preventDefault();canvas.focus();this.move(move[0],move[1]*Math.max(1,this.height()/200)).catch(console.error)}});
  }
  async load(pack,camera){await this.dataIdle;await this.idle;initialized??=init();await initialized;if(!navigator.gpu)throw Error('WebGPU is unavailable. Open this local page in a WebGPU-enabled browser.');
    if(this.preview)this.preview.free();this.preview=null;this.pose=null;this.base=null;this.calibration=camera;this.sizeCanvas();this.preview=await Preview.create_display(JSON.stringify(pack),JSON.stringify(this.displayCamera()),read,this.canvas,navigator.gpu.getPreferredCanvasFormat());const context=await(await fetch('/context/earth.json')).json(),bytes=new Uint8Array(await(await fetch(context.url)).arrayBuffer());if(await sha256(bytes)!==context.sha256)throw Error('Globe context checksum failed');this.preview.set_globe_context(bytes);this.pack=pack;this.pack_id=pack.pack_id;this.displayPacks=new Map([[pack.pack_id,pack]]);
  }
  async setCalibration(camera){await this.dataIdle;await this.idle;this.calibration=camera;this.sizeCanvas();this.preview.resize_display(JSON.stringify(this.displayCamera()));await this.draw()}
  async addPack(pack){if(this.displayPacks?.has(pack.pack_id))return;if(this.displayPacks?.size>=4)throw Error('Four display packages are loaded. Select a region to start a new view.');await this.dataIdle;await this.idle;this.loading=true;let complete;this.dataIdle=new Promise(resolve=>complete=resolve);try{await this.preview.add_display_package(JSON.stringify(pack),read);this.displayPacks??=new Map();this.displayPacks.set(pack.pack_id,pack)}finally{this.loading=false;complete()}await this.draw()}
  async setPose(pose){this.orbit=false;const globePose=toGlobePose(this.pack,pose);this.base=structuredClone(globePose);this.pose=structuredClone(globePose);await this.draw()}
  changed(){this.canvas.dispatchEvent(new CustomEvent('viewchange',{detail:this.orbit?'globe':'free'}))}
  async globe(){this.orbit=true;this.pose={position_enu_m:[0,0,15_000_000],eye_to_enu_xyzw:[0,0,0,1]};this.changed();await this.draw()}
  async reset(){if(this.base){this.orbit=false;this.canvas.dispatchEvent(new CustomEvent('viewchange',{detail:'reset'}));this.pose=structuredClone(this.base);await this.draw()}}
  height(){if(!this.pose)return 100;return Math.max(20,Math.hypot(this.pose.position_enu_m[0],this.pose.position_enu_m[1],this.pose.position_enu_m[2]+6371008.8)-6371008.8)}
  async move(axis,amount){if(this.pose){this.changed();if(this.orbit&&axis===2){const p=this.pose.position_enu_m,v=[p[0],p[1],p[2]+6371008.8],r=Math.hypot(...v),next=Math.max(6371028.8,Math.min(46371008.8,r+amount));this.pose.position_enu_m=v.map((x,i)=>x*next/r-(i===2?6371008.8:0))}else this.pose.position_enu_m[axis]+=amount;await this.draw()}}
  rotate(yaw,pitch){this.changed();const multiply=(a,b)=>[a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]];
    if(this.orbit){const a=multiply([0,Math.sin(-yaw/2),0,Math.cos(yaw/2)],[Math.sin(-pitch/2),0,0,Math.cos(pitch/2)]),p=this.pose.position_enu_m,v=multiply(multiply(a,[p[0],p[1],p[2]+6371008.8,0]),[-a[0],-a[1],-a[2],a[3]]);this.pose.position_enu_m=[v[0],v[1],v[2]-6371008.8];const q=multiply(a,this.pose.eye_to_enu_xyzw),n=Math.hypot(...q);this.pose.eye_to_enu_xyzw=q.map(x=>x/n);return}
    const q=multiply([0,0,Math.sin(yaw/2),Math.cos(yaw/2)],multiply(this.pose.eye_to_enu_xyzw,[Math.sin(pitch/2),0,0,Math.cos(pitch/2)]));const norm=Math.hypot(...q);this.pose.eye_to_enu_xyzw=q.map(v=>v/norm);
  }
  displayCamera(){const c=this.calibration,w=this.canvas.width,h=this.canvas.height,scale=Math.min(w/c.width,h/c.height);return {width:w,height:h,fx:c.fx*scale,fy:c.fy*scale,cx:(c.cx+.5)*scale+(w-c.width*scale)/2-.5,cy:(c.cy+.5)*scale+(h-c.height*scale)/2-.5}}
  sizeCanvas(){const b=this.canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2),scale=Math.min(ratio,1920/Math.max(b.width,b.height));const w=Math.max(64,Math.round(b.width*scale)),h=Math.max(64,Math.round(b.height*scale));if(w===this.canvas.width&&h===this.canvas.height)return false;this.canvas.width=w;this.canvas.height=h;return true}
  pan(dx,dy){this.changed();const c=this.displayCamera(),scale=this.height()/(c.fx/this.canvas.width*this.canvas.clientWidth);const q=this.pose.eye_to_enu_xyzw;const right=rotateVector(q,[1,0,0]),up=rotateVector(q,[0,1,0]);for(let i=0;i<3;i++)this.pose.position_enu_m[i]+=(right[i]*dx+up[i]*dy)*scale}
  async draw(){if(this.loading)return;if(this.pending){this.again=true;return this.idle}if(!this.preview||!this.pose)return;this.pending=true;let complete;this.idle=new Promise(resolve=>complete=resolve);
    try{let settle=2;do{this.again=false;await frameOpportunity();if(this.sizeCanvas())this.preview.resize_display(JSON.stringify(this.displayCamera()));const start=performance.now();this.preview.present(JSON.stringify(this.pose));this.frames++;this.canvas.dataset.renderMs=(performance.now()-start).toFixed(2);this.canvas.dataset.frames=this.frames;this.canvas.dataset.presentation='wgpu-direct';}while(this.again||settle-->0)}finally{this.pending=false;complete()}
  }

}

function rotateVector(q,v){const [x,y,z,w]=q,[a,b,c]=v;const tx=2*(y*c-z*b),ty=2*(z*a-x*c),tz=2*(x*b-y*a);return [a+w*tx+y*tz-z*ty,b+w*ty+z*tx-x*tz,c+w*tz+x*ty-y*tx]}

function frameOpportunity(){return new Promise(resolve=>{let frame,timer;const done=()=>{cancelAnimationFrame(frame);clearTimeout(timer);resolve()};frame=requestAnimationFrame(done);timer=setTimeout(done,100)})}
