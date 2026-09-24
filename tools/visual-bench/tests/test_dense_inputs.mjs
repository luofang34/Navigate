import assert from 'node:assert/strict';
import {denseInput,supported} from '../webapp/inference/loftr-input.js';
const image={width:960,height:540,gray:Uint8Array.from({length:960*540},(_,i)=>i%256)};
const input=denseInput(image);assert.equal(input.data.length,640*480);assert.equal(input.flat,false);
for(const point of [[0,0],[479.5,269.5],[959,539]]){const model=[point[0]*2/3-1/6,point[1]*2/3-1/6+60],actual=input.pixel(model);assert.ok(Math.hypot(actual[0]-point[0],actual[1]-point[1])<1e-10)}
assert.equal(input.data[0],0,'padding cannot be treated as camera evidence');
assert.equal(denseInput({...image,gray:new Uint8Array(960*540)}).flat,true);
assert.equal(supported(image,[2,200]),false);assert.equal(supported(image,[500,200]),true);
const valid=new Uint8Array(960*540).fill(255);valid[200*960+500]=0;
assert.equal(supported({...image,valid},[503,201]),false,'exclude features that touch invalid reference data');
console.info('Dense matcher letterboxing, pixel centres, blank inputs and validity mask passed');
const {quarterTurn}=await import('../webapp/inference/loftr-input.js');
const tiny={width:3,height:2,gray:new Uint8Array([1,2,3,4,5,6]),valid:new Uint8Array([255,255,255,255,0,255])};
for(const turn of [1,2,3]){const r=quarterTurn(tiny,turn);for(let y=0;y<r.height;y++)for(let x=0;x<r.width;x++){const [sx,sy]=r.unrotate([x,y]);assert.equal(r.gray[y*r.width+x],tiny.gray[sy*tiny.width+sx]);assert.equal(r.valid[y*r.width+x],tiny.valid[sy*tiny.width+sx])}}
assert.deepEqual([...quarterTurn(tiny,1).gray],[4,1,5,2,6,3]);
