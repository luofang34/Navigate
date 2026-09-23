import assert from 'node:assert/strict';
import {referenceBatchSize,packDescriptors} from '../webapp/inference/retrieval-batch.js';
const feature=count=>({count,descriptors:Float32Array.from({length:count*64},(_,i)=>i+1)});
const a=feature(2),b=feature(3),packed=packDescriptors([a,b,feature(0)],64);
assert.equal(packed.stride,3);assert.deepEqual([...packed.counts],[2,3,0]);
for(let c=0;c<64;c++){
  assert.deepEqual([...packed.data.slice(c*3,c*3+3)],[c*2+1,c*2+2,0]);
  assert.deepEqual([...packed.data.slice((64+c)*3,(64+c+1)*3)],[c*3+1,c*3+2,c*3+3]);
}
assert.ok(packed.data.slice(128*3).every(v=>v===0),'empty references do not inherit a neighbour descriptor');
for(const dimensions of [64,256])for(const count of [0,1,17,512,4096]){
  const n=referenceBatchSize([{count}],dimensions);
  assert.ok(n>=1&&n<=160);assert.ok(n*Math.max(1,count)*dimensions*4<=16*1024*1024,'packed descriptors fit the upload budget');
}
assert.throws(()=>packDescriptors([],64),/batch/);
assert.throws(()=>packDescriptors(Array.from({length:161},()=>a),64),/batch/);
assert.throws(()=>packDescriptors([{count:4097,descriptors:new Float32Array(0)}],64),/shape/);
assert.throws(()=>packDescriptors([{count:-1,descriptors:new Float32Array(0)}],64),/shape/);
assert.throws(()=>packDescriptors([{count:2,descriptors:new Float32Array(2)}],64),/shape/);
assert.throws(()=>packDescriptors([a],128),/shape/);
console.info('Channel-major packing, variable lengths, empty references and memory bounds passed');
