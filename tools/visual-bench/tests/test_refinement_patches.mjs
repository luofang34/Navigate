import assert from 'node:assert/strict';
import {refinementPairs} from '../webapp/inference/refinement-patches.js';
const width=512,height=288,reference={width,height,gray:Uint8Array.from({length:width*height},(_,i)=>i%251),valid:new Uint8Array(width*height).fill(255)};
reference.valid[100*width+300]=0;
const query={...reference,gray:Uint8Array.from(reference.gray,v=>(v+1)%256)};let runs=0;
const pairs=await refinementPairs(reference,query,async(a,b)=>{runs++;assert.equal(a.width,b.width);assert.equal(a.height,b.height);for(let i=0;i<a.gray.length;i++)assert.equal(b.gray[i],(a.gray[i]+1)%256);assert.equal(a.valid[0],b.valid[0]);return [{reference:[10,20],query:[13,18]}]});
assert.equal(runs,5);assert.equal(pairs.length,5);
assert.deepEqual(pairs.map(p=>p.query.map((v,i)=>v-p.reference[i])),Array(5).fill([3,-2]),'patch offsets must preserve the measured pixel displacement');
assert.deepEqual(pairs.at(-1),{reference:[202,128],query:[205,126]});
let sawMissing=false;
await refinementPairs(reference,query,async(a)=>{sawMissing||=a.valid.some(v=>v===0);return []});assert.equal(sawMissing,true,'missing reference pixels must survive cropping');
let mismatchRuns=0;await refinementPairs(reference,{...query,width:256},async()=>{mismatchRuns++;return []});assert.equal(mismatchRuns,1,'different camera shapes only use the full-image adapter');
console.info('Overlapping refinement keeps pixel coordinates and missing-data masks in the original images');

const bounded=await refinementPairs(reference,query,async()=>Array.from({length:2000},()=>({reference:[10,20],query:[13,18]})));assert.equal(bounded.length,4096);assert.ok(bounded.some(p=>p.reference[0]===202&&p.reference[1]===128),'the correspondence budget must retain the last patch');
