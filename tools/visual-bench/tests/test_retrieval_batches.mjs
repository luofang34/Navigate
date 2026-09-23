import assert from 'node:assert/strict';
import {DescriptorRetrieval} from '../webapp/inference/retrieval-gpu.js';
globalThis.GPUBufferUsage={STORAGE:1,COPY_DST:2};
const references=Array.from({length:321},(_,id)=>({id,descriptors:new Float32Array(128)}));
const queries=Array.from({length:3},(_,id)=>({id,descriptors:new Float32Array(128)}));
const score=(reference,query)=>(reference.id*7+query.id*3)%17;
const ranker=new DescriptorRetrieval(),calls=[];let destroyed=0;
Object.assign(ranker,{uploads:new Map(),uploadCount:0,uploadBytes:0,device:{createBuffer:()=>({destroy:()=>destroyed++}),queue:{writeBuffer:()=>{}}}});
ranker.rankMany=async(batch,query)=>{calls.push([batch.map(f=>f.id),query.id]);for(const reference of batch){ranker.descriptors(reference);ranker.descriptors(query)}return batch.map(reference=>score(reference,query))};
const expected=queries.flatMap((query,q)=>references.map((reference,r)=>({reference_index:r,query_index:q,score:score(reference,query)}))).sort((a,b)=>b.score-a.score).slice(0,64).map(({reference_index,query_index})=>({reference_index,query_index}));
assert.deepEqual(await ranker.rankGrid(references,queries,64),expected,'ranking and ties retain the same candidate order');
assert.equal(calls.length,9);assert.ok(calls.every(([batch])=>batch.length<=160));
assert.deepEqual(calls.slice(0,3).map(([,q])=>q),[0,1,2]);assert.ok(calls.slice(0,3).every(([batch])=>batch[0]===0&&batch.at(-1)===159));
assert.ok(ranker.uploadCount<=references.length+queries.length*3,'uploads scale with references, not reference-heading pairs');
assert.equal(ranker.uploadBytes,ranker.uploadCount*128*4);assert.ok(destroyed>0);assert.ok(ranker.uploads.size<=192,'cache remains bounded');
assert.deepEqual(await ranker.rankGrid([],queries,10),[]);assert.deepEqual(await ranker.rankGrid(references,[],10),[]);assert.deepEqual(await ranker.rankGrid(references,queries,0),[]);assert.equal(calls.length,9);
await assert.rejects(ranker.rankGrid(references,queries,-1),/limit/);
ranker.rankMany=async()=>[NaN];await assert.rejects(ranker.rankGrid([references[0]],queries,1),/scores/);
ranker.rankMany=async()=>[];await assert.rejects(ranker.rankGrid([references[0]],queries,1),/scores/);
console.info('Retrieval batch reuse, stable ranking, bounded uploads, empty work and invalid scores passed');

const batched=new DescriptorRetrieval();batched.dimensions=64;let prepared=0,released=0;
const packedReferences=references.map(f=>({...f,count:2})),packedQueries=queries.map(f=>({...f,count:2}));
batched.batched={prepare(batch){prepared++;return {batch,close(){released++}}},async rank({batch},query){return batch.map(reference=>score(reference,query))}};
assert.deepEqual(await batched.rankGrid(packedReferences,packedQueries,64),expected);
assert.equal(prepared,3);assert.equal(released,prepared,'each packed reference batch is released after its queries');
batched.batched.rank=async()=>{throw Error('device lost')};
await assert.rejects(batched.rankGrid(packedReferences,packedQueries,64),/device lost/);
assert.equal(released,prepared,'a failed GPU batch releases its reference buffers');
console.info('Batched ranking preserves candidate order and releases references on success and failure');

let pipelines=0,buffers=0,destroyedBuffers=0;
const initializationFailure={
 createShaderModule:()=>({}),
 async createComputePipelineAsync(){if(++pipelines===4)throw Error('unsupported shader');return {}},
 createBuffer(){buffers++;return {destroy(){destroyedBuffers++}}}
};
await assert.rejects(DescriptorRetrieval.create(initializationFailure,64),/unsupported shader/);
assert.equal(destroyedBuffers,buffers,'shader setup failure releases all allocated scratch buffers');
