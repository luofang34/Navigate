import assert from 'node:assert/strict';
import {cropPlan} from '../webapp/crop-plan.js';
import {LocalMatcher} from '../webapp/inference/local.js';
import {DescriptorRetrieval} from '../webapp/inference/retrieval-gpu.js';
const area={width:10752,height:9216,baseSize:1014,center:[8962,1984],metresPerPixel:.2266585,radius:500,scales:[.75,1,1.5]};
const plan=cropPlan(area);
assert.equal(plan.length,280,'all planned scales and positions remain in the search');
assert.ok(plan.slice(160).some(p=>p.distance>500),'edge crops remain available');
assert.throws(()=>cropPlan({...area,limit:160}),/full prior needs more than 160/,'a work limit must not silently reduce geographic coverage');
const matcher=new LocalMatcher(),seen=[];
const retrieval=new DescriptorRetrieval();retrieval.rankMany=async(refs,query)=>{assert.ok(refs.length<=160);seen.push(...refs.map(r=>r.id));return refs.map(r=>r.id===query.id?100:0)};
matcher.matcher={features:async image=>image,gpu:{phase:()=>{}},retrieval};
const references=Array.from({length:731},(_,id)=>({image:{id},key:`map/${id}`}));
const result=await matcher.retrievePairs(references,[{image:{id:710},key:'query/0'},{image:{id:170},key:'query/1'}],2,()=>{});
assert.deepEqual(result,[{reference_index:710,query_index:0},{reference_index:170,query_index:1}]);
assert.equal(seen.length,731*2);assert.equal(new Set(seen).size,731);
console.info('Full prior coverage, explicit work limits, and cross-batch candidate ranking passed');

// Buffers referenced by a queued batch must stay live until that batch is submitted.
const usage=globalThis.GPUBufferUsage;globalThis.GPUBufferUsage={STORAGE:1,COPY_DST:2};
try {
  const gpu=new DescriptorRetrieval();gpu.uploads=new Map();gpu.uploadCount=0;gpu.uploadBytes=0;gpu.device={createBuffer:()=>({destroyed:false,destroy(){this.destroyed=true}}),queue:{writeBuffer(){}}};
  const query={descriptors:new Float32Array(64)};
  for(let batch=0;batch<5;batch++){
    const queued=[];
    for(let i=0;i<160;i++){
      queued.push(gpu.descriptors({descriptors:new Float32Array(64)}),gpu.descriptors(query));
      assert.ok(queued.every(b=>!b.destroyed),'descriptor eviction cannot invalidate a pending GPU batch');
    }
    assert.ok(gpu.uploads.size<=192,'GPU descriptor memory stays bounded');
  }
} finally {if(usage===undefined)delete globalThis.GPUBufferUsage;else globalThis.GPUBufferUsage=usage;}
