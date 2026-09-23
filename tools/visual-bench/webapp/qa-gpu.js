import {DescriptorRetrieval} from './inference/retrieval-gpu.js';
export async function checkGpuRetrieval(){
 const adapter=await navigator.gpu.requestAdapter(),device=await adapter.requestDevice(),gpu=await DescriptorRetrieval.create(device);
 try{
  const features=(count,reverse=false)=>{const descriptors=new Float32Array(count*256);for(let i=0;i<count;i++)descriptors[(reverse?count-i-1:i)*count+i]=1;return {count,descriptors}};
  const query=features(23),reference=features(17),reversed=features(23,true);const scores=await gpu.rankMany([reference,reversed],query);
  if(scores[0]!==17||scores[1]!==23)throw Error(`GPU retrieval disagrees with known correspondences: ${scores}`);
  const ambiguous={count:17,descriptors:new Float32Array(256*17).fill(1/16)};const ties=await gpu.rankMany([ambiguous],ambiguous);if(ties[0]!==0)throw Error('GPU retrieval accepted tied descriptors');return true;
 }finally{gpu.close();device.destroy()}
}

function seededFeatures(count,dimensions,offset=0){
 const descriptors=new Float32Array(count*dimensions);
 for(let i=0;i<count;i++){
  let state=(i+offset+1)>>>0,norm=0;
  for(let c=0;c<dimensions;c++){state=(Math.imul(state,1664525)+1013904223)>>>0;const value=state/4294967296-.5;descriptors[c*count+i]=value;norm+=value*value}
  const scale=1/Math.sqrt(norm);for(let c=0;c<dimensions;c++)descriptors[c*count+i]*=scale;
 }
 return {count,descriptors};
}

export async function checkBatchedGpuRetrieval(){
 const adapter=await navigator.gpu.requestAdapter(),device=await adapter.requestDevice(),report=[];
 device.pushErrorScope('validation');
 try {
  for(const dimensions of [64,256]){
   const gpu=await DescriptorRetrieval.create(device,dimensions,{minimumSimilarity:.82,maximumDistanceRatio:.9});
   try {
    const make=(n,offset=0)=>seededFeatures(n,dimensions,offset);
    const tied=n=>({count:n,descriptors:new Float32Array(n*dimensions).fill(1/Math.sqrt(dimensions))});
    const cases=[
     {name:'partial and empty references',references:[make(0),make(5),make(6),make(17),make(63),make(65,2)],query:make(71)},
     {name:'tied descriptors',references:[tied(17),tied(65)],query:tied(23)},
     {name:'multiple scratch groups',references:Array.from({length:96},(_,i)=>make(17+i%17,i%5)),query:make(1025)},
     {name:'empty query',references:[make(17),make(23)],query:make(0)},
     {name:'maximum feature count',references:[make(4096),make(4093,1)],query:make(4096)}
    ];
    for(const test of cases){
     const before=gpu.dispatches,started=performance.now();
     const expected=await gpu.rankMany(test.references,test.query),referenceMs=performance.now()-started,referenceDispatches=gpu.dispatches-before;
     const packed=gpu.batched.prepare(test.references),batchBefore=gpu.dispatches,batchStarted=performance.now();let actual;
     try{actual=await gpu.batched.rank(packed,test.query)}finally{packed.close()}
     const batchMs=performance.now()-batchStarted,batchDispatches=gpu.dispatches-batchBefore;
     if(JSON.stringify(actual)!==JSON.stringify(expected))throw Error(`${dimensions}: ${test.name}: batch ${actual} differs from scalar ${expected}`);
     if(test.name==='tied descriptors'&&actual.some(v=>v!==0))throw Error('Tied descriptors created retrieval evidence');
     if(test.name==='multiple scratch groups'&&!(batchDispatches<referenceDispatches))throw Error('GPU dispatch count did not decrease');
     report.push({dimensions,name:test.name,scores:actual,referenceDispatches,batchDispatches,referenceMs,batchMs});
    }
    const references=Array.from({length:321},(_,i)=>make(17+i%3,i%11)),queries=[make(37),make(23,5)];
    const expected=[];
    for(const [q,query] of queries.entries())for(let start=0;start<references.length;start+=160){const scores=await gpu.rankMany(references.slice(start,start+160),query);for(const [i,score] of scores.entries())expected.push({reference_index:start+i,query_index:q,score})}
    expected.sort((a,b)=>b.score-a.score||a.query_index-b.query_index||a.reference_index-b.reference_index);
    const ranked=await gpu.rankGrid(references,queries,64),indices=expected.slice(0,64).map(({reference_index,query_index})=>({reference_index,query_index}));
    if(JSON.stringify(ranked)!==JSON.stringify(indices))throw Error('GPU batching changed global candidate order');
    report.push({dimensions,name:'multiple reference batches retain top-64 order',passed:true});
   } finally {gpu.close()}
  }
  const error=await device.popErrorScope();if(error)throw Error(error.message);return report;
 }finally{device.destroy()}
}
