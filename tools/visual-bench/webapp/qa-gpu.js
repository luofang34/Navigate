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
