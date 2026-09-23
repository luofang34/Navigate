import * as ort from '../runtime/ort.webgpu.min.mjs';
import {lighterGlueInputs,lighterGluePairs} from './lighterglue-decode.js';
export async function learnedPairs(session,first,second){
  if(!first.count||!second.count)return [];
  const a=lighterGlueInputs(first),b=lighterGlueInputs(second),feeds={
    keypoints0:new ort.Tensor('float32',a.keypoints,[1,first.count,2]),keypoints1:new ort.Tensor('float32',b.keypoints,[1,second.count,2]),
    descriptors0:new ort.Tensor('float32',a.descriptors,[1,first.count,64]),descriptors1:new ort.Tensor('float32',b.descriptors,[1,second.count,64])};
  let output;
  try{
    output=await session.run(feeds);const assignment=output.log_assignment;
    if(!assignment||assignment.dims.length!==3||assignment.dims[0]!==1||assignment.dims[1]!==first.count||assignment.dims[2]!==second.count)throw Error('Invalid LighterGlue output dimensions');
    return lighterGluePairs(assignment.data,first.count,second.count);
  }finally{for(const value of Object.values(feeds))value.dispose();if(output)for(const value of Object.values(output))value.dispose()}
}
