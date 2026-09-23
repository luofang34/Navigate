export function lighterGlueInputs(features){
  const {count,modelPixels,descriptors}=features;
  if(!Number.isInteger(count)||count<0||modelPixels.length!==count*2||descriptors.length!==count*64)throw Error('Invalid LighterGlue feature dimensions');
  const keypoints=new Float32Array(count*2),data=new Float32Array(count*64);
  for(let i=0;i<count;i++){
    // The sparse XFeat export returns pixels in an 800 by 600 image.
    keypoints[i*2]=(modelPixels[i*2]-400)/400;keypoints[i*2+1]=(modelPixels[i*2+1]-300)/400;
    for(let c=0;c<64;c++)data[i*64+c]=descriptors[c*count+i];
  }
  if(!keypoints.every(Number.isFinite)||!data.every(Number.isFinite))throw Error('Non-finite LighterGlue features');
  return {keypoints,descriptors:data};
}
export function lighterGluePairs(scores,rows,columns){
  if(!Number.isInteger(rows)||!Number.isInteger(columns)||rows<0||columns<0||scores.length!==rows*columns)throw Error('Invalid LighterGlue assignment shape');
  const bestRows=new Int32Array(rows).fill(-1),bestColumns=new Int32Array(columns).fill(-1),rowScore=new Float32Array(rows).fill(-Infinity),columnScore=new Float32Array(columns).fill(-Infinity);
  for(let i=0;i<rows;i++)for(let j=0;j<columns;j++){
    const score=scores[i*columns+j];if(!Number.isFinite(score))throw Error('Non-finite LighterGlue assignment');
    if(score>rowScore[i]){rowScore[i]=score;bestRows[i]=j}if(score>columnScore[j]){columnScore[j]=score;bestColumns[j]=i}
  }
  const pairs=[];for(let i=0;i<rows;i++){const j=bestRows[i];if(j>=0&&bestColumns[j]===i&&rowScore[i]>Math.log(.1))pairs.push([i,j])}return pairs;
}
