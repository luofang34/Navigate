import assert from 'node:assert/strict';
import {decodeXFeat,xfeatInput,emptyXFeat,isEmptyXFeatOutput} from '../webapp/inference/xfeat-decode.js';
import {mutualPairs} from '../webapp/inference/retrieval-gpu.js';
const result={keypoints:{dims:[3,2],data:new Float32Array([8,8,16,16,24,24])},scores:{data:new Float32Array([.5,.9,.2])},descriptors:{dims:[3,64],data:new Float32Array(3*64)}};
for(let i=0;i<3;i++)result.descriptors.data[i*64+i]=2;
const image={width:32,height:32,valid:new Uint8Array(32*32).fill(255)};
image.valid[16*32+17]=0;
const f=decodeXFeat(result,image,2);
assert.equal(f.count,2);assert.deepEqual([...f.pixels],[8,8,24,24]);assert.equal(f.descriptors[0],1);assert.equal(f.descriptors[2*2+1],1);
assert.equal(decodeXFeat(result,image,1).count,1);
image.valid.fill(0);assert.equal(decodeXFeat(result,image).count,0);
const best=new Float32Array([1,.95,.5,0,0,.8,.79,0,1,.8,.79,0,0,.95,.5,0]);
assert.deepEqual(mutualPairs(best,2,2),[[1,0]]);
best[12]=1;assert.deepEqual(mutualPairs(best,2,2),[]);
console.info('XFeat validity, descriptor layout, limits, mutual matches and ambiguity checks passed');

const transformed=decodeXFeat(result,{width:32,height:32},1,{scale:2,x:0,y:0});assert.deepEqual([...transformed.pixels],[8,8]);assert.deepEqual([...transformed.modelPixels],[16,16]);
const input=xfeatInput({width:8,height:8,gray:new Uint8Array(64).fill(255)});assert.equal(input.width,800);assert.equal(input.height,600);assert.equal(input.data[300*800+400],1);assert.equal(input.data[300*800],0);assert.equal(input.transform.scale,75);

assert.equal(emptyXFeat(640,360).count,0);assert.equal(isEmptyXFeatOutput(Error("Name:'/Where' Status Message: Where: X operand cannot broadcast on dim 1 Condition Shape: {1,0,2}, X Shape: {1,0}, Y Shape: {}")),true);assert.equal(isEmptyXFeatOutput(Error('WebGPU device lost')),false);
