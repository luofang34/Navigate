import assert from 'node:assert/strict';
import {LocalizationPipeline} from '../webapp/localization.js';

const prepared=[],prior={latitude:1,longitude:2,radius_m:500,agl_m:110};
const pipeline=new LocalizationPipeline({prepareReferences:async crops=>{prepared.push(...crops)}},{scales:[.5,1]});
pipeline.camera={width:4,height:3};
pipeline.references={crops:(p,camera,scales)=>{assert.strictEqual(p,prior);assert.deepEqual(scales,[.5,1]);return [{key:'a'},{key:'b'}]}};
assert.equal(await pipeline.prepareReferences(prior,()=>{}),2);
assert.deepEqual(prepared.map(c=>c.key),['a','b'],'the crops of the first area search are prepared before any frame');
console.info('Reference features for the prior can be prepared before the first frame');
