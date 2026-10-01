import assert from 'node:assert/strict';
import {LocalMatcher} from '../webapp/inference/local.js';

function matcher({glue}){
  const local=new LocalMatcher({matcher:'dense'}),calls={dense:0,sparse:0};
  local.denseAsset={sha256:'dense-model'};
  local.dense={match:async()=>{calls.dense++;return [{reference:[1,1],query:[2,2]}]},refine:async()=>{calls.dense++;return []}};
  local.matcher={glue,features:async()=>({count:12}),pairs:async()=>{calls.sparse++;return [{reference:[1,1],query:[1,1]}]}};
  local.identity='browser-xfeat/lighterglue';
  return {local,calls};
}
const image={gray:new Uint8Array(4),width:2,height:2};

const loaded=matcher({glue:true});
const tracked=await loaded.local.matchImages(image,image,{stage:'tracking',reference:'a',query:'b'});
assert.equal(tracked.backend_identity,'browser-xfeat/lighterglue','consecutive frames use sparse learned matching');
assert.deepEqual(loaded.calls,{dense:0,sparse:1});
const refined=await loaded.local.matchImages(image,image,{stage:'refinement',reference:'c',query:'d'});
assert.match(refined.backend_identity,/^browser-loftr/,'map checks keep the dense matcher');
assert.deepEqual(loaded.calls,{dense:1,sparse:1});

const denseOnly=matcher({glue:null});
const fallback=await denseOnly.local.matchImages(image,image,{stage:'tracking',reference:'e',query:'f'});
assert.match(fallback.backend_identity,/^browser-loftr/,'without LighterGlue tracking stays dense');
console.info('Tracking uses sparse matching when available; map checks stay dense');
