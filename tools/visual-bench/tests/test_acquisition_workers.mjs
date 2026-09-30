import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

for(const filename of ['qa-acquisition-worker.js','qa-lightglue-search-worker.js']){
 const replies=[],created=[],events=[];let current,failEstimate=false,failInitialize=false;
 const canvas=()=>({getContext:()=>({putImageData(){}})});
 const self={postMessage:message=>replies.push(message)};
 const context=vm.createContext({self,OffscreenCanvas:class{constructor(){return canvas()}},makeCanvas:canvas,
  ImageData:class{},performance,
  async createAcquisitionPipeline(pack,camera,options,progress){
   created.push({pack,camera,options});progress('initializing');if(failInitialize)throw Error('index unavailable');
   current={options,referenceSearch:options.referenceSearch?{}:null,
    matcher:{async matchImages(){return {pairs:[]}}},
    beginSequence(...args){events.push(['begin',...args])},
    async estimate(image,prior,sequence,progress){
     assert.ok(image.canvas);assert.equal(sequence,7);assert.equal(prior.radius_m,50);progress('matching');
     if(failEstimate)throw Error('inference failed');return {decision:'unresolved',candidate_hypotheses:[]};
    },
   };return current;
  },
 });
 const source=await readFile(new URL('../webapp/'+filename,import.meta.url),'utf8');
 new vm.Script(source.replace(/^import .*;\n/gm,''),{filename}).runInContext(context);
 let id=0;
 const request=async(method,args)=>{const next=++id;await self.onmessage({data:{id:next,method,args}});return replies.filter(r=>r.id===next&&('value'in r||'error'in r)).at(-1)};
 const pack={pack_id:'map'},camera={width:2,height:2};
 for(const options of [{referenceSearch:true,traceMatching:'lightglue'},{referenceSearch:true,researchHybrid:true},{referenceSearch:false,traceMatching:'lightglue'}]){
  const reply=await request('initialize',[pack,camera,options]);
  assert.equal(reply.value.original_pixels,options.referenceSearch,filename+' reports actual source-pixel requirement');
  assert.strictEqual(created.at(-1).pack,pack);assert.strictEqual(created.at(-1).camera,camera);
  for(const [key,value] of Object.entries(options))assert.equal(created.at(-1).options[key],value);
 }
 await request('beginSequence',[{maxFrames:4}]);assert.equal(events.at(-1)[1].maxFrames,4);
 for(const fail of [false,true]){
  failEstimate=fail;let closes=0;
  const image={gray:new Uint8Array(4),width:2,height:2,original:{bitmap:{close(){closes++}}}};
  const reply=await request('estimate',[image,{radius_m:50},7]);
  if(fail)assert.match(reply.error,/inference failed/);else assert.equal(reply.value.decision,'unresolved');
  assert.equal(closes,1,filename+' releases unconsumed source bitmap on either outcome');
 }
 failInitialize=true;const reply=await request('initialize',[pack,camera,{referenceSearch:true}]);
 assert.match(reply.error,/index unavailable/);assert.equal(reply.value,undefined);
 assert.ok(replies.some(r=>r.progress==='matching'));
}
console.info('Both research worker entry points preserve requested adapters, report capture requirements, and clean up failed estimates');
