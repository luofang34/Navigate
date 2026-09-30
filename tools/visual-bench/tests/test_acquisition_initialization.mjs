import assert from 'node:assert/strict';
import {createAcquisitionPipeline} from '../webapp/qa-acquisition-pipeline.js';

function setup({missing=false,openError,initializeError,closeError}={}){
 const events=[],adapters=[],pack={pack_id:'map'},camera={width:640,height:360};let runtime;
 const make=kind=>{
  events.push('create '+kind);
  const adapter={kind,gpu:{kind},metrics:{kind},matcher:{kind:kind==='research'?'ORT matcher session':'public shared runtime'},
   async initialize(_progress,shared){events.push('initialize '+kind);this.shared=shared;if(kind==='research'&&initializeError)throw initializeError},
   async matchImages(){events.push('match '+kind);return {pairs:[],backend_identity:kind}},
   diagnostics(){return {kind}},async close(){events.push('close '+kind);if(kind==='research'&&closeError)throw closeError},
  };
  if(kind==='public')adapter.retrievePairs=async()=>['public retrieval'];
  adapters.push(adapter);return adapter;
 };
 const deps={createPublic:()=>make('public'),createResearch:()=>make('research'),
  createPipeline:matcher=>({matcher,renderer:{free(){events.push('free renderer')}},async initialize(p,c,progress){assert.strictEqual(p,pack);assert.strictEqual(c,camera);await matcher.initialize(progress)}}),
  async openSearch(p,_progress,shared){assert.strictEqual(p,pack);runtime=shared;events.push('open index');if(openError)throw openError;return missing?null:{async close(){events.push('close index')}}},
 };
 return {events,adapters,pack,camera,deps,runtime:()=>runtime,run:options=>createAcquisitionPipeline(pack,camera,options,()=>{},deps)};
}

let context=setup(),pipeline=await context.run({referenceSearch:true,traceMatching:'lightglue'});
assert.deepEqual(context.events,['create research','initialize research','open index']);
assert.equal(context.adapters.length,1,'indexed research must not load the public model first');
assert.equal(pipeline.matcher.retrievePairs,undefined,'the indexed path accepts an image matcher without a retrieval method');
assert.strictEqual(context.runtime(),pipeline.matcher,'CAMP shares the adapter runtime, not its internal ONNX session');
assert.equal((await pipeline.matcher.matchImages({},{})).backend_identity,'research');

context=setup();pipeline=await context.run({traceMatching:'lightglue'});
assert.deepEqual(await pipeline.matcher.retrievePairs(),['public retrieval']);
assert.strictEqual(context.adapters[1].shared,context.adapters[0].matcher);
assert.equal((await pipeline.matcher.matchImages({},{})).backend_identity,'research');
await pipeline.matcher.close();assert.equal(context.events.filter(e=>e==='close research').length,1);assert.equal(context.events.filter(e=>e==='close public').length,1);

context=setup();pipeline=await context.run({referenceSearch:true,traceMatching:'public',researchHybrid:true});
for(const stage of [undefined,'proposal','tracking'])assert.equal((await pipeline.matcher.matchImages({}, {},{stage})).backend_identity,'public');
assert.equal((await pipeline.matcher.matchImages({}, {},{stage:'refinement'})).backend_identity,'research');
assert.strictEqual(context.runtime(),context.adapters[0].matcher);
assert.match(pipeline.matcher.diagnostics().matcher_policy,/research refinement/);

context=setup({missing:true});await assert.rejects(context.run({referenceSearch:true,traceMatching:'lightglue'}),/No prepared reference index/);
assert.deepEqual(context.events,['create research','initialize research','open index','close research','free renderer']);
const failure=Error('index load failed');context=setup({openError:failure});await assert.rejects(context.run({referenceSearch:true,traceMatching:'lightglue'}),error=>error===failure);
assert.ok(context.events.includes('close research')&&context.events.includes('free renderer'));
const modelFailure=Error('research model failed');context=setup({initializeError:modelFailure});await assert.rejects(context.run({traceMatching:'lightglue'}),error=>error===modelFailure);
assert.ok(context.events.includes('close research')&&context.events.includes('close public'));
const cleanup=Error('device release failed');context=setup({openError:failure,closeError:cleanup});
await assert.rejects(context.run({referenceSearch:true,traceMatching:'lightglue'}),error=>error instanceof AggregateError&&error.errors[0]===failure&&error.errors[1].errors[0]===cleanup);
assert.ok(context.events.includes('free renderer'));
console.info('Acquisition initializes only required adapters, uses the declared hybrid matcher, and releases failed setups');
