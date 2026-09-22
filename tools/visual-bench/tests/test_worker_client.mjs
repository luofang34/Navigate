import assert from 'node:assert/strict';
import {BrowserPipeline} from '../webapp/browser-pipeline.js';
class Worker {
  constructor(){this.requests=[];Worker.last=this}
  postMessage(value){this.requests.push(value)}
  terminate(){this.terminated=true}
  reply(value){this.onmessage({data:value})}
}
globalThis.Worker=Worker;
const pipeline=new BrowserPipeline(),progress=[];
const initialize=pipeline.initialize({pack_id:'first'},{width:640},s=>progress.push(s));
const worker=Worker.last,first=worker.requests[0];
worker.reply({id:first.id,progress:'model ready'});worker.reply({id:first.id,value:true});
assert.equal(await initialize,true);assert.deepEqual(progress,['model ready']);
const request=pipeline.estimate({gray:new Uint8Array([7]),width:1,height:1,canvas:{notCloneable:()=>{}},time:0},{radius_m:500},0,()=>{});
assert.equal('canvas' in worker.requests[1].args[0],false);
pipeline.close();await assert.rejects(request,{name:'AbortError'});assert.equal(worker.terminated,true);
worker.reply({id:worker.requests[1].id,value:{accepted:true}});assert.equal(pipeline.pending.size,0);
const next=new BrowserPipeline();const failed=next.initialize({},{});Worker.last.onerror({message:'Worker failed'});await assert.rejects(failed,/Worker failed/);next.close();
