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
await assert.rejects(pipeline.beginSequence(),{name:'AbortError'});assert.equal(pipeline.pending.size,0,'a cancelled pipeline cannot wait forever on its terminated worker');
worker.reply({id:worker.requests[1].id,value:{accepted:true}});assert.equal(pipeline.pending.size,0);
const next=new BrowserPipeline();const failed=next.initialize({},{});Worker.last.onerror({message:'Worker failed'});await assert.rejects(failed,/Worker failed/);next.close();

const reverse=new BrowserPipeline(),reverseWorker=Worker.last;
const reverseCall=reverse.trackFrom({report:{sequence:3},image:{gray:new Uint8Array([3]),width:1,height:1,canvas:{notCloneable(){}}}},{gray:new Uint8Array([2]),width:1,height:1,time:2,canvas:{}},{radius_m:10},2,()=>{});
const message=reverseWorker.requests[0];assert.equal(message.method,'trackFrom');assert.equal('canvas' in message.args[0].image,false);assert.equal('canvas' in message.args[1],false);reverseWorker.reply({id:message.id,value:{accepted:false,decision:'relative_tracking'}});assert.equal((await reverseCall).accepted,false);reverse.close();

const refine=new BrowserPipeline(),refineWorker=Worker.last,refineCall=refine.refineAt({sequence:3},{gray:new Uint8Array([3]),width:1,height:1,time:3,canvas:{}},{radius_m:10},[{candidate_id:7}],()=>{});
const refinement=refineWorker.requests[0];assert.equal(refinement.method,'refineAt');assert.equal('canvas' in refinement.args[1],false);refineWorker.reply({id:refinement.id,value:{accepted:false,decision:'unresolved'}});assert.equal((await refineCall).accepted,false);refine.close();

const motion=new BrowserPipeline(),motionWorker=Worker.last,motionCall=motion.checkMotion({report:{sequence:4},image:{gray:new Uint8Array([4]),width:1,height:1,canvas:{}}},{sequence:3},{gray:new Uint8Array([3]),width:1,height:1,time:3,canvas:{}},{radius_m:10},[{candidate_id:7}],()=>{});
const check=motionWorker.requests[0];assert.equal(check.method,'checkMotion');assert.equal('canvas' in check.args[0].image,false);assert.equal('canvas' in check.args[2],false);motionWorker.reply({id:check.id,value:{checks:[{candidate_id:7,consistent:false}]}});assert.equal((await motionCall).checks[0].consistent,false);motion.close();

const fresh=new BrowserPipeline(),freshWorker=Worker.last,restarted=fresh.beginSequence(),reset=freshWorker.requests[0];assert.equal(reset.method,'beginSequence');assert.deepEqual(reset.args,[]);freshWorker.reply({id:reset.id,value:true});assert.equal(await restarted,true);fresh.close();

const ending=new BrowserPipeline(),endingWorker=Worker.last,finished=ending.finishSequence(),finish=endingWorker.requests[0];
assert.equal(finish.method,'finishSequence');endingWorker.reply({id:finish.id,value:[{stage:'image_associations',geographic_acceptance:false}]});assert.equal((await finished)[0].geographic_acceptance,false);ending.close();

const scenes=new BrowserPipeline(),sceneWorker=Worker.last,sceneCall=scenes.reconstructSequence([{sha256:'group'}],()=>{}),sceneRequest=sceneWorker.requests[0];
assert.equal(sceneRequest.method,'reconstructSequence');sceneWorker.reply({id:sceneRequest.id,value:{groups:[],geographic_acceptance:false}});assert.equal((await sceneCall).geographic_acceptance,false);scenes.close();
