import assert from 'node:assert/strict';
import {BrowserPipeline} from '../webapp/browser-pipeline.js';
class Worker {
  constructor(){this.requests=[];Worker.last=this}
  postMessage(value,transfer=[]){this.requests.push(value);this.transfer=transfer}
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


const lane=new BrowserPipeline(),laneWorker=Worker.last,loading=lane.initialize({},{});
await assert.rejects(lane.beginSequence(),{name:'InvalidStateError'});
assert.equal(laneWorker.requests.length,1,'initialization cannot overlap sequence mutation');
laneWorker.reply({id:laneWorker.requests[0].id,value:true});await loading;
const firstFrame=lane.estimate({gray:new Uint8Array([1]),width:1,height:1,time:1},{},1);
await assert.rejects(lane.estimate({gray:new Uint8Array([2]),width:1,height:1,time:2},{},2),{name:'InvalidStateError'});
await assert.rejects(lane.beginSequence(),{name:'InvalidStateError'});
assert.equal(laneWorker.requests.length,2,'busy evidence is not queued or cloned into the worker');
laneWorker.reply({id:laneWorker.requests[1].id,error:'matching rejected'});
await assert.rejects(firstFrame,/matching rejected/);
const current=lane.estimate({gray:new Uint8Array([3]),width:1,height:1,time:3},{},3);
assert.equal(laneWorker.requests[2].args[2],3,'host submits the current frame after completion');
laneWorker.reply({id:laneWorker.requests[2].id,value:{accepted:false}});await current;lane.close();

const transport=new BrowserPipeline(),transportWorker=Worker.last;
const post=transportWorker.postMessage;transportWorker.postMessage=()=>{throw new DOMException('Cannot clone','DataCloneError')};
await assert.rejects(transport.beginSequence(),{name:'DataCloneError'});
assert.equal(transport.pending.size,0,'a synchronous transport error releases admission');
transportWorker.postMessage=post;const recover=transport.beginSequence();
transportWorker.reply({id:transportWorker.requests[0].id,value:true});await recover;
const doomed=transport.beginSequence();transportWorker.onmessageerror();
await assert.rejects(doomed,/could not be decoded/);
await assert.rejects(transport.beginSequence(),{name:'AbortError'});
assert.equal(transportWorker.terminated,true,'a broken transport cannot accept work that would wait forever');

const suppliedWorker=new Worker(),injected=new BrowserPipeline(suppliedWorker);
const injectedCall=injected.beginSequence();
assert.equal(suppliedWorker.requests.length,1,'the supplied worker owns processing');
suppliedWorker.reply({id:suppliedWorker.requests[0].id,value:true});
assert.equal(await injectedCall,true);injected.close();
assert.equal(suppliedWorker.terminated,true,'closing the pipeline releases its supplied worker');

const sourcePipe=new BrowserPipeline(),sourceWorker=Worker.last,ready=sourcePipe.initialize({}, {},()=>{}, {referenceSearch:true});
sourceWorker.reply({id:sourceWorker.requests[0].id,value:{original_pixels:true}});await ready;assert.equal(sourcePipe.requiresOriginalImage,true);
const original={width:4,height:3,rgb:new Uint8Array(36)},sourceCall=sourcePipe.estimate({gray:new Uint8Array(4),width:2,height:2,original}, {},0);
assert.deepEqual(sourceWorker.requests[1].args[0].original,original,'the worker receives original retrieval pixels separately from the geometric observation');
sourceWorker.reply({id:sourceWorker.requests[1].id,value:{accepted:false}});await sourceCall;sourcePipe.close();

const bitmapPipe=new BrowserPipeline(),bitmapWorker=Worker.last;let bitmapClosed=0;
const bitmap={close(){bitmapClosed++}},bitmapCall=bitmapPipe.estimate({gray:new Uint8Array(4),width:2,height:2,original:{bitmap,width:4,height:3}}, {},0);
assert.deepEqual(bitmapWorker.transfer,[bitmap],'large original images move to the worker without reading RGB on the main thread');
bitmapWorker.reply({id:bitmapWorker.requests[0].id,value:{accepted:false}});await bitmapCall;assert.equal(bitmapClosed,1);
const geometricCall=bitmapPipe.refineAt({}, {gray:new Uint8Array(4),width:2,height:2,original:{bitmap}}, {}, [],()=>{}),geometricRequest=bitmapWorker.requests.at(-1);assert.equal('original' in geometricRequest.args[1],false,'geometric refinement does not resend a transferred source image');bitmapWorker.reply({id:geometricRequest.id,value:{accepted:false}});await geometricCall;bitmapPipe.close();

const noIndex=new BrowserPipeline(),noIndexWorker=Worker.last,noIndexReady=noIndex.initialize({}, {},()=>{}, {referenceSearch:true});
noIndexWorker.reply({id:noIndexWorker.requests[0].id,value:{original_pixels:false}});await noIndexReady;assert.equal(noIndex.requiresOriginalImage,false,'a missing optional index does not request large original images');noIndex.close();
