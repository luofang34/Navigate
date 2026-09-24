import assert from 'node:assert/strict';
import {LocalizationPipeline} from '../webapp/localization.js';
import {PipelineSession} from '../webapp/pipeline-session.js';

// Only device initialization is replaced. The client, worker protocol and sequence reset run unchanged.
let resident,loads=0;
LocalizationPipeline.prototype.initialize=async function(pack,camera,progress){
 resident=this;loads++;this.pack=pack;this.camera=camera;this.renderer={};this.references={};progress('ready');
};
globalThis.Worker=class {
 constructor(url){
  globalThis.self={postMessage:data=>queueMicrotask(()=>this.onmessage({data}))};
  this.ready=import(url).then(()=>{this.dispatch=self.onmessage});
 }
 postMessage(data){this.ready.then(()=>this.dispatch({data}))}
 terminate(){}
};
const sessions=new PipelineSession(),pack={pack_id:'same-map'},camera={width:960,height:544},options={},progress=[];
const first=await sessions.acquire(pack,camera,options,value=>progress.push(value));
assert.deepEqual(progress,['ready']);
await first.beginSequence();
const matcher=resident.matcher,renderer=resident.renderer,references=resident.references;
const report={observation_sha256:'first-video-frame',sequence:0,capture_time_ns:0,candidate_hypotheses:[
 {candidate_id:2,accepted:true,position_enu_m:[0,0,110],eye_to_enu_xyzw:[0,0,0,1]},
 {candidate_id:9,accepted:true,position_enu_m:[40,0,110],eye_to_enu_xyzw:[0,0,0,1]},
]};
resident.temporal.remember(report,{gray:new Uint8Array([7]),width:1,height:1},{regional:true});
assert.equal(resident.temporal.seeds('other-frame',[0,0,110],100).length,2);
assert.equal(resident.temporal.regionalDue(0),false);
const second=await sessions.acquire({...pack},{...camera},{...options},()=>{});
assert.strictEqual(second,first);await second.beginSequence();
assert.equal(resident.temporal.previous,null,'new video releases the previous camera pixels and pose seeds');
assert.deepEqual(resident.temporal.seeds('other-frame',[0,0,110],100),[]);
assert.equal(resident.temporal.regionalDue(0),true,'a repeated timestamp must start geographic retrieval');
assert.equal(resident.temporal.mapDue(0),true,'map validation does not inherit the previous video schedule');
assert.equal(resident.temporal.lastRegionalSearchNs,null);assert.equal(resident.temporal.lastMapSearchNs,null);
assert.equal(loads,1);assert.strictEqual(resident.matcher,matcher);assert.strictEqual(resident.renderer,renderer);assert.strictEqual(resident.references,references);
assert.equal(report.candidate_hypotheses.length,2,'reset does not mutate stored geographic alternatives');
await assert.rejects(second.call('unsupported-request',[]),/Unknown localization request/);
sessions.close();
console.info('Main-app worker reset clears capture history and keeps initialized resources');
