import assert from 'node:assert/strict';
import {SequenceImages} from '../webapp/sequence-images.js';
import {saveImageTracks,loadImageTracks,read,verifyChunk} from '../webapp/storage.js';
const created=[],saved=[],matches=[];
function create(){const value={sources:[],freed:false,push(id,json){this.sources.push({id,pairs:JSON.parse(json)})},snapshot(){return JSON.stringify({sources:this.sources})},free(){assert.equal(this.freed,false);this.freed=true}};created.push(value);return value}
const camera={width:2,height:2},image=i=>({...camera,gray:new Uint8Array([i,0,0,0]),time:i,requested_time_s:i});
const matcher={async matchImages(a,b,keys){matches.push({a,b,keys});return {pairs:b.time===2?[]:[{reference:[0,0],query:[1,0]}],backend_identity:'replaceable-test-adapter'}}};
const session=new SequenceImages({create,matcher,camera,maxFrames:3,overlapFrames:1,save:async text=>{saved.push(JSON.parse(text));return {sha256:String(saved.length)}}});
assert.equal(await session.observe(image(0),'a',0,()=>{}),null);
const pair=await session.observe(image(1),'b',1,()=>{});assert.equal(pair.reference,'a');assert.equal(pair.query,'b');
assert.equal(await session.observe(image(1),'b',1,()=>{}),pair);assert.equal(matches.length,1,'reprocessing does not append evidence or rerun inference');
await session.observe(image(2),'c',2,()=>{});assert.equal(created[0].sources[2].pairs.length,0,'missing matches remain explicit');
await session.observe(image(3),'d',3,()=>{});assert.equal(saved.length,1);assert.equal(saved[0].stage,'image_associations');assert.equal(saved[0].geographic_acceptance,false);
assert.deepEqual(saved[0].observations.map(o=>o.observation_sha256),['a','b','c']);
assert.deepEqual(created[1].sources.map(o=>o.id),['c','d'],'adjacent groups share their declared boundary observation');
assert.deepEqual(matches.map(m=>[m.a.gray[0],m.b.gray[0]]),[[0,1],[1,2],[2,3]],'map rejection cannot erase adjacent camera pixels');
await assert.rejects(session.observe(image(0),'a',0,()=>{}),/Repeated source/);
const groups=await session.finish();assert.equal(groups.length,2);assert.equal(created.every(c=>c.freed),true);assert.equal(session.previous,null);
assert.deepEqual(saved[1].matcher_identities,['replaceable-test-adapter']);assert.match(saved[1].evidence_correlation,/share observations/);
assert.deepEqual(await session.finish(),[],'finishing twice cannot duplicate retained evidence');
const failed=new SequenceImages({create,matcher,camera,save:async()=>{throw Error('disk full')}});
await failed.observe(image(0),'a',0,()=>{});await failed.observe(image(1),'b',1,()=>{});
await assert.rejects(failed.finish(),/disk full/);assert.equal(failed.groups.length,0);failed.close();
const files=new Map();let corrupt=false;
const dir={async getDirectoryHandle(name){assert(['pilotage','chunks'].includes(name));return this},async getFileHandle(name,{create=false}={}){
 if(!files.has(name)&&!create)throw new DOMException('Missing','NotFoundError');
 return {getFile:async()=>files.get(name),createWritable:async()=>{let bytes;return {write:async value=>{bytes=value.slice()},close:async()=>{if(corrupt)bytes[0]^=1;files.set(name,new Blob([bytes]))},abort:async()=>{}}}};
}};
const previous=Object.getOwnPropertyDescriptor(globalThis,'navigator');
Object.defineProperty(globalThis,'navigator',{configurable:true,value:{storage:{getDirectory:async()=>dir}}});
try{
 const text=JSON.stringify(saved[0]),stored=await saveImageTracks(text);assert.equal(stored.geographic_acceptance,false);assert.equal(await verifyChunk(stored),true);
 assert.equal(new TextDecoder().decode(await read(stored.uri,0,stored.size)),text);
 const valid={...saved[0],graph:{observation_sha256:saved[0].observations.map(o=>o.observation_sha256),tracks:[]}};const graphRecord=await saveImageTracks(JSON.stringify(valid));assert.deepEqual(await loadImageTracks(graphRecord),valid);
 const mismatched={...valid,graph:{...valid.graph,observation_sha256:['wrong']}};await assert.rejects(loadImageTracks(await saveImageTracks(JSON.stringify(mismatched))),/identities/);
 corrupt=true;await assert.rejects(saveImageTracks(JSON.stringify(saved[1])),/checksum/);
}finally{if(previous)Object.defineProperty(globalThis,'navigator',previous);else delete globalThis.navigator}
console.info('Image groups retain source identity across map gaps, reuse matching, bound storage, and reject corrupt writes.');

const overlapSaved=[],overlapMatches=[];
const overlapSession=new SequenceImages({create,camera,maxFrames:6,overlapFrames:3,
 matcher:{async matchImages(a,b){overlapMatches.push([a.gray[0],b.gray[0]]);return {pairs:[{reference:[0,0],query:[1,0]}],backend_identity:'same-source-pairs'}}},
 save:async text=>{overlapSaved.push(JSON.parse(text));return {sha256:String(overlapSaved.length)}}});
for(let i=0;i<10;i++)await overlapSession.observe(image(i),String(i),i,()=>{});
await overlapSession.finish();
assert.deepEqual(overlapSaved.map(g=>g.observations.map(s=>s.sequence)),[[0,1,2,3,4,5],[3,4,5,6,7,8],[6,7,8,9]]);
assert.equal(overlapMatches.length,9,'overlap never repeats image inference');
assert.deepEqual(overlapSaved[1].graph.sources[0].pairs,[],'first replayed observation has no preceding pair in this group');
assert.deepEqual(overlapSaved[1].graph.sources[1].pairs,overlapSaved[0].graph.sources[4].pairs,'shared links retain exact pair proposals');
assert.deepEqual(overlapSaved[1].matcher_identities,['same-source-pairs']);
assert.throws(()=>new SequenceImages({create,matcher,camera,maxFrames:6,overlapFrames:6}),/overlap/);
