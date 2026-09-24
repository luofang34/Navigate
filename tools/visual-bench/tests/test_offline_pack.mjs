import assert from 'node:assert/strict';
import {requireOfflinePack} from '../webapp/offline-pack.js';
import {sha256} from '../webapp/storage.js';

const contents=new Uint8Array([1,2,3,4]),digest=await sha256(contents);
const pack={pack_id:'selected',files:[{sha256:digest,size:contents.length}]};
const files=new Map([[digest+'.bin',new Blob([contents])]]);
const directory={
  async getDirectoryHandle(name){assert(['pilotage','chunks'].includes(name));return this},
  async getFileHandle(name){if(!files.has(name))throw new DOMException('Missing chunk','NotFoundError');return {getFile:async()=>files.get(name)}}
};
const previous=Object.getOwnPropertyDescriptor(globalThis,'navigator');
Object.defineProperty(globalThis,'navigator',{configurable:true,value:{storage:{getDirectory:async()=>directory}}});
let ready=true,invalidations=0;
const invalidate=()=>{ready=false;invalidations++};
try{
  assert.equal(await requireOfflinePack(pack,invalidate),pack);
  assert.equal(ready,true);assert.equal(invalidations,0);
  files.delete(digest+'.bin');
  await assert.rejects(requireOfflinePack(pack,invalidate),/Store this area offline/);
  assert.equal(ready,false,'missing bytes invalidate availability before processing resumes');
  files.set(digest+'.bin',new Blob([new Uint8Array([4,3,2,1])]));ready=true;
  await assert.rejects(requireOfflinePack(pack,invalidate),/missing or corrupt/);
  assert.equal(ready,false,'equal-size corrupt bytes cannot keep the package ready');
  files.set(digest+'.bin',new Blob([contents]));ready=true;
  assert.equal(await requireOfflinePack(pack,invalidate),pack);
  assert.equal(ready,true);assert.equal(invalidations,2);
  const failure=new DOMException('Storage access failed','SecurityError');
  directory.getFileHandle=async()=>{throw failure};
  await assert.rejects(requireOfflinePack(pack,invalidate),error=>error===failure);
  assert.equal(invalidations,2,'access errors keep their original context');
}finally{
  if(previous)Object.defineProperty(globalThis,'navigator',previous);else delete globalThis.navigator;
}
console.info('Missing and corrupt offline bytes invalidate availability; repaired data verifies.');
