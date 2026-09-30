import assert from 'node:assert/strict';
import {downloadFiles,sha256,verifyChunk} from '../webapp/storage.js';

const source=new Uint8Array(4*1024*1024+17);source.fill(73);
const chunk={sha256:await sha256(source),size:source.length,url:'/large-model.bin'},files=new Map(),requests=[];
function writable(name){
 const parts=[],stream=new WritableStream({write:value=>parts.push(value),close:()=>files.set(name,new Blob(parts))});
 for(const method of ['write','close','abort'])stream[method]=async(...args)=>{const writer=stream.getWriter();try{return await writer[method](...args)}finally{writer.releaseLock()}};
 return stream;
}
const directory={
 async getDirectoryHandle(){return this},
 async getFileHandle(name,{create=false}={}){
  if(!files.has(name)){if(!create)throw new DOMException('Missing file','NotFoundError');files.set(name,new Blob())}
  return {getFile:async()=>files.get(name),createWritable:async()=>writable(name)};
 },
 async removeEntry(name){files.delete(name)}
};
const previousNavigator=Object.getOwnPropertyDescriptor(globalThis,'navigator'),previousFetch=globalThis.fetch;
Object.defineProperty(globalThis,'navigator',{configurable:true,value:{storage:{getDirectory:async()=>directory,estimate:async()=>({quota:128*1024*1024,usage:0})}}});
let corrupt=false;
globalThis.fetch=async(url,options)=>{
 assert.equal(options.cache,'no-store');requests.push(options.headers.Range);
 const [,a,b]=/^bytes=(\d+)-(\d+)$/.exec(options.headers.Range),start=Number(a),end=Number(b),bytes=source.slice(start,end+1);
 if(corrupt)bytes[0]^=1;
 return new Response(bytes,{status:206,headers:{'Content-Range':`bytes ${start}-${end}/${source.length}`}});
};
try{
 const progress=[];await downloadFiles([chunk],(done,total)=>progress.push([done,total]));
 assert.deepEqual(requests,['bytes=0-4194303','bytes=4194304-4194320']);
 assert.equal(await verifyChunk(chunk),true);assert.equal(files.has(chunk.sha256+'.partial'),false);
 assert.deepEqual(progress.at(-1),[source.length,source.length]);
 await downloadFiles([chunk],()=>{});assert.equal(requests.length,2,'verified OPFS bytes need no HTTP request');
 files.clear();corrupt=true;
 await assert.rejects(downloadFiles([chunk],()=>{}),/Chunk checksum failed/);
 assert.equal(files.has(chunk.sha256+'.bin'),false,'mixed or corrupt range bytes cannot become a committed file');
 assert.equal(files.has(chunk.sha256+'.partial'),false);
}finally{
 globalThis.fetch=previousFetch;
 if(previousNavigator)Object.defineProperty(globalThis,'navigator',previousNavigator);else delete globalThis.navigator;
}
