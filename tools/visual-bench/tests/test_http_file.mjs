import assert from 'node:assert/strict';
import {streamFile} from '../webapp/http-file.js';
const bytes=new Uint8Array([1,2,3,4,5,6,7,8,9,10]);
async function collect(fetcher,size=bytes.length){const parts=[];for await(const part of streamFile('/file',size,{fetcher,rangeBytes:4}))parts.push(...part);return parts}
const calls=[];
const ranged=async(url,options)=>{
 calls.push(options);assert.equal(options.cache,'no-store');
 const [,a,b]=/^bytes=(\d+)-(\d+)$/.exec(options.headers.Range),start=Number(a),end=Number(b);
 return new Response(bytes.slice(start,end+1),{status:206,headers:{'Content-Range':`bytes ${start}-${end}/${bytes.length}`}});
};
assert.deepEqual(await collect(ranged),[...bytes]);
assert.deepEqual(calls.map(c=>c.headers.Range),['bytes=0-3','bytes=4-7','bytes=8-9']);
let count=0;
assert.deepEqual(await collect(async()=>{count++;return new Response(bytes)}),[...bytes]);
assert.equal(count,1,'a server that ignores Range supplies one complete stream');
assert.deepEqual(await collect(async(url,options)=>{assert.equal(options.headers,undefined);return new Response(bytes.slice(0,3))},3),[1,2,3]);
let cancelled=false;
await assert.rejects(collect(async()=>new Response(new ReadableStream({cancel(){cancelled=true}}),{status:206,headers:{'Content-Range':'bytes 1-4/10'}})),/Invalid download response/);
assert.equal(cancelled,true,'an incorrect range is cancelled before its bytes reach the caller');
await assert.rejects(collect(async()=>new Response(bytes.slice(0,3),{status:206,headers:{'Content-Range':'bytes 0-3/10'}})),/Truncated download/);
await assert.rejects(collect(async()=>new Response(bytes.slice(0,5),{status:206,headers:{'Content-Range':'bytes 0-3/10'}})),/exceeds expected/);
count=0;
await assert.rejects(collect(async()=>++count===1?new Response(bytes.slice(0,4),{status:206,headers:{'Content-Range':'bytes 0-3/10'}}):new Response(bytes)),/Invalid download response: 200 at byte 4/);
cancelled=false;
const body=new ReadableStream({start(controller){controller.enqueue(bytes)},cancel(){cancelled=true}});
for await(const part of streamFile('/file',10,{fetcher:async()=>new Response(body)})){assert.equal(part.length,10);break}
assert.equal(cancelled,true,'a stopped consumer releases the network body');
await assert.rejects(collect(async()=>{throw Error('network failure')}),/network failure/);
await assert.rejects(collect(ranged,0),/Invalid download size/);
