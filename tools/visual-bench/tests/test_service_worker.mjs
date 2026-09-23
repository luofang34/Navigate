import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
for(const base of ['http://localhost/','https://example.github.io/Navigate/']){
  const handlers={};let selected,resources;
  const cache={match:async()=>new Response('current shell'),addAll:async urls=>{resources=urls}};
  const context={URL,Response,fetch:async()=>{throw Error('offline')},
    self:{location:{href:base+'sw.js',origin:new URL(base).origin},skipWaiting:()=>{},addEventListener:(name,fn)=>{handlers[name]=fn}},
    caches:{open:async name=>{selected=name;return cache},match:async()=>new Response('stale shell')}};
  vm.runInNewContext(await fs.readFile(new URL('../webapp/sw.js',import.meta.url),'utf8'),context);
  let install;handlers.install({waitUntil:p=>install=p});await install;
  assert.ok(resources.every(url=>url.startsWith(base)),'app shell remains under the deployment prefix');
  let response;handlers.fetch({request:{method:'GET',url:base},respondWith:p=>response=p});
  assert.equal(await(await response).text(),'current shell');assert.equal(selected,'navigate-vnav-shell-v16');
  for(const path of ['api/catalog','chunks/hash.bin','jobs/id','models/model.onnx']){
    let intercepted=false;handlers.fetch({request:{method:'GET',url:base+path},respondWith:()=>intercepted=true});assert.equal(intercepted,false,path+' bypasses the app-shell cache');
  }
  let outside=false;handlers.fetch({request:{method:'GET',url:'https://elsewhere.example/'},respondWith:()=>outside=true});assert.equal(outside,false);
}
