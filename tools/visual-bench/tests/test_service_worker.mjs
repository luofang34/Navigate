import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
const handlers={};let selected;
const cache={match:async()=>new Response('current shell')};
const context={URL,Response,fetch:async()=>{throw Error('service stopped')},
  self:{location:{origin:'http://localhost'},addEventListener:(name,fn)=>{handlers[name]=fn}},
  caches:{open:async name=>{selected=name;return cache},match:async()=>new Response('stale shell')}};
vm.runInNewContext(await fs.readFile(new URL('../webapp/sw.js',import.meta.url),'utf8'),context);
let response;
handlers.fetch({request:{method:'GET',url:'http://localhost/'},respondWith:p=>{response=p}});
assert.equal(await (await response).text(),'current shell');
assert.equal(selected,'navigate-visual-shell-v8');
let intercepted=false;
handlers.fetch({request:{method:'GET',url:'http://localhost/api/catalog'},respondWith:()=>{intercepted=true}});
assert.equal(intercepted,false);
