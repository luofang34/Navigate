import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
for(const base of ['http://localhost/','https://example.github.io/Navigate/']){
  const handlers={};let selected,resources;const warnings=[];
  const cache={match:async()=>new Response('current shell'),addAll:async urls=>{resources=urls}};
  const context={URL,Response,console:{warn:(...args)=>warnings.push(args)},fetch:async()=>{throw Error('offline')},
    self:{location:{href:base+'sw.js',origin:new URL(base).origin},skipWaiting:()=>{},addEventListener:(name,fn)=>{handlers[name]=fn}},
    caches:{open:async name=>{selected=name;return cache},match:async()=>new Response('stale shell')}};
  vm.runInNewContext(await fs.readFile(new URL('../webapp/sw.js',import.meta.url),'utf8'),context);
  let install;handlers.install({waitUntil:p=>install=p});await install;
  assert.ok(resources.every(url=>url.startsWith(base)),'app shell remains under the deployment prefix');
  for(const path of ['camera-image.js','candidate-refinement.js','indexed-reference-search.js','reference-search-loading.js','inference/camp.js','inference/camp-index.js','inference/camp-pixels.js','inference/camp-storage.js','http-file.js','retrieval-poses.js','inference/refinement-patches.js','video-timing.js','pose-playback.js','track-preview.js','track-projection.js','inference/loftr.js','inference/loftr-input.js','temporal-search.js','map-check-motion.js','shortlist.js','inference/lighterglue.js','inference/lighterglue-decode.js','inference/retrieval-batch.js','inference/feature-cache.js'])assert.ok(resources.includes(base+path),'learned matcher remains available offline');
  assert.ok(resources.includes(base+'scene-sequence.js'),'offline installation includes reconstruction orchestration');
  assert.ok(resources.includes(base+'sequence-images.js'),'offline installation includes image association collection');
  assert.ok(resources.includes(base+'offline-pack.js'),'offline installation includes package integrity handling');
  assert.ok(resources.includes(base+'track-selection.js'),'offline installation includes review-path selection');
  let response;handlers.fetch({request:{method:'GET',url:base},respondWith:p=>response=p});
  assert.equal(await(await response).text(),'current shell');assert.match(selected,/^navigate-vnav-shell-v[0-9]+$/);
  for(const path of ['api/catalog','chunks/hash.bin','jobs/id','models/model.onnx','models/test-frames/large-original.png','models/manifest.json','packs/map.json','qa-dataset.js']){
    let intercepted=false;handlers.fetch({request:{method:'GET',url:base+path},respondWith:()=>intercepted=true});assert.equal(intercepted,false,path+' bypasses the app-shell cache');
  }
  const cacheFailure=Error('quota exceeded');let writes=0,background;
  cache.put=async()=>{writes++;throw cacheFailure};
  context.fetch=async()=>new Response('fresh shell');
  handlers.fetch({request:{method:'GET',url:base+'app.js'},respondWith:p=>response=p,waitUntil:p=>background=p});
  assert.equal(await(await response).text(),'fresh shell','cache failure does not replace a usable network response');
  await background;assert.equal(writes,1);assert.equal(warnings.length,1);assert.equal(warnings[0][1],cacheFailure);
  let outside=false;handlers.fetch({request:{method:'GET',url:'https://elsewhere.example/'},respondWith:()=>outside=true});assert.equal(outside,false);
}
