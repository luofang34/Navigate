import assert from 'node:assert/strict';
import {DataService} from '../webapp/data-service.js';
import {assetUrl} from '../webapp/asset-url.js';

for(const base of ['https://vnav.example/','https://example.github.io/Navigate/']){
  const requests=[];
  const service=new DataService(async(url,options)=>{
    requests.push([url,options]);const path=new URL(url).pathname.slice(new URL(base).pathname.length);
    if(path==='api/catalog')return new Response('No API',{status:404});
    if(path==='catalog.json')return Response.json([{id:'region',pack_id:'pack'}]);
    if(path==='packs/pack.json')return Response.json({pack_id:'pack'});
    throw Error('Unexpected request '+url);
  },base);
  assert.deepEqual(await service.request('/api/catalog'),[{id:'region',pack_id:'pack'}]);
  assert.equal(service.static,true);
  assert.deepEqual(await service.request('/api/offline-plan',{region_id:'region'}),{pack_id:'pack'});
  await assert.rejects(service.request('/api/coverage-download',{}),/Rust package service/);
  assert.equal(requests.length,3);
  assert.ok(requests.every(([url,options])=>url.startsWith(base)&&!options.method));
  assert.equal(assetUrl('/models/model.onnx',base),base+'models/model.onnx');
  assert.equal(assetUrl('/chunks/abc.bin',base),base+'chunks/abc.bin');
  assert.equal(assetUrl('https://provider.example/data',base),'https://provider.example/data');
}
const originalFetch=globalThis.fetch;
try{
  globalThis.fetch=function(url){assert.equal(this,globalThis);assert.equal(url,'https://example.test/Navigate/api/catalog');return Promise.resolve(Response.json([]));};
  assert.deepEqual(await new DataService(undefined,'https://example.test/Navigate/').request('/api/catalog'),[]);
}finally{globalThis.fetch=originalFetch;}
