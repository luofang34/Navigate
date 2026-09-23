import assert from 'node:assert/strict';
import {DataService} from '../webapp/data-service.js';
const requests=[];
const service=new DataService(async(url,options)=>{requests.push([url,options]);if(url==='/api/catalog')return new Response('No API',{status:404});if(url==='/catalog.json')return Response.json([{id:'region',pack_id:'pack'}]);if(url==='/packs/pack.json')return Response.json({pack_id:'pack'});throw Error('Unexpected request')});
assert.deepEqual(await service.request('/api/catalog'),[{id:'region',pack_id:'pack'}]);
assert.equal(service.static,true);
assert.deepEqual(await service.request('/api/offline-plan',{region_id:'region'}),{pack_id:'pack'});
await assert.rejects(service.request('/api/coverage-download',{}),/Rust package service/);
assert.equal(requests.length,3);
assert.ok(requests.every(([,options])=>!options.method),'static deployment uses file GET requests');

const originalFetch=globalThis.fetch;
try {
  globalThis.fetch=function(url){assert.equal(this,globalThis,'native browser fetch requires the global receiver');assert.equal(url,'/api/catalog');return Promise.resolve(Response.json([]));};
  assert.deepEqual(await new DataService().request('/api/catalog'),[]);
} finally {globalThis.fetch=originalFetch;}
