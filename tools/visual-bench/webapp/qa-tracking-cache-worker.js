import {LocalMatcher} from './inference/local.js';import {get,queryBlob,sha256} from './storage.js';import {gray} from './observation.js';import {matchingOptions} from './matching-options.js';
self.onmessage=async()=>{const matcher=new LocalMatcher(matchingOptions('balanced'));try{
 const progress=progress=>self.postMessage({progress}),mission=await get('missions','b1333953386146fe958f824bc9a0ad8f'),camera=mission.view.camera;
 const frames=mission.view.frames.filter(f=>f.capture_time_ns>=139.65e9).slice(0,2),images=[];
 for(const frame of frames){const bitmap=await createImageBitmap(await queryBlob(mission,frame));try{images.push(gray(bitmap,camera.width,camera.height))}finally{bitmap.close()}}
 await matcher.initialize(progress);const keys={stage:'tracking',reference:frames[0].observation_sha256+'/query',query:frames[1].observation_sha256+'/query',progress};
 const firstStart=performance.now(),first=await matcher.matchImages(...images,keys),firstMs=performance.now()-firstStart,before=matcher.diagnostics();
 const secondStart=performance.now(),second=await matcher.matchImages(...images,keys),secondMs=performance.now()-secondStart,after=matcher.diagnostics();
 const same=JSON.stringify(first)===JSON.stringify(second);if(!same||after.match_runs!==before.match_runs||after.matching_gpu_dispatches!==before.matching_gpu_dispatches||after.tracking_pair_cache_hits!==1)throw Error('Cached pair ran inference or changed results');
 self.postMessage({done:true,stage:'tracking_pair_cache_measurement',first_ms:firstMs,cached_ms:secondMs,pairs:first.pairs.length,same,before,after,geographic_acceptance:false,source_pixel_sha256:await Promise.all(images.map(i=>sha256(i.gray)))});
}catch(error){self.postMessage({error:String(error),stack:error.stack})}finally{await matcher.close()}};
