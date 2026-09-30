import init,* as kernel from './wasm/navigate_visual_preview.js';import {LocalMatcher} from './inference/local.js';import {SequenceImages} from './sequence-images.js';import {reconstructImageGroups} from './scene-sequence.js';import {saveImageTracks,loadImageTracks,saveSceneReconstruction,loadSceneReconstruction} from './storage.js';
const progress=text=>self.postMessage({progress:text});
self.onmessage=async({data})=>{let matcher,sequence;try{await init();const start=performance.now();matcher=new LocalMatcher({matcher:'dense'});await matcher.initialize(progress);sequence=new SequenceImages({create:()=>new kernel.SceneTracks(JSON.stringify(data.camera)),matcher,save:saveImageTracks,camera:data.camera});
 for(const [i,image] of data.images.entries()){const frame=data.observations[i];await sequence.observe(image,frame.observation_sha256,frame.sequence,text=>progress(`Frame ${i+1}/${data.images.length}: ${text}`))}
 const records=await sequence.finish(),result=await reconstructImageGroups(records,{kernel,load:loadImageTracks,loadScene:loadSceneReconstruction,save:saveSceneReconstruction,progress});
 self.postMessage({stage:'facade_sampling_probe',elapsed_ms:performance.now()-start,records,result,execution:matcher.diagnostics(),geographic_acceptance:false});
 }catch(error){self.postMessage({error:String(error),stack:error.stack})}finally{sequence?.close();await matcher?.close()}};
