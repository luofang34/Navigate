import {BrowserPipeline} from './browser-pipeline.js';
import {matchingOptions} from './matching-options.js';
import {cameraForImage} from './calibration.js';
import {gray} from './observation.js';
import {download} from './storage.js';
const names=["DJI_0029_frame_1.png","DJI_0029_frame_2.png","DJI_0029_frame_3.png","DJI_0029_frame_4.png","DJI_0029_frame_5.png","DJI_0030_frame_1.png","DJI_0030_frame_2.png","DJI_0030_frame_3.png","DJI_0030_frame_4.png"];
const params=new URLSearchParams(location.search),region=params.get('region')||'naip-2864a5ec8e4abb24';
const report={test:'real-image-dataset',region,mode:params.get('mode')||'balanced',started_at:new Date().toISOString(),independent_geographic_accuracy:'not measured; supplied telemetry is a placeholder',cases:[]};
async function emit(phase){report.phase=phase;document.querySelector('pre').textContent=JSON.stringify(report,null,2);await fetch('/__test__/report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(report)})}
let pipeline;
try{
 await emit('package');const response=await fetch('/api/offline-plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({region_id:region})});if(!response.ok)throw Error(await response.text());const pack=await response.json();await download(pack,()=>{});
 const options=matchingOptions(report.mode),camera=cameraForImage(1920,1080,82.1,options.longEdge);report.camera=camera;report.prior={latitude:Number(params.get('lat')??pack.anchor_lat_lon[0]),longitude:Number(params.get('lon')??pack.anchor_lat_lon[1]),radius_m:Number(params.get('radius')??500),agl_m:Number(params.get('agl')??110)};
 pipeline=new BrowserPipeline();await emit('initialize');await pipeline.initialize(pack,camera,()=>{},options);
 for(const [sequence,name] of names.entries()){
  await emit('matching '+name);const bitmap=await createImageBitmap(await(await fetch('./models/test-frames/'+name)).blob());const frame={...gray(bitmap,camera.width,camera.height),time:sequence};bitmap.close();
  const result=await pipeline.estimate(frame,report.prior,sequence,()=>{});
  report.cases.push({name,decision:result.decision,processing_ms:result.processing_ms,retrieval:result.retrieval,hypotheses:result.candidate_hypotheses,execution:result.execution,geometric_acceptance:result.candidate_hypotheses.some(h=>h.accepted)});await emit('finished '+name);
 }
 report.geometrically_accepted=report.cases.filter(c=>c.geometric_acceptance).length;report.total=report.cases.length;report.finished_at=new Date().toISOString();document.title=report.geometrically_accepted+'/'+report.total+' real-image geometric results';await emit('complete');
}catch(error){report.error=String(error);await emit('failed')}finally{pipeline?.close()}
