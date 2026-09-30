import {decodeCameraImage} from './camera-image.js';
import {streamFile} from './http-file.js';
import {emitDatasetReport} from './qa-dataset-report.js';
import {BrowserPipeline} from './browser-pipeline.js';
import {matchingOptions} from './matching-options.js';
import {cameraForImage} from './calibration.js';
import {gray,originalFrame} from './observation.js';
import {download} from './storage.js';
const allNames=["DJI_0029_frame_1.png","DJI_0029_frame_2.png","DJI_0029_frame_3.png","DJI_0029_frame_4.png","DJI_0029_frame_5.png","DJI_0030_frame_1.png","DJI_0030_frame_2.png","DJI_0030_frame_3.png","DJI_0030_frame_4.png"];
const params=new URLSearchParams(location.search),region=params.get('region')||'naip-2864a5ec8e4abb24';
const originalNegatives=params.has('controls')||params.get('negative')==='original';
const controls=['negative-drone_image_1.png','negative-drone_image_2.png','negative-test.png'];
const available=params.has('negative')?controls:params.has('controls')?[...allNames,...controls]:allNames;
const names=params.has('case')?available.filter(name=>params.getAll('case').includes(name)):available;
const report={test:'real-image-dataset',region,mode:params.get('mode')||'balanced',model:['lightglue','hybrid'].includes(params.get('model'))?'research-'+params.get('model'):'public-loftr',started_at:new Date().toISOString(),independent_geographic_accuracy:'not measured; supplied telemetry is a placeholder',image_decode:'encoded RGB samples; display color conversion disabled',image_sampling:params.get('sampling')??'low',retrieval_image:params.get('retrieval-image')??'original',processing_scope:(originalNegatives?'Unrelated-image controls use the original full-size source files, resized to the declared suite calibration. ':params.has('negative')?'Unrelated-image controls use the existing 640 by 360 grayscale comparison fixtures, resized to the declared suite calibration. ':'')+'Independent still images. Capture times are unknown. No previous camera pose is used. Resident model and map caches remain available.',cases:[]};
const emit=phase=>emitDatasetReport(report,phase,document.querySelector('pre'));
let pipeline;
try{
 await emit('package');const response=await fetch('/api/offline-plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({region_id:region})});if(!response.ok)throw Error(await response.text());const pack=await response.json();await download(pack,()=>{});
 const options={...matchingOptions(report.mode),temporal:false,...(params.has('long-edge')?{longEdge:Number(params.get('long-edge'))}:{}),...(params.has('crop-seeds')?{cropSeedRegions:Number(params.get('crop-seeds'))}:{}),...(params.has('similarity-candidates')?{similarityCandidates:Number(params.get('similarity-candidates'))}:{}),...(params.has('reference-candidates')?{referenceCandidates:Number(params.get('reference-candidates'))}:{}),traceReplay:params.has('replay'),referenceSearch:params.get('retriever')==='camp',traceMatching:report.model==='research-lightglue'?'lightglue':'public',researchHybrid:report.model==='research-hybrid'},camera=cameraForImage(1920,1080,82.1,options.longEdge);report.options=options;report.camera=camera;report.prior={latitude:Number(params.get('lat')??pack.anchor_lat_lon[0]),longitude:Number(params.get('lon')??pack.anchor_lat_lon[1]),radius_m:Number(params.get('radius')??500),agl_m:Number(params.get('agl')??110)};
 pipeline=new BrowserPipeline(params.has('trace')?new Worker('./qa-acquisition-worker.js',{type:'module'}):report.model.startsWith('research-')?new Worker('./qa-lightglue-search-worker.js',{type:'module'}):undefined);await emit('initialize');await pipeline.initialize(pack,camera,text=>{report.initialization_progress=text;document.title=text},options);
 for(const [sequence,name] of names.entries()){
  await pipeline.beginSequence();await emit('matching '+name);const negative=name.startsWith('negative-'),url=(negative&&!originalNegatives?'./models/research-lightglue/':'./models/test-frames/')+(negative&&originalNegatives?name.replace(/^negative-/,''):name);const {bitmap,source_sha256}=await loadTestImage(url);const frame={...gray(bitmap,camera.width,camera.height,true,report.image_sampling),time:0,timing:'still image'};if(options.referenceSearch&&report.retrieval_image==='original')frame.original=await originalFrame(bitmap);bitmap.close();
  const progress=text=>{document.title=name+' · '+text};
  const result=options.traceReplay?await pipeline.call('estimate',[{gray:frame.gray,width:frame.width,height:frame.height,time:0,timing:'still image'},report.prior,sequence,{source_sha256}],progress):await pipeline.estimate(frame,report.prior,sequence,progress);
  report.cases.push({name,negative,source_sha256,camera,acquisition_trace:result.acquisition_trace,retrieval_trace:result.retrieval_trace,decision:result.decision,processing_ms:result.processing_ms,stage_ms:result.stage_ms,verification_work:result.verification_work,recovery_work:result.recovery_work,retrieval:result.retrieval,hypotheses:result.candidate_hypotheses,execution:result.execution,geometric_acceptance:result.candidate_hypotheses.some(h=>h.accepted)});await emit('finished '+name);
 }
 report.geometrically_accepted=report.cases.filter(c=>c.geometric_acceptance).length;report.total=report.cases.length;report.finished_at=new Date().toISOString();document.title=report.geometrically_accepted+'/'+report.total+' real-image geometric results';await emit('complete');
}catch(error){report.failed_phase=report.phase;report.error=String(error);report.error_stack=error.stack;await emit('failed')}finally{pipeline?.close()}

async function loadTestImage(url){
 const head=await fetch(url,{method:'HEAD',cache:'no-store'});if(!head.ok)throw Error(`Test image header request failed: ${head.status}`);
 const parts=[];for await(const part of streamFile(url,Number(head.headers.get('Content-Length'))))parts.push(part);
 const blob=new Blob(parts,{type:head.headers.get('Content-Type')??'image/png'});
 const source_sha256=[...new Uint8Array(await crypto.subtle.digest('SHA-256',await blob.arrayBuffer()))].map(v=>v.toString(16).padStart(2,'0')).join('');
 return {bitmap:await decodeCameraImage(blob),source_sha256};
}
