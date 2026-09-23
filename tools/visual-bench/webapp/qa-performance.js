import {checkGlobeContext} from './qa-globe.js';
import {checkGpuRetrieval} from './qa-gpu.js';
import {BrowserPipeline} from './browser-pipeline.js';
import {MapView,localPosition} from './map.js';
import {download,read} from './storage.js';
import init,{Preview} from './wasm/navigate_visual_preview.js';
import {ReferencePack} from './reference-pack.js';
import {cameraForImage} from './calibration.js';
import {matchingOptions} from './matching-options.js';
import {gray} from './observation.js';
const $=id=>document.getElementById(id),report={},check=(name,condition)=>{if(!condition)throw Error(name);report[name]=true};
let interactions=0;$('heartbeat').onclick=()=>{$('heartbeat').textContent=`Interaction count: ${++interactions}`};
$('run').onclick=async()=>{let truth,pipeline,raf,previous=performance.now(),gaps=[],active=false;const progress=text=>{$('progress').textContent=text};const heartbeat=now=>{if(active)gaps.push(now-previous);previous=now;raf=setTimeout(()=>heartbeat(performance.now()),16)};raf=setTimeout(()=>heartbeat(performance.now()),16);$('run').disabled=true;
 try{
  check('GPU retrieval handles partial workgroups and ties',await checkGpuRetrieval());
  const region=await(await fetch('/api/offline-plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({region_id:new URLSearchParams(location.search).get('region')||(new URLSearchParams(location.search).has('synthetic')?'naip-2864a5ec8e4abb24':'new-jersey')})})).json();await download(region,(n,t)=>progress(`Verified package ${n}/${t}`));
  const bitmap=await createImageBitmap(await(await fetch('/models/test-input.png')).blob()),options=matchingOptions('balanced'),camera=cameraForImage(bitmap.width,bitmap.height,82.1,options.longEdge);let image={...gray(bitmap,camera.width,camera.height),time:0};bitmap.close();
  const useDefaultPrior=new URLSearchParams(location.search).has('default-prior');const prior={latitude:useDefaultPrior?region.anchor_lat_lon[0]:40.5442,longitude:useDefaultPrior?region.anchor_lat_lon[1]:-74.4564,radius_m:500,agl_m:110};report.prior=prior;
  const map=new MapView($('map'));await map.load(region,camera);await map.setPose({position_enu_m:localPosition(region,prior.latitude,prior.longitude,800),eye_to_enu_xyzw:[0,0,0,1]});
  check('direct GPU presentation',$('map').dataset.presentation==='wgpu-direct');check('display resolution independent of matching',$('map').width>camera.width);
  const base=structuredClone(map.pose);map.pan(40,20);await map.draw();check('pan moves camera',Math.hypot(...map.pose.position_enu_m.map((v,i)=>v-base.position_enu_m[i]))>10);await map.reset();check('reset restores selected pose',JSON.stringify(map.pose)===JSON.stringify(base));
  map.rotate(.1,.1);await map.draw();check('look changes orientation',JSON.stringify(map.pose.eye_to_enu_xyzw)!==JSON.stringify(base.eye_to_enu_xyzw));await map.reset();
  $('map').style.width='700px';await map.draw();check('resize changes physical output',$('map').width!==800*Math.min(devicePixelRatio,2));
  await map.globe();check('globe changes camera',map.height()>10_000_000);check('globe shows land context',await checkGlobeContext(region,camera,map.pose));await map.reset();
  if(new URLSearchParams(location.search).has('synthetic')){
    const references=await ReferencePack.open(region),lat=40.544,lon=-74.456,pose={position_enu_m:localPosition(region,lat,lon,references.elevation(lat,lon)+110),eye_to_enu_xyzw:[0,0,Math.sin(.3),Math.cos(.3)]};
    truth=structuredClone(pose);
    const capture=await Preview.create(JSON.stringify(region),JSON.stringify(camera),read,false),pixels=await capture.render(JSON.stringify(pose));capture.free();
    const canvas=new OffscreenCanvas(camera.width,camera.height);canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(pixels),camera.width,camera.height),0,0);image={...gray(canvas,camera.width,camera.height),time:0};
    const blob=await canvas.convertToBlob({type:'image/jpeg',quality:.92});report.synthetic_image_base64=btoa(String.fromCharCode(...new Uint8Array(await blob.arrayBuffer())));report.observation_source='Map-derived synthetic view; shares reference data. No independent geographic validation.';
  }
  pipeline=new BrowserPipeline();await pipeline.initialize(region,camera,progress,options);active=true;previous=performance.now();
  const longTasks=[];const observer=new PerformanceObserver(list=>longTasks.push(...list.getEntries().map(e=>({start:e.startTime,duration:e.duration}))));observer.observe({type:'longtask',buffered:false});
  const result=await pipeline.estimate(image,prior,0,progress);active=false;longTasks.push(...observer.takeRecords().map(e=>({start:e.startTime,duration:e.duration})));observer.disconnect();report.main_thread_long_tasks=longTasks;report.visibility=document.visibilityState;const accepted=result.candidate_hypotheses.filter(h=>h.accepted);
  check('image produces geometric alternatives',accepted.length>0);
  if(truth){report.pose_errors=accepted.map(h=>({candidate_id:h.candidate_id,position_m:Math.hypot(...h.position_enu_m.map((v,i)=>v-truth.position_enu_m[i])),attitude_deg:2*Math.acos(Math.min(1,Math.abs(h.eye_to_enu_xyzw.reduce((sum,v,i)=>sum+v*truth.eye_to_enu_xyzw[i],0))))*180/Math.PI}));report.withheld_pose=truth;check('withheld camera pose recovered within 10 m and 1 degree',report.pose_errors.some(e=>e.position_m<10&&e.attitude_deg<1));}
  check('GPU retrieval executes',result.execution.retrieval_gpu_dispatches>0);report.execution=result.execution;check('learned GPU kernels execute',result.execution.feature_gpu_dispatches>0&&result.execution.matching_gpu_dispatches>0);check('UI event loop responds during matching',gaps.length>0&&longTasks.length===0);
  report.result={decision:result.decision,inliers:accepted.map(h=>h.inliers),rms_px:accepted.map(h=>h.reprojection_rms_px),processing_ms:result.processing_ms,execution:result.execution,retrieval:result.retrieval,stage_ms:result.stage_ms,ui_frames:gaps.length,ui_max_gap_ms:Math.max(...gaps),ui_p95_gap_ms:gaps.sort((a,b)=>a-b)[Math.floor(gaps.length*.95)]};
  const blank=gray(new OffscreenCanvas(camera.width,camera.height),camera.width,camera.height);const rejected=await pipeline.estimate({...blank,time:1},prior,1,progress);check('blank frame rejected',!rejected.accepted&&!rejected.candidate_hypotheses.some(h=>h.accepted));
  const pending=pipeline.estimate(image,prior,2,progress);pipeline.close();let cancelled=false;try{await pending}catch(e){cancelled=e.name==='AbortError'}check('cancel rejects outstanding work',cancelled);pipeline=null;
  const h=accepted[0];await map.setPose({position_enu_m:h.position_enu_m,eye_to_enu_xyzw:h.eye_to_enu_xyzw});
  report.map={width:$('map').width,height:$('map').height,submit_ms:$('map').dataset.renderMs,frames:map.frames};document.title=`PASS · ${Math.round(result.processing_ms)} ms · ${accepted.length} hypotheses`;
 }catch(e){report.error=String(e);document.title='FAIL visual regression'}finally{active=false;clearTimeout(raf);pipeline?.close();$('result').textContent=JSON.stringify(report,null,2);$('run').disabled=false;progress(report.error||'Checks complete');if(new URLSearchParams(location.search).has('report'))await fetch('/__test__/report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(report)}).catch(()=>{})}
};
if(new URLSearchParams(location.search).has('run'))$('run').click();
