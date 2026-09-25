import {get,queryBlob} from './storage.js';
import {BrowserPipeline} from './browser-pipeline.js';
import {gray} from './observation.js';
import {matchingOptions} from './matching-options.js';
import {refineSequenceGaps} from './sequence-refinement.js';
import {trackBranches} from './track-preview.js';
import {playbackPose} from './pose-playback.js';
import {toGlobePose} from './geography.js';
const report={test:'main-application-video-track',source:'map-derived translated video; UI and geometric control, not independent DJI accuracy'},frame=document.querySelector('iframe');
const check=(name,value)=>{if(!value)throw Error(name);report[name]=true};
const until=(target,test)=>new Promise(resolve=>{const observer=new MutationObserver(()=>{if(test()){observer.disconnect();resolve()}});observer.observe(target,{subtree:true,attributes:true,childList:true,characterData:true});if(test()){observer.disconnect();resolve()}});
const event=(target,name,test=()=>true)=>new Promise(resolve=>{const receive=value=>{if(!test(value))return;target.removeEventListener(name,receive);resolve(value)};target.addEventListener(name,receive)});
async function ready(){if(!frame.contentDocument||frame.contentDocument.URL==='about:blank'||frame.contentDocument.readyState==='loading')await event(frame,'load');const w=frame.contentWindow;if(w.document.documentElement.dataset.visualReady!=='true')await event(w,'visual-ready');return w}
async function emit(phase){report.phase=phase;document.querySelector('pre').textContent=JSON.stringify(report,null,2)}
async function run(){
 const w=await ready(),d=w.document,$=id=>d.getElementById(id),map=$('map'),video=$('video');
 $('region').value='naip-2864a5ec8e4abb24';await $('region').onchange();await $('download').onclick();
 check('reference overview keeps browsing clearance',JSON.parse(map.dataset.clearance).constrained);
 const response=await fetch('./models/test-track.mp4');if(!response.ok)throw Error('Missing generated video fixture');
 const transfer=new w.DataTransfer();transfer.items.add(new w.File([await response.arrayBuffer()],'map-derived-track.mp4',{type:'video/mp4'}));$('input').files=transfer.files;await $('input').onchange();
 check('whole video is the default',$('video-mode').value==='whole');check('projection is globe',map.dataset.projection==='globe');
 $('period').value='.19';$('max-frames').value='1';
 await emit('processing whole video');const processing=$('locate').onclick();
 check('original video controls remain enabled during processing',video.controls&&!video.hidden);
 await until(d.body,()=>!$('cancel').disabled||!$('locate').disabled);report.initial_status=$('run-status').textContent;check('processing entered the worker',!$('cancel').disabled);
 const decoding=new Promise(resolve=>{let count=0;const next=()=>video.requestVideoFrameCallback(()=>{count++;if(count===3)resolve(count);else next()});next()});
 await video.play();report.decoded_frames_during_processing=await decoding;video.pause();
 check('matching still runs while original video plays',!$('cancel').disabled);
 const oldFrames=Number(map.dataset.frames),rendered=event(map,'render');$('zoom-in').click();await rendered;
 check('map moves during matching',Number(map.dataset.frames)>oldFrames);
 await until(d.body,()=>$('frames').options.length>=2||$('cancel').disabled);
 if(!$('cancel').disabled){
  $('frames').value='0';await $('frames').onchange();
  check('frame selection remains available during matching',!$('frames').disabled&&!$('frame-time').disabled);
  check('hypothesis selection remains available during matching',!$('hypotheses').disabled);
  const reviewed=JSON.parse($('details').textContent).observation.observation_sha256;
  const count=$('frames').options.length;await until(d.body,()=>$('frames').options.length>count||$('cancel').disabled);
  if(!$('cancel').disabled)check('later results preserve the reviewed frame',JSON.parse($('details').textContent).observation.observation_sha256===reviewed);
 }
 await processing;
 const mission=await get('missions',$('missions').value);check('whole video saves more than the custom sequence limit',mission?.view.frames.length>1);
 check('comparison panes preserve camera aspect',Math.abs(map.getBoundingClientRect().width/map.getBoundingClientRect().height-mission.view.camera.width/mission.view.camera.height)<.01);
 check('completed video opens in overview',JSON.parse(map.dataset.clearance).constrained&&!$('track-follow').checked);
 check('observations use decoded timestamps',mission.view.frames.every(f=>f.timing_scope==='browser decoded frame presentation timestamp'));
 check('seek requests remain distinct from decoded frame timestamps',mission.view.frames.some(f=>Math.abs(f.requested_time_s-f.capture_time_ns/1e9)>.01));
 report.frames=mission.view.frames.map(f=>({time:f.capture_time_ns/1e9,decision:f.decision,supported:f.candidate_hypotheses.filter(h=>h.accepted||h.tracking_supported).length}));
 check('each control frame has supported geometry',report.frames.every(f=>f.supported>0));check('original video stays visible with the saved track',!video.hidden&&$('query').hidden);
 const branches=trackBranches(mission.view.frames),interval=[...branches.values()].map(samples=>({samples,pair:samples.findIndex((s,i)=>samples[i+1]&&samples[i+1].time-s.time>.15&&playbackPose(samples,(s.time+samples[i+1].time)/2))})).find(value=>value.pair>=0);
 if(!interval)throw Error('No supported playback interval');
 const {samples:intervalSamples,pair}=interval,time=(intervalSamples[pair].time+intervalSamples[pair+1].time)/2,seeked=event(video,'seeked'),decoded=new Promise(resolve=>video.requestVideoFrameCallback((_,metadata)=>resolve(metadata.mediaTime)));video.currentTime=time;await seeked;const decodedTime=await decoded;report.follow_requested_time=time;report.follow_decoded_time=decodedTime;
 const samples=[...branches.values()].find(samples=>playbackPose(samples,decodedTime));if(!samples)throw Error('No supported pose at decoded time');
 check('follow check uses an interior decoded time',playbackPose(samples,decodedTime).interpolated);
 const presented=event(map,'render');$('track-follow').checked=true;$('track-follow').dispatchEvent(new w.Event('change'));const render=await presented;
 const pack=await get('packs',mission.pack_id),expected=toGlobePose(pack,playbackPose(samples,decodedTime).pose);
 report.follow_position_error_m=Math.hypot(...expected.position_enu_m.map((v,i)=>v-render.detail.pose.position_enu_m[i]));
 check('follow uses the interpolated pose at the decoded video timestamp',report.follow_position_error_m<.05);
 check('camera view is available',!$('track-camera').disabled);
 const zoomRender=event(map,'render');$('zoom-out').click();await zoomRender;check('manual movement releases camera follow',!$('track-follow').checked);
 check('track overlay shares the rendered canvas dimensions',$('track-overlay').width===map.width&&$('track-overlay').height===map.height);
 check('camera control labels fit their buttons',['track-overview','track-camera'].every(id=>$(id).scrollWidth<=$(id).clientWidth));
 $('latitude').value='0';$('longitude').value='0';$('radius').value='1';
 await $('missions').onchange();check('saved track replay exposes source video attachment',!$('attach-video').disabled);
 check('saved track restores its own precise prior',Number($('latitude').value)===mission.view.prior.latitude&&Number($('longitude').value)===mission.view.prior.longitude&&Number($('radius').value)===mission.view.prior.radius_m);
 checkOverlayRedraw(w,$);
 video.hidden=true;
 const cameraRender=event(map,'render');$('track-camera').click();await cameraRender;
 check('camera button labels a saved still view',$('map-label').textContent.includes('CAMERA POSE')||$('map-label').textContent.includes('SELECTED HYPOTHESIS'));
 const overviewRender=event(map,'viewchange',e=>e.detail==='overview');$('track-overview').click();await overviewRender;
 check('overview label returns after a saved still camera view',$('map-label').textContent==='TRACK OVERVIEW');
 video.hidden=false;
 const attachmentPose=JSON.stringify(JSON.parse($('map').dataset.clearance).pose);
 const attach=new w.DataTransfer();attach.items.add(transfer.files[0]);$('source-video').files=attach.files;await $('source-video').onchange();
 check('saved track reattaches its source video without matching',!video.hidden&&$('query').hidden&&$('run-status').textContent==='Source video attached. Saved poses are unchanged.');
 check('reattachment preserves the overview camera',JSON.stringify(JSON.parse($('map').dataset.clearance).pose)===attachmentPose);
 check('reattachment preserves stored hypotheses',JSON.stringify((await get('missions',mission.id)).view.frames)===JSON.stringify(mission.view.frames));
 const other=await fetch('./models/test-video.mp4'),wrong=new w.DataTransfer();wrong.items.add(new w.File([await other.arrayBuffer()],'unrelated.mp4',{type:'video/mp4'}));$('source-video').files=wrong.files;await $('source-video').onchange();
 check('mismatched replay video is rejected',$('run-status').textContent.includes('does not match the saved samples'));
 check('mismatched video keeps the current track and source',!video.hidden&&$('missions').value===mission.id&&$('input-name').textContent.includes('map-derived-track.mp4'));

 await checkBackwardRecovery(mission,pack);
 await emit('track preview passed');
 await $('region').onchange();check('region change clears the old camera track',$('track-status').textContent===''&&$('track-camera').disabled);
 check('region change removes all track pixels',overlayPixels($).every((v,i)=>i%4!==3||v===0));
}
function overlayPixels($){const canvas=$('track-overlay');return canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data}
function checkOverlayRedraw(w,$){
 const alternatives=$('track-alternatives'),toggle=value=>{alternatives.checked=value;alternatives.dispatchEvent(new w.Event('change'))};
 // Readback can change the canvas raster backend. Compare a stable backend.
 for(let i=0;i<4;i++){toggle(false);overlayPixels($)}
 const selected=overlayPixels($);
 check('selected overview draws a visible track',selected.some((v,i)=>i%4===3&&v>0));
 for(let i=0;i<3;i++){
  toggle(true);overlayPixels($);toggle(false);
  check('hiding alternatives restores the exact selected track',overlayPixels($).every((v,j)=>v===selected[j]));
  toggle(false);check('repeated redraws leave no stale track pixels',overlayPixels($).every((v,j)=>v===selected[j]));
 }
}
async function checkBackwardRecovery(mission,pack){
 await emit('checking backward gap refinement');
 const pipeline=new BrowserPipeline(),original=mission.view.frames.slice(0,2),frames=[{...original[0],accepted:false,decision:'rejected',candidate_hypotheses:[]},original[1]];
 const image=async index=>{const bitmap=await createImageBitmap(await queryBlob(mission,original[index]));try{return {...gray(bitmap,mission.view.camera.width,mission.view.camera.height),time:original[index].capture_time_ns/1e9,requested_time_s:original[index].requested_time_s,timing:original[index].timing_scope}}finally{bitmap.close()}};
 try{
  await pipeline.initialize(pack,mission.view.camera,()=>{},matchingOptions('balanced'));
  const recovered=await refineSequenceGaps(frames,{image,track:(reference,current,sequence,progress)=>pipeline.trackFrom(reference,current,mission.view.prior,sequence,progress)});
  check('a withheld control frame is recovered from its later observation',recovered===1&&frames[0].candidate_hypotheses.some(h=>h.tracking_supported));
  check('backward recovery preserves observation identity',frames[0].observation_sha256===original[0].observation_sha256);
  check('backward recovery stays conditional',!frames[0].accepted&&frames[0].decision==='relative_tracking'&&frames[0].candidate_hypotheses.every(h=>!h.accepted));
  check('backward recovery preserves the original attempt',frames[0].sequence_refinement.previous_attempt.decision==='rejected');
  const motion=await pipeline.checkMotion({report:original[1],image:await image(1)},frames[0],await image(0),mission.view.prior,frames[0].candidate_hypotheses.filter(h=>h.tracking_supported),()=>{});
  check('Rust fixed-pose checks run through the browser worker',motion.checks.some(check=>check.consistent&&check.inliers>=20));
  check('motion checks preserve evidence identities',motion.observation_sha256===original[0].observation_sha256&&motion.reference_observation_sha256===original[1].observation_sha256);
  const referenceImage=await image(1);referenceImage.gray[0]^=255;let rejected=false;
  try{await pipeline.trackFrom({report:original[1],image:referenceImage},await image(0),mission.view.prior,original[0].sequence,()=>{})}catch(error){rejected=/pixels do not match/.test(String(error))}
  check('backward recovery rejects changed reference pixels',rejected);
  report.backward_execution=frames[0].execution;
 }finally{pipeline.close()}
}
let timer;try{await Promise.race([run(),new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('Video track regression timed out')),420000)})]);document.title='PASS main application video track';await emit('complete')}catch(e){report.error=String(e);document.title='FAIL main application video track';await emit('failed')}finally{clearTimeout(timer)}
