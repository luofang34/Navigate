import {requireOfflinePack} from './offline-pack.js';
import {replaySeekTime} from './video-timing.js';
import {checkSavedVideo} from './saved-video.js';
import {TrackPreview} from './track-preview.js';
import {selectedInputs,sequenceTrack,videoTimes} from './input-sequence.js';
import {assetUrl} from './asset-url.js';
import {missionSummary,resultSummary,executionSummary} from './result-summary.js';
import {DataService} from './data-service.js';
import {CoverageLoader,viewCoverage} from './dynamic-coverage.js';
import * as storage from './storage.js';
import {MapView,localPosition} from './map.js';
import {hypotheses,frameLabel} from './hypotheses.js';
import {PipelineSession} from './pipeline-session.js';
import {openInput,frameAt,gray} from './observation.js';
import {cameraForImage} from './calibration.js';
import {matchingOptions} from './matching-options.js';
const $=id=>document.getElementById(id);const map=new MapView($('map'));let regions=[],region,pack,mission,input,queryURL,loadedCamera,busy=false,media,baseMapLabel,activePipeline,mediaLoading=false,mediaMissionId=null;let inputFiles=[];
const trackPreview=new TrackPreview(map,$('video'),{overlay:$('track-overlay'),follow:$('track-follow'),camera:$('track-camera'),overview:$('track-overview'),status:$('track-status'),alternatives:$('track-alternatives'),selection:playbackSelection});
const pipelineSession=new PipelineSession();
window.addEventListener('pagehide',()=>{pipelineSession.close();trackPreview.close()});
const formatBytes=n=>(n/1024/1024).toFixed(1)+' MB';
const status=text=>$('run-status').textContent=text;
function error(value){status(String(value));console.error(value)}
const dataService=new DataService();const request=(url,value)=>dataService.request(url,value);
async function storageStatus(){const e=await navigator.storage.estimate(),persisted=await navigator.storage.persisted();$('storage-status').textContent=`${formatBytes(e.usage||0)} used / ${formatBytes(e.quota||0)} quota · ${persisted?'Persistent permission granted':'Best effort storage'}`}
function drawPrior(){if(!region)return;const [w,s,e,n]=region.bounds;const lat=+$('latitude').value,lon=+$('longitude').value;const merc=v=>Math.asinh(Math.tan(v*Math.PI/180));$('prior-dot').style.left=(lon-w)/(e-w)*100+'%';$('prior-dot').style.top=(merc(n)-merc(lat))/(merc(n)-merc(s))*100+'%'}
function enable(){$('attach-video').disabled=busy||mediaLoading||!mission||mission.view.frames.some(f=>f.timing_scope==='still image');$('export-track').disabled=busy||!mission;if(busy)for(const id of ['missions','frames','hypotheses'])$(id).disabled=true;else{$('missions').disabled=false;$('frames').disabled=!mission;$('hypotheses').disabled=!$('hypotheses').options.length}$('cancel').disabled=!activePipeline;$('video').controls=true;for(const id of ['region','latitude','longitude','radius','agl','fov','input','period','max-frames','download','frame-time','video-mode','matching-quality','fetch-coverage'])$(id).disabled=busy;const ready=pack&&region&&pack.pack_id===region.pack_id;$('download').disabled=busy||!region;$('input').disabled=busy||mediaLoading;$('locate').disabled=busy||mediaLoading||!(ready&&input&&media);$('track-overview').disabled=$('reset').disabled=$('zoom-in').disabled=$('zoom-out').disabled=!map.preview;$('fetch-coverage').disabled=busy||dataService.static}
async function chooseRegion(preservePrior=false){trackPreview.clear();region=regions.find(r=>r.id===$('region').value);if(!region){$('pack-status').textContent='Download a new area or route to begin.';enable();return}pipelineSession.close();pack=null;mission=null;$('frames').replaceChildren();$('hypotheses').replaceChildren();$('frames').disabled=$('hypotheses').disabled=true;$('result').replaceChildren();$('result').textContent='No observation selected for this region.';$('overview').src=assetUrl(region.thumbnail);if(!preservePrior)[$('latitude').value,$('longitude').value]=region.anchor_lat_lon;drawPrior();enable();
  $('pack-status').textContent=`${formatBytes(region.bytes)} · prepared imagery and terrain`;
  const saved=await storage.get('packs',region.pack_id);
  if(saved&&await storage.verifyPack(saved)){pack=saved;$('pack-status').textContent=`Verified offline package · ${formatBytes(region.bytes)}`;await openPrior()}
  else if(saved)$('pack-status').textContent='Offline data is missing or corrupt. Download to repair it.';
}
function camera(){const source=media?.source,width=media?.type==='video'?source.videoWidth:source?.width||16,height=media?.type==='video'?source.videoHeight:source?.height||9;return cameraForImage(width,height,+$('fov').value,matchingOptions($('matching-quality').value).longEdge)}
async function loadMap(cam){document.querySelector('.views').style.setProperty('--camera-aspect',cam.width+'/'+cam.height);map.setMinimumClearance(+$('agl').value);if(map.preview&&map.pack_id===pack.pack_id)await map.setCalibration(cam);else await map.load(pack,cam);loadedCamera=cam;$('map-empty').hidden=true;enable()}
async function openPrior(){await loadMap(camera());await map.setPose({position_enu_m:localPosition(pack,+$('latitude').value-.0035,+$('longitude').value,800),eye_to_enu_xyzw:[Math.sin(.275),0,0,Math.cos(.275)]},{constrain:true});baseMapLabel=$('map-label').textContent='REFERENCE AREA';$('attribution').textContent=pack.attribution;enable()}
$('region').onchange=()=>chooseRegion().catch(error);
$('overview').onclick=e=>{const b=e.target.getBoundingClientRect(),[w,s,east,n]=region.bounds;const mx=Math.asinh(Math.tan(n*Math.PI/180)),my=Math.asinh(Math.tan(s*Math.PI/180));$('longitude').value=(w+(east-w)*(e.clientX-b.left)/b.width).toFixed(7);$('latitude').value=(Math.atan(Math.sinh(mx+(my-mx)*(e.clientY-b.top)/b.height))*180/Math.PI).toFixed(7);drawPrior()};
for(const id of ['latitude','longitude'])$(id).onchange=drawPrior;
$('download').onclick=async()=>{const button=$('download');busy=true;enable();try{const plan=await request('/api/offline-plan',{region_id:region.id});$('pack-status').textContent='Downloading and checking data…';await storage.download(plan,(done,total)=>{$('progress').value=done/total;$('pack-status').textContent=`${formatBytes(done)} / ${formatBytes(total)}`});pack=plan;$('pack-status').textContent='Available offline';await storageStatus();await openPrior()}catch(e){error(e);$('pack-status').textContent=pack?'Package verified; map initialization failed.':'Package incomplete. Retry to resume by chunk.'}finally{busy=false;enable()}};
$('persist').onclick=async()=>{await navigator.storage.persist();await storageStatus()};
function clearObservation(message,{preserveTrack=false}={}){
  mission=null;if(!preserveTrack)trackPreview.clear();$('export-track').disabled=true;$('missions').value='';$('frames').replaceChildren();$('hypotheses').replaceChildren();
  $('result').classList.remove('rejected');$('result').textContent=message;
  $('details').textContent='';$('compute-status').textContent='';
  baseMapLabel=$('map-label').textContent='REFERENCE VIEW · NO CAMERA POSE FOR THIS OBSERVATION';
  enable();
}
$('input').onchange=async()=>{mediaMissionId=null;mediaLoading=true;clearObservation('No estimate for this observation.');status('');$('query').removeAttribute('src');$('query-empty').hidden=false;enable();try{media?.close();inputFiles=selectedInputs($('input').files);input=inputFiles[0];media=null;$('input-name').textContent=input?(inputFiles.length>1?`${inputFiles.length} images · filename order; flight times unknown`:`${input.name} · ${formatBytes(input.size)}`):'No file selected';if(!input){enable();return}$('video').hidden=!input.type.startsWith('video/');media=await openInput(input,input.type.startsWith('video/')?document.createElement('video'):$('video'));if(media.type==='video'){$('video').src=media.source.src;$('video').load()}$('video-controls').hidden=media.type!=='video';if(media.type==='video'){$('video').hidden=false;$('frame-time').max=Math.max(0,media.duration-.04);$('frame-time').value=0;$('frame-time-label').textContent=`0.00 / ${media.duration.toFixed(2)} s`}else $('video').hidden=true;await previewInput();$('query').hidden=media.type==='video';enable()}catch(e){input=null;error(e)}finally{mediaLoading=false;enable()}};
async function previewInput(){if(!media)return;if(media.type==='video'&&Number.isFinite($('video').duration))$('video').currentTime=+$('frame-time').value||0;clearObservation('No estimate for this observation.');status('');const frame=await frameAt(media,+$('frame-time').value||0,camera());if(queryURL)URL.revokeObjectURL(queryURL);queryURL=URL.createObjectURL(frame.blob);$('query').src=queryURL;$('query-empty').hidden=true}
$('matching-quality').onchange=()=>previewInput().catch(error);
$('frame-time').onchange=()=>{if(mission&&media?.type==='video'){$('video').currentTime=+$('frame-time').value;return} $('frame-time-label').textContent=`${(+$('frame-time').value).toFixed(2)} / ${media.duration.toFixed(2)} s`;previewInput().catch(error)};
function priorValues(){return {region_id:region.id,pack_id:pack.pack_id,latitude:+$('latitude').value,longitude:+$('longitude').value,radius_m:+$('radius').value,agl_m:+$('agl').value,fov_deg:+$('fov').value,sample_period:+$('period').value,max_frames:+$('max-frames').value}}
$('locate').onclick=async()=>{const selectedPack=pack,selectedPrior=priorValues(),cam=camera();let pipeline;busy=true;clearObservation('Preparing this observation.');enable();try{
  if(!media)throw Error('Select a decoded image or video first');await requireOfflinePack(pack,()=>{pack=null;pipelineSession.close();$('pack-status').textContent='Offline data is missing or corrupt. Store this area offline to repair it.';enable()});
  await loadMap(cam);const loading=pipelineSession.acquire(selectedPack,cam,matchingOptions($('matching-quality').value),status);activePipeline=pipelineSession.current;enable();pipeline=await loading;await pipeline.beginSequence();
  const start=media.type==='video'?+$('frame-time').value:0,times=media.type==='image'?inputFiles.map(()=>0):videoTimes(media.duration,start,{mode:$('video-mode').value,period:selectedPrior.sample_period,maxFrames:selectedPrior.max_frames});const sourceAspect=media.type==='image'?media.source.width/media.source.height:null;
  const frames=[],blobs=[];
  for(let i=0;i<times.length;i++){if(sourceAspect!==null){media.close();media=await openInput(inputFiles[i],$('video'));if(Math.abs(media.source.width/media.source.height-sourceAspect)>1e-6)throw Error('Sequence images must have the same aspect ratio. Process this image separately.')}clearObservation(`Processing frame ${i+1}/${times.length}. No estimate for this frame.`,{preserveTrack:true});const observation=await frameAt(media,times[i],cam);if(queryURL)URL.revokeObjectURL(queryURL);queryURL=URL.createObjectURL(observation.blob);$('query').src=queryURL;$('query-empty').hidden=true;const result=await pipeline.estimate(observation,selectedPrior,i,s=>status(`Frame ${i+1}/${times.length} · ${s}`));result.input_name=sourceAspect!==null?inputFiles[i].name:input.name;frames.push(result);blobs.push(observation.blob);trackPreview.setFrames(frames,{period:selectedPrior.sample_period});await showHypothesis(result,hypotheses(result)[0],{moveCamera:media.type!=='video'});$('details').textContent=JSON.stringify({pack_id:selectedPack.pack_id,camera:cam,observation:result},null,2)}
  mission=await storage.saveLocalMission({camera:cam,prior:selectedPrior,frames,input:{name:input.name,names:inputFiles.map(f=>f.name),size:inputFiles.reduce((n,f)=>n+f.size,0),processing:'browser-local',calibration:'assumed full-width 4:3 sensor crop; not independently calibrated'}},selectedPack,blobs);mediaMissionId=mission.id;await refreshMissions(mission.id);await showMission(mission);
  status(missionSummary(frames));await storageStatus();
}catch(e){pipelineSession.close();error(e)}finally{activePipeline=null;busy=false;enable()}};
$('cancel').onclick=()=>{pipelineSession.close();status('Processing cancelled. No partial result was accepted.')};
$('attach-video').onclick=()=>$('source-video').click();
$('source-video').onchange=async()=>{
 const selected=mission,file=$('source-video').files[0];if(!selected||!file)return;
 const index=+$('frames').value;let opened;mediaLoading=true;enable();status('Checking source video against saved samples…');
 try{
  opened=await openInput(file,document.createElement('video'));
  if(opened.type!=='video')throw Error('Select the source video.');
  await checkSavedVideo(selected,{decode:time=>frameAt(opened,time,selected.view.camera),savedPixels:async frame=>{const bitmap=await createImageBitmap(await storage.queryBlob(selected,frame));try{return gray(bitmap,selected.view.camera.width,selected.view.camera.height).gray}finally{bitmap.close()}}});
  if(mission!==selected)throw Error('The saved observation changed. Attach the video again.');
  media?.close();media=opened;opened=null;input=file;inputFiles=[file];mediaMissionId=selected.id;
  $('input-name').textContent=file.name+' · source video for saved track';$('video').src=media.source.src;$('video').load();$('video').hidden=false;$('query').hidden=true;$('video-controls').hidden=false;
  $('frame-time').max=Math.max(0,media.duration-.04);$('frame-time').value=replaySeekTime(selected.view.frames[index]);
  await showFrame(index);status('Source video attached. Saved poses are unchanged.');
 }catch(e){error(e)}finally{opened?.close();mediaLoading=false;$('source-video').value='';enable()}
};
async function refreshMissions(selected){const items=await storage.all('missions');$('missions').replaceChildren(new Option('Select a saved observation',''),...items.sort((a,b)=>b.saved_at-a.saved_at).map(m=>new Option(`${m.view.input?.name||'Observation'} · ${new Date(m.saved_at).toLocaleString()} · ${m.view.frames.length} frame(s)`,m.id)));if(selected)$('missions').value=selected}
async function showMission(value){mission=value;$('video').hidden=media?.type!=='video'||mediaMissionId!==value.id;$('query').hidden=!$('video').hidden;const saved=await storage.get('packs',value.pack_id);if(!saved||!await storage.verifyPack(saved))throw Error('Saved observation requires a missing package. Download its region.');pack=saved;region=regions.find(r=>r.pack_id===pack.pack_id);if(!region)throw Error('Saved region catalog is missing');$('region').value=region.id;$('overview').src=assetUrl(region.thumbnail);
  if(!loadedCamera||JSON.stringify(loadedCamera)!==JSON.stringify(mission.view.camera)||map.pack_id!==pack.pack_id){await loadMap(mission.view.camera);map.pack_id=pack.pack_id}
  $('frames').replaceChildren(...mission.view.frames.map((f,i)=>new Option(`${f.timing_scope==='still image'?(f.input_name||'Image '+(i+1)):(f.capture_time_ns/1e9).toFixed(2)+' s'} · ${frameLabel(f)}`,i)));$('frames').disabled=false;status(missionSummary(mission.view.frames));$('export-track').disabled=false;trackPreview.setFrames(mission.view.frames,{period:mission.view.prior.sample_period});await showFrame(0);await trackPreview.overview()}
async function showFrame(index){const f=mission.view.frames[index];if(media?.type==='video'&&mediaMissionId===mission.id)$('video').currentTime=replaySeekTime(f);if(queryURL)URL.revokeObjectURL(queryURL);queryURL=URL.createObjectURL(await storage.queryBlob(mission,f));$('query').src=queryURL;$('query-empty').hidden=true;
  const options=fillHypotheses(f);await showHypothesis(f,options[0]);
  $('details').textContent=JSON.stringify({pack_id:pack.pack_id,elevation_datum:pack.elevation_datum,camera:mission.view.camera,observation:f},null,2);$('attribution').textContent=pack.attribution;enable()}
function fillHypotheses(frame,selected){
  const options=hypotheses(frame);$('hypotheses').replaceChildren(...options.map((h,i)=>new Option(`Candidate ${h.candidate_id??i} · ${h.inliers} geometric inliers`,i)));$('hypotheses').disabled=!options.length;
  if(selected)$('hypotheses').value=String(options.indexOf(selected));
  $('hypotheses').onchange=()=>showHypothesis(frame,options[+$('hypotheses').value]).catch(error);return options;
}
function playbackSelection({time,current}){
  if(!mission||media?.type!=='video'||mediaMissionId!==mission.id)return;
  if(!current){$('result').textContent=`No supported camera pose at ${time.toFixed(2)} s.`;$('hypotheses').replaceChildren();$('hypotheses').disabled=true;if($('track-follow').checked)$('map-label').textContent='NO SUPPORTED CAMERA POSE AT THIS TIME';return}
  const {frame}=current.sample,h=current.sample.source_h??current.sample.h;$('frames').value=String(mission.view.frames.indexOf(frame));fillHypotheses(frame,h);showEvidence(frame,h);
  $('details').textContent=JSON.stringify({pack_id:pack.pack_id,camera:mission.view.camera,observation:frame},null,2);
  if($('track-follow').checked)$('map-label').textContent=frame.accepted?'ESTIMATED CAMERA POSE':'SELECTED HYPOTHESIS · NO UNIQUE VISUAL FIX';
}
function showEvidence(frame,h){
  $('compute-status').textContent=executionSummary(frame);
  const value=resultSummary(frame,h);$('result').classList.toggle('rejected',!frame.accepted);$('result').replaceChildren();
  const title=document.createElement('strong');title.textContent=value.title;$('result').append(title);
  if(value.location){const location=document.createElement('div');location.className='coordinates';location.textContent=value.location;$('result').append(location)}
  const explanation=document.createElement('p');explanation.textContent=value.explanation;$('result').append(explanation);
  if(value.metrics.length){const metrics=document.createElement('div');metrics.className='metrics';for(const [label,number] of value.metrics){const metric=document.createElement('div'),name=document.createElement('small'),content=document.createElement('b');metric.className='metric';name.textContent=label;content.textContent=number;metric.append(name,content);metrics.append(metric)}$('result').append(metrics)}
}
async function showHypothesis(frame,h,{moveCamera=true}={}){
  showEvidence(frame,h);trackPreview.select(frame,h);
  if(!moveCamera)return;
  if(h){await map.setPose({position_enu_m:h.position_enu_m,eye_to_enu_xyzw:h.eye_to_enu_xyzw});baseMapLabel=$('map-label').textContent=frame.accepted?'ESTIMATED CAMERA POSE':'SELECTED HYPOTHESIS · NO UNIQUE VISUAL FIX'}
  else{await map.setPose({position_enu_m:localPosition(pack,...pack.anchor_lat_lon,800),eye_to_enu_xyzw:[0,0,0,1]},{constrain:true});baseMapLabel=$('map-label').textContent='REGION OVERVIEW · NO ACCEPTED CAMERA POSE'}

}
$('missions').onchange=()=>storage.get('missions',$('missions').value).then(m=>m&&showMission(m)).catch(error);$('frames').onchange=()=>showFrame(+$('frames').value).catch(error);
$('reset').onclick=()=>map.reset().catch(error);$('zoom-in').onclick=()=>map.move(2,-map.height()*.25).catch(error);$('zoom-out').onclick=()=>map.move(2,map.height()/3).catch(error);
async function start(){if('serviceWorker' in navigator)await navigator.serviceWorker.register(assetUrl('sw.js'));try{regions=await request('/api/catalog');await storage.put('state','catalog',regions)}catch(e){regions=await storage.get('state','catalog');if(!regions)throw e;status('Offline. Using saved coverage.')}
  if(dataService.static){$('dynamic-coverage').disabled=true;$('coverage-status').textContent='New area and route downloads are unavailable on this site. Choose a prepared area.';$('fetch-coverage').disabled=true;}
  $('region').replaceChildren(...regions.map(r=>new Option(r.label,r.id)));const active=await storage.get('state','active-pack'),preferred=regions.find(r=>r.pack_id===active);if(preferred)$('region').value=preferred.id;await storageStatus();await refreshMissions();await chooseRegion();const id=new URLSearchParams(location.search).get('job');if(id){if(!/^[a-f0-9]{32}$/.test(id))throw Error('Invalid job ID');const job=await request('/api/jobs/'+id);if(job.status!=='complete')throw Error('The selected job is not complete');const manifest=await storage.get('packs',job.pack_id||regions.find(r=>JSON.stringify(r.anchor_lat_lon)===JSON.stringify(job.view.frames.find(f=>f.anchor_lat_lon)?.anchor_lat_lon))?.pack_id||region.pack_id);const saved=await storage.saveMission(job,manifest);await refreshMissions(saved.id);await showMission(saved)}}
start().then(()=>{document.documentElement.dataset.visualReady='true';window.dispatchEvent(new Event('visual-ready'))}).catch(error);

$('fetch-coverage').onclick=async()=>{busy=true;enable();$('fetch-coverage').disabled=true;try{
 const selection=$('coverage-mode').value==='area'?{bounds:['west','south','east','north-bound'].map(id=>+$(id).value),zoom:+$('coverage-zoom').value}:{route:$('route-points').value.trim().split('\n').map(line=>line.trim().split(',').map(Number)),buffer_m:+$('route-buffer').value,zoom:+$('coverage-zoom').value};
 const plan=await request('/api/coverage-plan',selection);$('coverage-status').textContent=`${plan.imagery_tiles.length} imagery tiles · provider: ${plan.provider}`;
 let job=await request('/api/coverage-download',selection);while(['queued','running'].includes(job.status)){$('coverage-status').textContent=job.progress;await new Promise(r=>setTimeout(r,1000));job=await request('/api/downloads/'+job.id)}if(job.status!=='complete')throw Error(job.error);
 const preservePrior=Boolean(region),item=job.region;regions=regions.filter(r=>r.id!==item.id);regions.push(item);await storage.put('state','catalog',regions);$('region').replaceChildren(...regions.map(r=>new Option(r.label,r.id)));$('region').value=item.id;await chooseRegion(preservePrior);await storage.download(item.manifest,(n,t)=>{$('progress').value=n/t;$('coverage-status').textContent=`Storing verified package ${(n/1048576).toFixed(1)} / ${(t/1048576).toFixed(1)} MB`});pack=item.manifest;await openPrior();await storageStatus();$('coverage-status').textContent='New coverage is available offline';
 }catch(e){error(e);$('coverage-status').textContent=String(e)}finally{busy=false;$('fetch-coverage').disabled=false;enable()}};
$('coverage-mode').onchange=()=>{$('area-fields').hidden=$('coverage-mode').value!=='area';$('route-fields').hidden=$('coverage-mode').value!=='route'};

$('map').addEventListener('clearance',({detail:c})=>{$('clearance-status').textContent=c.known?(c.adjusted?'Display raised to terrain clearance. Estimated pose unchanged. · ':'')+c.agl.toFixed(0)+' m above rendered terrain'+(c.constrained?' · minimum '+c.minimum.toFixed(0)+' m':' · estimated camera'):(c.constrained?'Terrain unavailable here · display stays at least 10 km above datum':'Terrain unavailable at the estimated camera');});
$('map').addEventListener('viewchange',({detail})=>{if(detail==='pose')return;$('map-label').textContent=detail==='reset'?baseMapLabel:detail==='globe'?'GLOBE OVERVIEW · DISPLAY CONTEXT':'FREE CAMERA · RESET TO RETURN'});

const dynamicCoverage=new CoverageLoader({request,download:(p,progress)=>storage.download(p,progress,{activate:false}),installed:async()=>[...(map.displayPacks?.values()??[])],available:()=>storage.all('packs'),attach:pack=>map.addPack(pack),status:text=>$('dynamic-status').textContent=text,remember:async item=>{regions=regions.filter(r=>r.id!==item.id);regions.push(item);await storage.put('state','catalog',regions);$('region').append(new Option(item.label,item.id));await storageStatus()}});
let coverageTimer;
function scheduleCoverage(){clearTimeout(coverageTimer);coverageTimer=setTimeout(()=>dynamicCoverage.update(viewCoverage(map.pack,map.pose,JSON.parse($('map').dataset.clearance||'{}').known===false&&map.height()<=10001?5000:map.height())).catch(error),700)}
$('dynamic-coverage').onchange=()=>{dynamicCoverage.enabled=$('dynamic-coverage').checked;dynamicCoverage.failed.clear();$('dynamic-status').textContent=dynamicCoverage.enabled?'Viewed-area downloads enabled.':'Dynamic coverage is off.';if(dynamicCoverage.enabled)scheduleCoverage()};
$('map').addEventListener('viewchange',scheduleCoverage);

$('export-track').onclick=()=>{if(!mission)return;const url=URL.createObjectURL(new Blob([JSON.stringify(sequenceTrack(mission.view.frames),null,2)],{type:'application/geo+json'})),link=document.createElement('a');link.href=url;link.download='visual-hypotheses.geojson';link.click();setTimeout(()=>URL.revokeObjectURL(url),0)};
