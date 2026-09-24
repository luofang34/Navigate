import {get} from './storage.js';
import {trackBranches} from './track-preview.js';
import {playbackPose} from './pose-playback.js';
import {toGlobePose} from './geography.js';
const report={test:'main-application-video-track',source:'map-derived translated video; UI and geometric control, not independent DJI accuracy'},frame=document.querySelector('iframe');
const check=(name,value)=>{if(!value)throw Error(name);report[name]=true};
const until=(target,test)=>new Promise(resolve=>{const observer=new MutationObserver(()=>{if(test()){observer.disconnect();resolve()}});observer.observe(target,{subtree:true,attributes:true,childList:true,characterData:true});if(test()){observer.disconnect();resolve()}});
const event=(target,name)=>new Promise(resolve=>target.addEventListener(name,resolve,{once:true}));
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
 check('map moves during matching',Number(map.dataset.frames)>oldFrames);await processing;
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
 await $('missions').onchange();check('saved track replay exposes source video attachment',!$('attach-video').disabled);
 const attach=new w.DataTransfer();attach.items.add(transfer.files[0]);$('source-video').files=attach.files;await $('source-video').onchange();
 check('saved track reattaches its source video without matching',!video.hidden&&$('query').hidden&&$('run-status').textContent==='Source video attached. Saved poses are unchanged.');
 check('reattachment preserves stored hypotheses',JSON.stringify((await get('missions',mission.id)).view.frames)===JSON.stringify(mission.view.frames));
 const other=await fetch('./models/test-video.mp4'),wrong=new w.DataTransfer();wrong.items.add(new w.File([await other.arrayBuffer()],'unrelated.mp4',{type:'video/mp4'}));$('source-video').files=wrong.files;await $('source-video').onchange();
 check('mismatched replay video is rejected',$('run-status').textContent.includes('does not match the saved samples'));
 check('mismatched video keeps the current track and source',!video.hidden&&$('missions').value===mission.id&&$('input-name').textContent.includes('map-derived-track.mp4'));

 await emit('track preview passed');
 await $('region').onchange();check('region change clears the old camera track',$('track-status').textContent===''&&$('track-camera').disabled);
}
let timer;try{await Promise.race([run(),new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('Video track regression timed out')),420000)})]);document.title='PASS main application video track';await emit('complete')}catch(e){report.error=String(e);document.title='FAIL main application video track';await emit('failed')}finally{clearTimeout(timer)}
