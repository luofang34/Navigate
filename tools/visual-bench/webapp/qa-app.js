import {get} from './storage.js';
const report={test:'upload-theme',image_source:'map-derived; not independent accuracy evidence',started_at:new Date().toISOString()},frame=document.querySelector('iframe');
const emit=async phase=>{report.phase=phase;document.querySelector('pre').textContent=JSON.stringify(report,null,2);await fetch('/__test__/report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(report)})};
const check=(name,value)=>{if(!value)throw Error(name);report[name]=true};
const event=(target,name)=>new Promise(resolve=>target.addEventListener(name,resolve,{once:true}));
async function ready(){if(!frame.contentDocument||frame.contentDocument.URL==='about:blank'||frame.contentDocument.readyState==='loading')await event(frame,'load');const w=frame.contentWindow;if(w.document.documentElement.dataset.visualReady!=='true')await event(w,'visual-ready');return w}
async function upload(w,path,name,type){const response=await fetch(path);if(!response.ok)throw Error('Fixture unavailable: '+path);const transfer=new w.DataTransfer();transfer.items.add(new w.File([await response.arrayBuffer()],name,{type}));const input=w.document.getElementById('input');input.files=transfer.files;await input.onchange()}
async function run(){
 await emit('loading application');let w=await ready(),d=w.document,$=id=>d.getElementById(id);
 const theme=$('theme');theme.value='dark';theme.dispatchEvent(new w.Event('change'));check('dark theme selected',d.documentElement.dataset.theme==='dark');
 const reloaded=event(frame,'load');frame.src='./index.html?qa-reload';await reloaded;w=await ready();d=w.document;$=id=>d.getElementById(id);check('dark theme survives reload',d.documentElement.dataset.theme==='dark'&&$('theme').value==='dark');
 $('region').value='naip-2864a5ec8e4abb24';await $('region').onchange();if($('locate').disabled)await $('download').onclick();
 await emit('image upload');await upload(w,'/assets/demo-frame.jpg','map-derived-example.jpg','image/jpeg');
 check('image preview shown before matching',!$('query').hidden&&$('query-empty').hidden&&$('query').src.startsWith('blob:'));
 check('image enables matching',!$('locate').disabled);check('map has direct GPU presentation',$('map').dataset.presentation==='wgpu-direct');
 await emit('image matching');await $('locate').onclick();
 const mission=await get('missions',$('missions').value),result=mission?.view.frames[0];
 check('image creates a saved result',Boolean(result));check('image has geometric hypotheses',result.candidate_hypotheses.some(h=>h.accepted));
 check('matched pose controls are available',$('hypotheses').options.length>0&&!$('reset').disabled);report.processing_ms=result.processing_ms;report.geographic_accuracy=result.geographic_accuracy;
 await emit('video upload');await upload(w,'/models/test-video.mp4','local-test-video.mp4','video/mp4');
 check('video controls available',!$('video-controls').hidden&&!$('video').hidden);
 const time=Math.min(.5,$('video').duration/2),shown=event($('query'),'load');$('frame-time').value=String(time);$('frame-time').onchange();await shown;
 check('selected video frame decoded',Math.abs($('video').currentTime-time)<.05);check('video frame enables matching',!$('locate').disabled);
 report.video_time_s=$('video').currentTime;await emit('video frame matching');await $('locate').onclick();
 const video=await get('missions',$('missions').value),observation=video?.view.frames[0];check('selected video frame creates a result',Math.abs(observation.requested_time_s-time)<.05);check('video frame runs in browser',video.view.input.processing==='browser-local');report.video_decision=observation.decision;
}
let deadline;
try{
 await Promise.race([run(),new Promise((resolve,reject)=>{deadline=setTimeout(()=>reject(Error('Application regression timed out')),180000)})]);
 document.title='PASS application upload, frame selection and dark theme';await emit('complete');
}catch(error){report.error=String(error);document.title='FAIL application regression';await emit('failed')}finally{clearTimeout(deadline);frame.src='about:blank'}
