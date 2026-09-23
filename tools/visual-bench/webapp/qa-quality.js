import {BrowserPipeline} from './browser-pipeline.js';
import {matchingOptions} from './matching-options.js';
import {cameraForImage} from './calibration.js';
import {gray} from './observation.js';
import {download} from './storage.js';
const progress=text=>document.getElementById('progress').textContent=text,report={};
try {
  const pack=await(await fetch('/api/offline-plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({region_id:'new-jersey'})})).json();
  await download(pack,(n,t)=>progress(`Package ${n}/${t}`));
  const bitmap=await createImageBitmap(await(await fetch('/models/test-input.png')).blob());
  for(const mode of ['fast','balanced']) {
    const options=matchingOptions(mode),camera=cameraForImage(bitmap.width,bitmap.height,82.1,options.longEdge),pipeline=new BrowserPipeline();
    try {
      await pipeline.initialize(pack,camera,progress,options);
      const frame={...gray(bitmap,camera.width,camera.height),time:0};
      const result=await pipeline.estimate(frame,{latitude:40.5442,longitude:-74.4564,radius_m:500,agl_m:110},0,text=>progress(`${mode}: ${text}`));
      report[mode]={camera,processing_ms:result.processing_ms,retrieval:result.retrieval,decision:result.decision,hypotheses:result.candidate_hypotheses,execution:result.execution};
      if(!result.candidate_hypotheses.some(h=>h.accepted))throw Error(`${mode} produced no geometric hypothesis`);
      document.getElementById('result').textContent=JSON.stringify(report,null,2);
    } finally {pipeline.close();}
  }
  bitmap.close();document.title='PASS matching detail comparison';
}catch(error){report.error=String(error);document.title='FAIL matching detail comparison';}
document.getElementById('result').textContent=JSON.stringify(report,null,2);
await fetch('/__test__/report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(report)});
