import init,{Preview} from './wasm/navigate_visual_preview.js';
import {LocalMatcher} from './inference/local.js';
import {download,read} from './storage.js';
import {gray} from './observation.js';
import {cameraForImage} from './calibration.js';
import {ReferencePack} from './reference-pack.js';
import {localPosition} from './geography.js';
async function png(pixels,width,height){const c=new OffscreenCanvas(width,height),rgba=new Uint8ClampedArray(pixels.length*4);for(let i=0;i<pixels.length;i++){rgba[i*4]=rgba[i*4+1]=rgba[i*4+2]=pixels[i];rgba[i*4+3]=255}c.getContext('2d').putImageData(new ImageData(rgba,width,height),0,0);const bytes=new Uint8Array(await(await c.convertToBlob({type:'image/png'})).arrayBuffer());let b='';for(let i=0;i<bytes.length;i+=8192)b+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(b)}
self.onmessage=async({data:params})=>{
 const report={test:'NAIP supplied-pose resolution diagnostic',scope:'Candidate poses from supplied imagery, not independent truth',cases:[],references:[]};let matcher,renderer,dense;
 try {
  const response=await fetch('/api/offline-plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({region_id:params.region||'naip-2864a5ec8e4abb24'})}),pack=await response.json();
  await download(pack,(n,t)=>self.postMessage({progress:`Package ${n}/${t}`}));await init();const camera=cameraForImage(1920,1080,82.1,960),references=await ReferencePack.open(pack);
  matcher=new LocalMatcher({keypoints:2048});await matcher.initialize(progress=>self.postMessage({progress}));if(params.model==='browser-loftr'){const {DenseMatcher}=await import('./qa-loftr-adapter.js');dense=await DenseMatcher.create(new Uint8Array(await(await fetch('./models/diagnostic-loftr.onnx')).arrayBuffer()),matcher.matcher.gpu,matcher.matcher.metrics)}renderer=await Preview.create(JSON.stringify(pack),JSON.stringify(camera),read,false);
  const external=params.model&&params.model!=='browser-loftr'?await(await fetch('./models/diagnostic-pairs.json')).json():null;const seeds=await(await fetch('/diagnostic-seeds.json')).json();
  for(const [name,h] of Object.entries(seeds).filter(([name])=>!params.frame||name===params.frame)){
   const bitmap=await createImageBitmap(await(await fetch('./models/test-frames/'+name)).blob()),original=gray(bitmap,camera.width,camera.height);bitmap.close();
   const pose={position_enu_m:localPosition(pack,h.latitude_deg,h.longitude_deg,h.altitude_m),eye_to_enu_xyzw:h.eye_to_enu_xyzw};
   const prior={pose:{position_enu_m:localPosition(pack,40.5442,-74.4564,references.elevation(40.5442,-74.4564)+110),eye_to_enu_xyzw:[0,0,0,1]},position_radius_m:500,attitude_radius_rad:Math.PI};
   for(const edge of params.edges?params.edges.split(",").map(Number):[960,640,480,320,240]){
    self.postMessage({progress:`${name} · query prefilter ${edge}`});const start=performance.now();renderer.begin(original.gray,JSON.stringify(prior),0,0);
    const small=gray(original.canvas,edge,Math.round(edge*camera.height/camera.width)),query=gray(small.canvas,camera.width,camera.height);
    let candidate=pose;const passes=[];
    for(let pass=0;pass<(external?1:3);pass++){
     const pixels=await renderer.render_reference(0,JSON.stringify(candidate)),reference={gray:pixels,width:camera.width,height:camera.height};
     if(params.export&&pass===0)report.references.push({name,pose,camera,reference_png:await png(pixels,camera.width,camera.height),query_png:await png(original.gray,camera.width,camera.height)});
     const {pairs,backend_identity}=dense?{pairs:await dense.match(reference,query),backend_identity:'loftr-browser-diagnostic'}:external?{pairs:external[params.model].find(row=>row.name===name+'-reference.png').pairs,backend_identity:params.model+'/local-diagnostic'}:await matcher.matchImages(reference,query,{query:`${name}/prefilter/${edge}`,stage:'refinement'});
     const result=JSON.parse(renderer.refine(0,JSON.stringify(pairs),`${backend_identity}/diagnostic-prefilter-${edge}`));passes.push({pairs:pairs.length,result});if(!result.accepted)break;candidate=result;
    }
    report.cases.push({name,edge,elapsed_ms:performance.now()-start,passes});self.postMessage({report});
   }
  }
 }catch(e){report.error=String(e)}finally{await dense?.close();await matcher?.close();renderer?.free();self.postMessage({report,done:true})}
};
