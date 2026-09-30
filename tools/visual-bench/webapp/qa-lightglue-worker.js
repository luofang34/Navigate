import {refineCandidates} from './temporal-search.js';
import {LocalMatcher} from './inference/local.js';
import {ResearchLightGlue} from './qa-lightglue-adapter.js';
import init,{Preview} from './wasm/navigate_visual_preview.js';
import {download,read} from './storage.js';
import {gray} from './observation.js';
self.onmessage=async({data})=>{
 const model=data.model==='public'?'public-loftr':data.model==='adaptive'?'public-loftr-adaptive':data.model==='patches'?'public-loftr-patches':'research-lightglue';
 const report={test:'prepared candidate browser comparison',model,scope:'Prepared estimated reference candidates; up to three same-observation refinements; no retrieval recall or independently measured camera truth',cases:[]};let renderer,matcher;
 const progress=text=>self.postMessage({progress:text});
 try{
  const cases=await(await fetch('/models/research-lightglue/cases.json')).json();
  const response=await fetch('/api/offline-plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({region_id:'naip-3754222efffa06cf'})});if(!response.ok)throw Error(await response.text());const pack=await response.json();await download(pack,()=>{});await init();
  renderer=await Preview.create(JSON.stringify(pack),JSON.stringify(cases[0].camera),read,false);matcher=model.startsWith('public-')?new LocalMatcher({matcher:'dense',refinementPatches:model==='public-loftr-adaptive'?'on_rejection':model==='public-loftr-patches'}):new ResearchLightGlue();await matcher.initialize(progress);
  for(const [index,row] of cases.entries()){
   progress('Matching '+row.id);const bitmap=await createImageBitmap(await(await fetch(row.query)).blob()),image=gray(bitmap,row.camera.width,row.camera.height);bitmap.close();
   const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',image.gray))].map(v=>v.toString(16).padStart(2,'0')).join('');if(hash!==row.query_image_sha256)throw Error('Query differs from the native fixture');
   renderer.begin(image.gray,JSON.stringify(row.prior),index,0);const start=performance.now();
   let candidate=row.reference_pose,result;const history=[];
   if(model==='public-loftr-adaptive'){
    const observed={render_reference:(...args)=>renderer.render_reference(...args),select:()=>renderer.select(),refine(...args){const text=renderer.refine(...args);history.push({pass:history.length,pairs:JSON.parse(args[1]).length,result:JSON.parse(text)});return text}};
    const checked=await refineCandidates(observed,matcher,row.camera,[candidate],image,hash,3,progress);result=checked.candidate_hypotheses[0];
   }else for(let pass=0;pass<3;pass++){
    const pixels=await renderer.render_reference(0,JSON.stringify(candidate));
    const {pairs,backend_identity}=await matcher.matchImages({gray:pixels,width:row.camera.width,height:row.camera.height},image,{stage:'refinement',progress});
    result=JSON.parse(renderer.refine(0,JSON.stringify(pairs),backend_identity));history.push({pass,pairs:pairs.length,result});
    const next=result.accepted?result:result.refinement_proposal;if(!next)break;
    candidate={position_enu_m:next.position_enu_m,eye_to_enu_xyzw:next.eye_to_enu_xyzw};
   }
   report.cases.push({name:row.id,history,elapsed_ms:performance.now()-start,geometric_acceptance:result.accepted,result,execution:matcher.diagnostics()});self.postMessage({report});
  }
 }catch(error){report.error=String(error)}finally{renderer?.free();await matcher?.close();self.postMessage({report,done:true})}
};
