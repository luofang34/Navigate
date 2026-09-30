import {refineReferenceCandidates} from './candidate-refinement.js';
export function retrievalFixture(fixture,sourceSha256,pack,prior){
 if(fixture.schema!==2||fixture.map_manifest_sha256!==pack.pack_id)throw Error('Retrieval fixture map identity differs');
 if(!/^[a-f0-9]{64}$/.test(sourceSha256))throw Error('Retrieval fixture needs the source image digest');
 for(const key of ['latitude','longitude','radius_m','agl_m'])if(fixture.prior[key]!==prior[key])throw Error('Retrieval fixture prior differs');
 const entry=fixture.cases[sourceSha256];
 if(!entry||!Array.isArray(entry.candidates)||entry.candidates.length>128)throw Error('Retrieval fixture has no bounded candidate set for this image');
 const candidates=entry.candidates.map(p=>{
  const v=p.position_enu_m,q=p.eye_to_enu_xyzw;
  if(!Array.isArray(v)||v.length!==3||!Array.isArray(q)||q.length!==4||[...v,...q].some(x=>!Number.isFinite(x))||Math.abs(Math.hypot(...q)-1)>1e-6)throw Error('Invalid retrieval fixture pose');
  return {position_enu_m:v.slice(),eye_to_enu_xyzw:q.slice()};
 });
 return {candidates,source:fixture.source,source_sha256:sourceSha256,reference_ids:entry.reference_ids,scope:'Native retrieval proposals only; browser matching and geometry are evaluated separately'};
}

export function refineRetrievalFixture(pipeline,seeds,image,observation,progress){
 return refineReferenceCandidates(pipeline.renderer,pipeline.matcher,pipeline.camera,seeds.candidates,image,observation,progress);
}
