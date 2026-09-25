import {loadSceneNode} from './scene-refinement-source.js';
import {scenePaths} from './scene-paths.js';
// A failed single-image check can initialize a separate scene-to-map solve.
function referenceSeeds(frame){
 return frame.candidate_hypotheses.flatMap(h=>{
  if(!h.map_manifest_sha256)return [];
  if(h.accepted)return [{...h,initialization_stage:'map_pose_geometry'}];
  const pose=h.refinement_proposal??h.reference_pose;
  if(!pose||!h.reference_image_sha256||!h.reference_depth_sha256)return [];
  return [{...pose,candidate_id:h.candidate_id,map_manifest_sha256:h.map_manifest_sha256,
   accepted:false,initialization_stage:'reference_proposal',source_geometry_reason:h.reason}];
 });
}
function seedPriority(seeds){return Math.max(...seeds.map(s=>s.accepted?1e6+(s.inliers??0):0))}
// Registration and preview retain each exact parent chain. Shared axes are not a path ID.
export function sceneMapPlans(reconstruction,frames,{maxAnchors=3,maxPaths=32}={}){
 if(!Number.isInteger(maxAnchors)||maxAnchors<1||maxAnchors>8||!Number.isInteger(maxPaths)||maxPaths<1||maxPaths>128)throw Error('Invalid scene map budget');
 const observations=new Map(frames.map((f,index)=>[f.observation_sha256,{frame:f,index}]));
 const {selected,deferred}=scenePaths(reconstruction,{maxPaths});
 const nodes=new Map(reconstruction.groups.flatMap(g=>g.candidates.map(c=>[c.record.sha256,c])));
 const roots=[...new Set(selected.map(p=>p.root.record.sha256))],plans=[];
 for(const rootId of roots){
  const root=nodes.get(rootId),related=selected.filter(p=>p.root.record.sha256===rootId),available=[];
  const sources=new Map(related.flatMap(p=>p.nodes).map(n=>[n.record.sha256,n]));
  for(const source of sources.values())for(const c of source.cameras){
   const observation=observations.get(c.observation_sha256);if(!observation)continue;
   const seeds=referenceSeeds(observation.frame);if(!seeds.length)continue;
   available.push({source,anchor:{...observation,seeds},paths:related.filter(p=>p.nodes.some(n=>n.record.sha256===source.record.sha256))});
  }
  const used=new Set(),covered=new Set(),anchorIndices=[];
  for(let count=0;count<maxAnchors;count++){
   const candidates=available.filter(p=>!used.has(p.anchor.frame.observation_sha256));if(!candidates.length)break;
   const coverage=p=>p.paths.filter(path=>!covered.has(path.leaf_sha256)).length;
   const geometric=p=>Number(p.anchor.seeds.some(seed=>seed.accepted));
   const spread=p=>anchorIndices.length?Math.min(...anchorIndices.map(index=>Math.abs(index-p.anchor.index))):0;
   candidates.sort((a,b)=>coverage(b)-coverage(a)||geometric(b)-geometric(a)||spread(b)-spread(a)||seedPriority(b.anchor.seeds)-seedPriority(a.anchor.seeds)||a.anchor.index-b.anchor.index||a.source.record.sha256.localeCompare(b.source.record.sha256));
   const chosen=candidates[0];plans.push({root,...chosen});used.add(chosen.anchor.frame.observation_sha256);anchorIndices.push(chosen.anchor.index);
   for(const path of chosen.paths)covered.add(path.leaf_sha256);
  }
 }
 return {plans,deferred_leaf_sha256:deferred.map(p=>p.leaf_sha256),
  unregistered_leaf_sha256:selected.filter(p=>!plans.some(plan=>plan.paths.includes(p))).map(p=>p.leaf_sha256),
  unregistered_root_sha256:roots.filter(id=>!plans.some(p=>p.root.record.sha256===id)),stage:'bounded_scene_registration',geographic_acceptance:false};
}

export function appendScenePaths(frames,registrations){
 const byId=new Map(frames.map(f=>[f.observation_sha256,f])),pending=new Map();let added=0;
 for(const item of registrations)for(const candidate of item.candidates)for(const path of candidate.paths){
  const track_id=JSON.stringify([item.record.sha256,candidate.index,path.leaf_sha256]);
  for(const camera of path.cameras){
   const frame=byId.get(camera.observation_sha256);if(!frame)throw Error('Registered path has an unknown observation');
   const proposed=pending.get(frame)??[];
   if([...frame.candidate_hypotheses,...proposed].some(h=>h.track_id===track_id))continue;
   const used=new Set([...frame.candidate_hypotheses,...proposed].map(h=>h.candidate_id));let id=0;while(used.has(id))id=(id+1)>>>0;
   const h={...camera,candidate_id:id,accepted:false,scene_supported:true,geographic_acceptance:false,track_id,
    map_manifest_sha256:item.map_manifest_sha256,anchor_observation_sha256:item.observation_sha256,
    scene_registration:{record:item.record,candidate_index:candidate.index,point_fit_rms_m:candidate.point_fit_rms_m,inliers:candidate.inliers},
    scene_path_sha256:path.scene_path_sha256,evidence_correlation:'unknown; estimated scene and registration share image and map evidence',
    geographic_accuracy:'not_independently_measured'};
   proposed.push(h);pending.set(frame,proposed);added=(added+1)>>>0;
  }
 }
 for(const [frame,proposed] of pending){frame.candidate_hypotheses=[...proposed,...frame.candidate_hypotheses];frame.accepted=false;frame.decision='scene_registration'}
 return added;
}

export async function registerScenePaths(plan,image,prior,{renderer,matcher,kernel,camera,pack,navigationPrior,load,save,progress}){
 const limit=plan.preview_path_budget??32;if(!Number.isInteger(limit)||limit<0||limit>128)throw Error('Invalid scene preview budget');
 let previewPaths=0,deferredPreviewPaths=0;
 const {root,anchor,paths}=plan,source=plan.source??root,scene=JSON.stringify(await registrationScene(plan,{load,kernel})),frame=anchor.frame;
 const verifiedFits=await previewCameraFits(paths,load);
 renderer.begin(image.gray,JSON.stringify(navigationPrior(prior)),frame.sequence,frame.capture_time_ns);
 if(JSON.parse(renderer.select()).observation_sha256!==frame.observation_sha256)throw Error('Scene registration changed observation identity');
 const registrations=[],attempts=[];
 for(const [id,seed] of anchor.seeds.entries()){
  if(seed.map_manifest_sha256!==pack.pack_id)throw Error('Scene registration map identity changed');
  progress('Aligning reconstructed path with map imagery…');
  const pixels=await renderer.render_reference(id,JSON.stringify(seed));
  const matches=await matcher.matchImages({gray:pixels,width:camera.width,height:camera.height},image,{query:`${frame.observation_sha256}/query`,stage:'refinement',progress});
  let result;
  try{result=JSON.parse(renderer.register_scene(id,scene,JSON.stringify(matches.pairs),matches.backend_identity))}
  catch(error){attempts.push({candidate_id:seed.candidate_id,reason:String(error)});continue}
  if(result.observation_sha256!==frame.observation_sha256||result.map_manifest_sha256!==pack.pack_id||result.geographic_acceptance!==false)throw Error('Invalid registration provenance');
  const record=await save(JSON.stringify({...result,source_scene:source.record,coordinate_frame:root.coordinate_frame,source_candidate_id:seed.candidate_id,initialization_stage:seed.initialization_stage}));
  const candidates=result.candidates.map((candidate,index)=>({index,inliers:candidate.inlier_indices.length,point_fit_rms_m:candidate.point_fit_rms_m,
   paths:paths.flatMap(path=>{
    if(previewPaths>=limit){deferredPreviewPaths=(deferredPreviewPaths+1)>>>0;return []}previewPaths=(previewPaths+1)>>>0;
    const cameras=new Map();
    for(const node of path.nodes){
     const source=verifiedFits.get(node.record.sha256)??node.cameras;
     for(let start=0;start<source.length;start+=129)for(const pose of JSON.parse(kernel.registered_scene_cameras(JSON.stringify({cameras:source.slice(start,start+129),points:[]}),JSON.stringify(candidate))))cameras.set(pose.observation_sha256,pose);
    }
    return [{leaf_sha256:path.leaf_sha256,scene_path_sha256:path.nodes.map(n=>n.record.sha256),cameras:[...cameras.values()]}];
   })}));
  registrations.push({record,observation_sha256:result.observation_sha256,map_manifest_sha256:result.map_manifest_sha256,candidate_budget_exhausted:result.candidate_budget_exhausted,candidates});
 }
 return {registrations,attempts,preview_paths:previewPaths,deferred_preview_paths:deferredPreviewPaths,source_scene_sha256:source.record.sha256,geographic_acceptance:false};
}

async function registrationScene(plan,{load,kernel}){
 const {root,paths}=plan,node=plan.source??root;
 if(root.record.sha256!==root.coordinate_frame||node.coordinate_frame!==root.coordinate_frame||!paths.length||paths.some(p=>p.root.record.sha256!==root.record.sha256||!p.nodes.some(n=>n.record.sha256===node.record.sha256)))throw Error('Registration source is outside the scene path');
 return loadSceneNode(node,{load,kernel});
}

async function previewCameraFits(paths,load){
 const verified=new Map();
 for(const node of paths.flatMap(p=>p.nodes))if(node.camera_fits&&!verified.has(node.record.sha256)){
  const stored=await load(node.record);
  if(stored.stage!=='local_scene_reconstruction'||stored.method!=='conditional_path_refinement'||stored.geographic_acceptance!==false||JSON.stringify(stored.scene?.cameras)!==JSON.stringify(node.cameras)||JSON.stringify(stored.camera_fits)!==JSON.stringify(node.camera_fits))throw Error('Refined preview cameras changed');
  const ids=new Set(),fits=[];
  for(const result of stored.camera_fit_evidence??[]){
   if(result.scene_sha256!==stored.camera_fit_scene_sha256||result.geographic_acceptance!==false||ids.has(result.observation_sha256))throw Error('Refined camera evidence changed');
   ids.add(result.observation_sha256);
   if(result.fit)fits.push({observation_sha256:result.observation_sha256,position_scene_units:result.fit.position_scene_units,eye_to_scene_xyzw:result.fit.eye_to_scene_xyzw,fixed:false});
  }
  if(JSON.stringify(fits)!==JSON.stringify(node.camera_fits))throw Error('Refined camera support changed');
  verified.set(node.record.sha256,node.camera_fits);
 }
 return verified;
}
