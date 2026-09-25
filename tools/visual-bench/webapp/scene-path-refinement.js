import {scenePaths} from './scene-paths.js';
import {sceneMapPlans} from './scene-map.js';
import {refinementFrames,addUnresolvedFrames,connectRefinementFrames} from './scene-refinement-frames.js';
import {refinementSource} from './scene-refinement-source.js';
const correlation='unknown; scene, camera, and registration estimates reuse source observations; repeated processing adds no independent evidence';

export async function refineScenePaths(records,reconstruction,frames,options){
 const {kernel,save,progress=()=>{}}=options;
 if(reconstruction.geographic_acceptance!==false)throw Error('Path refinement requires conditional scene estimates');
 if(reconstruction.path_refinement)return reconstruction;
 const {selected,deferred}=scenePaths(reconstruction,{maxPaths:3}),schedule=sceneMapPlans(reconstruction,frames),groups=[],attempts=[];
 const sources=[...records,...(reconstruction.sampling_refinement?.attempts??[]).flatMap(a=>a.image_groups)];
 for(const path of selected){
  const source=await refinementSource(path,reconstruction,sources,{...options,progress});
  try{
   const required=schedule.plans.filter(p=>p.paths.some(p=>p.leaf_sha256===path.leaf_sha256)).map(p=>p.anchor.frame.observation_sha256);
   const result=jointPath(source,required,{kernel,progress});
   attempts.push({source_scene_path_sha256:path.nodes.map(n=>n.record.sha256),passes:result.passes,reason:result.reason,geographic_acceptance:false});
   if(!result.last)continue;
   const {last}=result,dense=denseScene(source,last,{kernel,progress});
   const training=await save(JSON.stringify({stage:'local_scene_reconstruction',method:'conditional_joint_refinement',geographic_acceptance:false,
    source_scene_path_sha256:path.nodes.map(n=>n.record.sha256),source_groups:source.sourceGroups,graph:last.graph,
    scene:last.fit.scene,initial_cost:last.fit.initial_cost,final_cost:last.fit.final_cost,evidence_correlation:correlation}));
   const cameraFits=last.results.filter(r=>r.fit).map(r=>({observation_sha256:r.observation_sha256,position_scene_units:r.fit.position_scene_units,eye_to_scene_xyzw:r.fit.eye_to_scene_xyzw,fixed:false}));
   const payload={stage:'local_scene_reconstruction',method:'conditional_path_refinement',geographic_acceptance:false,
    scene:dense.scene,training_scene:training,camera_fit_scene_sha256:last.scene_sha256,camera_fits:cameraFits,camera_fit_evidence:last.results,
    source_scene_path_sha256:path.nodes.map(n=>n.record.sha256),source_groups:source.sourceGroups,graph:dense.graph,
    unresolved_observation_sha256:last.results.filter(r=>!r.fit).map(r=>r.observation_sha256),unresolved_feature_ids:dense.unresolved_feature_ids,
    point_refinement:dense.batches,evidence_correlation:correlation,uncertainty:'unknown; estimated calibration, pose, and scene geometry; absolute accuracy is not measured'};
   const record=await save(JSON.stringify(payload));
   groups.push({source_group_sha256:null,observations:source.observations,candidates:[{record,local_solution:record,coordinate_frame:record.sha256,parent_scene_sha256:null,
    cameras:dense.scene.cameras,camera_fits:cameraFits,continuation:'scheduled',continuation_rank:0}],geographic_acceptance:false});
  }finally{source.merger.free()}
 }
 return {...reconstruction,groups:[...reconstruction.groups,...groups],path_refinement:{attempts,deferred_leaf_sha256:deferred.map(p=>p.leaf_sha256),geographic_acceptance:false,evidence_correlation:correlation}};
}

function jointPath(source,required,{kernel,progress}){
 const {camera,merger,poses,points,observations}=source,cameraJson=JSON.stringify(camera),passes=[];
 let selected=refinementFrames(observations,required),last,reason;
 for(let pass=0;pass<8;pass++){
  progress(`Refining camera path · pass ${pass+1}/8 · ${selected.length} keyframes…`);
  const graph=JSON.parse(merger.select(JSON.stringify(selected.map(o=>o.observation_sha256)),16000,40));
  const cameras=selected.map((o,i)=>({...poses.get(o.observation_sha256),fixed:i===0}));
  const seeds=graph.association_sources.flatMap(t=>{
   const p=t.source_tracks.map(s=>points.get(s.group_sha256+'/'+s.feature_id)).find(Boolean);
   return p?[{feature_id:t.feature_id,position_scene_units:p.position_scene_units}]:[];
  });
  const initial=JSON.parse(kernel.initialize_scene_points(cameraJson,JSON.stringify(graph.graph),JSON.stringify({cameras,points:[]}),JSON.stringify(seeds))).scene;
  const connection=connectRefinementFrames(selected,observations,initial.points);
  if(connection.disconnected_observation_sha256.length){
   passes.push({stage:'connection',keyframes:selected.length,disconnected_observation_sha256:connection.disconnected_observation_sha256,added:connection.frames.length-selected.length});
   if(connection.frames.length===selected.length){reason='Image support remains disconnected within the source and work budget';break}
   selected=connection.frames;continue;
  }
  let scale=1;const distance=c=>Math.hypot(...c.position_scene_units.map((v,i)=>v-cameras[0].position_scene_units[i]));
  for(let i=2;i<cameras.length;i++)if(distance(cameras[i])>distance(cameras[scale]))scale=i;
  initial.coordinate_gauge={origin_camera:0,scale_camera:scale};
  let fit;
  try{fit=JSON.parse(kernel.refine_scene(cameraJson,JSON.stringify(initial),60))}
  catch(error){reason=String(error);passes.push({stage:'joint_refinement',reason});break}
  for(const c of fit.scene.cameras)poses.set(c.observation_sha256,c);
  const scene_sha256=merger.set_scene(JSON.stringify(fit.scene)),results=[],unresolved=[];
  for(const [index,o] of observations.entries()){
   if(index%64===0)progress(`Checking reconstructed cameras · ${index+1}/${observations.length}…`);
   const result=JSON.parse(merger.fit_camera(o.observation_sha256,JSON.stringify(poses.get(o.observation_sha256))));
   if(result.observation_sha256!==o.observation_sha256||result.scene_sha256!==scene_sha256||result.geographic_acceptance!==false)throw Error('Camera fit changed evidence identity');
   results.push(result);if(!result.fit)unresolved.push(o.observation_sha256);
   else poses.set(o.observation_sha256,{observation_sha256:o.observation_sha256,position_scene_units:result.fit.position_scene_units,eye_to_scene_xyzw:result.fit.eye_to_scene_xyzw,fixed:false});
  }
  last={graph,fit,scene_sha256,results,selected};
  passes.push({stage:'joint_refinement',keyframes:selected.length,points:fit.scene.points.length,supported:results.length-unresolved.length,unresolved_observation_sha256:unresolved,initial_cost:fit.initial_cost,final_cost:fit.final_cost});
  if(!unresolved.length)break;
  const next=addUnresolvedFrames(selected,observations,unresolved);if(next.length===selected.length){reason='Unresolved cameras remain within the work budget';break}selected=next;
 }
 return {last,passes,reason};
}

function denseScene(source,last,{kernel,progress}){
 const camera=JSON.stringify(source.camera),graph=JSON.parse(source.merger.select(JSON.stringify(last.selected.map(o=>o.observation_sha256)),65536,16));
 const cameras=last.fit.scene.cameras.map(c=>({...c,fixed:true}));
 const result=JSON.parse(kernel.triangulate_scene_points(camera,JSON.stringify(graph.graph),JSON.stringify({cameras,points:[]})));
 const points=[],batches=[];
 for(let i=0;i<result.scene.points.length;i+=16000){
  progress(`Refining reference points · ${Math.min(i+16000,result.scene.points.length)}/${result.scene.points.length}…`);
  const fit=JSON.parse(kernel.refine_scene(camera,JSON.stringify({cameras,points:result.scene.points.slice(i,i+16000)}),60));
  points.push(...fit.scene.points);batches.push({points:fit.scene.points.length,initial_cost:fit.initial_cost,final_cost:fit.final_cost});
 }
 return {scene:{cameras,points},graph,unresolved_feature_ids:result.unresolved_feature_ids,batches};
}
