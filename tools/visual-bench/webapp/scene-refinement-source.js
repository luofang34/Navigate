export async function loadSceneNode(node,{kernel,load}){
 const saved=await load(node.record);
 if(saved.geographic_acceptance!==false||JSON.stringify(saved.scene?.cameras)!==JSON.stringify(node.cameras))throw Error('Scene source cameras changed');
 if(!node.parent_scene_sha256){
  if(saved.stage!=='local_scene_reconstruction'||node.coordinate_frame!==node.record.sha256)throw Error('Invalid scene root');
  return saved.scene;
 }
 if(saved.stage!=='conditional_scene_alignment'||saved.parent_scene_sha256!==node.parent_scene_sha256||saved.coordinate_frame!==node.coordinate_frame||saved.local_solution?.sha256!==node.local_solution?.sha256)throw Error('Saved scene alignment changed lineage');
 const local=await load(saved.local_solution);
 if(local.stage!=='local_scene_reconstruction'||local.geographic_acceptance!==false||local.source_group_sha256!==saved.source_group_sha256)throw Error('Saved scene alignment changed source');
 const scene=JSON.parse(kernel.restore_aligned_scene(JSON.stringify(local.scene),JSON.stringify(saved.alignment)));
 if(!sameSceneCameras(scene.cameras,node.cameras))throw Error('Restored scene cameras differ from the saved alignment');
 return scene;
}
export function sameSceneCameras(a,b){
 return a.length===b.length&&a.every((camera,i)=>{
  const saved=b[i];return camera.observation_sha256===saved.observation_sha256&&camera.fixed===saved.fixed&&
   camera.position_scene_units.every((v,j)=>Number.isFinite(v)&&Math.abs(v-saved.position_scene_units[j])<=1e-8*Math.max(1,Math.abs(v)))&&
   Math.abs(Math.abs(camera.eye_to_scene_xyzw.reduce((sum,v,j)=>sum+v*saved.eye_to_scene_xyzw[j],0))-1)<=1e-8;
 });
}

export async function refinementSource(path,reconstruction,records,{kernel,load,loadScene,progress}){
 const groupByNode=new Map(reconstruction.groups.flatMap(g=>g.candidates.map(c=>[c.record.sha256,g.source_group_sha256]))),recordById=new Map(records.map(r=>[r.sha256,r]));
 const poses=new Map(),points=new Map(),observations=new Map(),sourceGroups=[];let camera,merger;
 try{
  for(const node of path.nodes){
   const id=groupByNode.get(node.record.sha256),record=recordById.get(id);if(!record)throw Error('Scene path has no verified image group');
   const scene=await loadSceneNode(node,{kernel,load:loadScene});
   for(const c of scene.cameras)poses.set(c.observation_sha256,c);
   for(const p of scene.points)points.set(id+'/'+p.feature_id,p);
   if(sourceGroups.some(r=>r.sha256===id))continue;
   progress(`Reading image links · group ${sourceGroups.length+1}/${path.nodes.length}…`);
   const group=await load(record);
   if(camera&&JSON.stringify(group.camera)!==JSON.stringify(camera))throw Error('Scene path calibration changed');
   camera??=group.camera;merger??=new kernel.SceneTrackGraph(JSON.stringify(camera));
   merger.push(id,JSON.stringify(group.graph));sourceGroups.push(record);
   for(const o of group.observations){
    const existing=observations.get(o.observation_sha256);
    if(existing&&(existing.capture_time_ns!==o.capture_time_ns||existing.sequence!==o.sequence))throw Error('Scene source capture identity changed');
    observations.set(o.observation_sha256,o);
   }
  }
  const ordered=[...observations.values()].filter(o=>poses.has(o.observation_sha256)).sort((a,b)=>a.capture_time_ns-b.capture_time_ns||a.sequence-b.sequence||a.observation_sha256.localeCompare(b.observation_sha256));
  if(ordered.length!==poses.size)throw Error('Scene camera has no source observation');
  return {camera,merger,poses,points,observations:ordered,sourceGroups};
 }catch(error){merger?.free();throw error}
}
