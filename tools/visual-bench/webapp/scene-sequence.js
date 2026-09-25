// Selection controls work per group. It does not accept a geographic location.
function continuationSet(candidates,limit){
  const ranked=[...candidates].sort((a,b)=>b.scene.cameras.length-a.scene.cameras.length||a.image_cost_per_observation-b.image_cost_per_observation);
  const selected=[],frames=new Set(),solutions=new Set();
  const add=c=>{selected.push(c);frames.add(c.coordinate_frame);solutions.add(c.local_solution.sha256)};
  for(const c of ranked)if(!frames.has(c.coordinate_frame)&&!solutions.has(c.local_solution.sha256)&&selected.length<limit)add(c);
  for(const c of ranked)if(selected.length<limit&&!selected.includes(c)&&(!frames.has(c.coordinate_frame)||!solutions.has(c.local_solution.sha256)))add(c);
  for(const c of ranked)if(selected.length<limit&&!selected.includes(c))add(c);
  return selected;
}

export async function reconstructImageGroups(records,{kernel,load,loadScene,save,progress,maxInitialPairs=3,maxContinuations=3,initialParents=[]}){
  for(const limit of [maxInitialPairs,maxContinuations])if(!Number.isInteger(limit)||limit<1||limit>8)throw Error('Invalid scene reconstruction budget');
  if(initialParents.length>8)throw Error('Too many initial scene parents');
  let previous=[];const groups=[];
  for(const parent of initialParents){
    if(!loadScene)throw Error('Saved scene parents require verification');
    await parentCloud(parent,{kernel,loadScene});previous.push(parent);
  }
  for(const [groupIndex,record] of records.entries()){
    const clouds=new Map(),group=await load(record),camera=JSON.stringify(group.camera),graph=JSON.stringify(group.graph),candidates=[],attempts=[],localSolutions=[];
    const retain=async(result,seed)=>{
      const payload={...result,source_group_sha256:record.sha256,seed,geographic_acceptance:false};
      const stored=await save(JSON.stringify(payload));localSolutions.push(stored);
      const observations=result.scene.points.reduce((n,p)=>n+p.observations.length,0);
      const meanCost=observations>0&&Number.isFinite(result.final_cost)?result.final_cost/observations:Infinity;
      const base={record:stored,local_solution:stored,scene:{...result.scene,points:[]},source_camera_indices:result.source_camera_indices,unresolved_camera_indices:result.unresolved_camera_indices,image_cost_per_observation:meanCost,parent_scene_sha256:null};
      let linked=false;
      for(const parent of previous){
        let alignments;
        const identity={local_solution_sha256:stored.sha256,parent_scene_sha256:parent.record.sha256};
        try{alignments=[JSON.parse(kernel.align_scene_groups(JSON.stringify(base.scene),JSON.stringify(parent.scene)))]}
        catch(error){
          attempts.push({...identity,reason:String(error)});
          if(!loadScene||!kernel.align_scene_groups_with_points)continue;
          progress(`Aligning image group ${groupIndex+1}/${records.length} through shared scene points…`);
          if(!clouds.has(parent.record.sha256))clouds.set(parent.record.sha256,await parentCloud(parent,{kernel,loadScene}));
          let proposals;
          try{proposals=JSON.parse(kernel.align_scene_groups_with_points(JSON.stringify(result.scene),JSON.stringify(clouds.get(parent.record.sha256))))}
          catch(error){attempts.push({...identity,stage:'shared_scene_points',reason:String(error)});continue}
          if(proposals.geographic_acceptance!==false)throw Error('Scene alignment cannot accept a geographic pose');
          attempts.push({...identity,stage:'shared_scene_points',point_proposals:proposals.point_proposals,candidate_budget_exhausted:proposals.candidate_budget_exhausted});
          alignments=proposals.candidates;
        }
        for(const alignment of alignments){
          if(alignment.geographic_acceptance!==false)throw Error('Scene alignment cannot accept a geographic pose');
          // Store the points once. The alignment record identifies their transform.
          const aligned=await save(JSON.stringify({scene:alignment.scene,local_solution:stored,source_group_sha256:record.sha256,
            alignment:{...alignment,scene:undefined},parent_scene_sha256:parent.record.sha256,coordinate_frame:parent.coordinate_frame,
            stage:'conditional_scene_alignment',geographic_acceptance:false}));
          candidates.push({...base,record:aligned,scene:alignment.scene,coordinate_frame:parent.coordinate_frame,parent_scene_sha256:parent.record.sha256});linked=true;
        }
      }
      if(!linked)candidates.push({...base,coordinate_frame:stored.sha256});
    };
    const pairs=JSON.parse(kernel.scene_seed_pairs(camera,graph));let examined=0;
    for(const pair of pairs.slice(0,maxInitialPairs)){
      examined=(examined+1)>>>0;
      progress(`Initializing image group ${groupIndex+1}/${records.length} · cameras ${pair[0]+1} and ${pair[1]+1}…`);
      let proposals;
      try{proposals=JSON.parse(kernel.scene_proposals(camera,graph,...pair))}catch(error){attempts.push({pair,reason:String(error)});continue}
      for(const {seed} of proposals){
        let result;
        try{result=JSON.parse(kernel.reconstruct_scene(camera,graph,JSON.stringify(seed)))}catch(error){attempts.push({pair,seed,reason:String(error)});continue}
        await retain(result,seed);
      }
    }
    attempts.push({seed_pairs_examined:examined,seed_pairs_available:pairs.length,seed_pairs_deferred:pairs.length-examined,scope:'bounded geometric initialization; unexamined alternatives remain'});
    if(!candidates.length)for(const parent of previous){
      progress(`Reconstructing image group ${groupIndex+1}/${records.length} from shared cameras…`);
      let seed,result;
      try{seed=kernel.seed_from_scene(graph,JSON.stringify(parent.scene));result=JSON.parse(kernel.reconstruct_scene(camera,graph,seed))}
      catch(error){attempts.push({parent_scene_sha256:parent.record.sha256,reason:String(error)});continue}
      await retain(result,JSON.parse(seed));
    }
    const next=continuationSet(candidates,maxContinuations);
    groups.push({source_group_sha256:record.sha256,observations:group.observations,local_solutions:localSolutions,
      candidates:candidates.map(candidate=>{const {scene,image_cost_per_observation,...c}=candidate;return {...c,cameras:scene.cameras,continuation:next.includes(candidate)?'scheduled':'deferred',continuation_rank:next.includes(candidate)?next.indexOf(candidate):null}}),
      attempts,geographic_acceptance:false});
    previous=next;
  }
  return {groups,stage:'local_scene_reconstruction',geographic_acceptance:false,
    evidence_correlation:'unknown; groups share observations and conditional camera estimates; each coordinate frame and alternative remains separate',
    continuation_scope:'bounded scheduling by camera coverage and image fit; saved alternatives remain unresolved and are not independent evidence'};
}

async function parentCloud(parent,{kernel,loadScene}){
  const saved=await loadScene(parent.record);
  if(saved.geographic_acceptance!==false||JSON.stringify(saved.scene?.cameras)!==JSON.stringify(parent.scene.cameras))throw Error('Saved parent scene cameras changed');
  if(parent.record.sha256===parent.local_solution.sha256){
    if(saved.stage!=='local_scene_reconstruction')throw Error('Invalid parent scene root');
    return saved.scene;
  }
  if(saved.stage!=='conditional_scene_alignment'||saved.parent_scene_sha256!==parent.parent_scene_sha256||saved.coordinate_frame!==parent.coordinate_frame||saved.local_solution?.sha256!==parent.local_solution.sha256)throw Error('Saved parent scene lineage changed');
  const local=await loadScene(parent.local_solution);
  if(local.stage!=='local_scene_reconstruction'||local.geographic_acceptance!==false||local.source_group_sha256!==saved.source_group_sha256)throw Error('Saved parent cloud source changed');
  const restored=JSON.parse(kernel.restore_aligned_scene(JSON.stringify(local.scene),JSON.stringify(saved.alignment)));
  const same=restored.cameras.length===parent.scene.cameras.length&&restored.cameras.every((c,i)=>{
    const p=parent.scene.cameras[i];return c.observation_sha256===p.observation_sha256&&c.fixed===p.fixed&&
      c.position_scene_units.every((v,j)=>Number.isFinite(v)&&Math.abs(v-p.position_scene_units[j])<=1e-8*Math.max(1,Math.abs(v)))&&
      Math.abs(Math.abs(c.eye_to_scene_xyzw.reduce((sum,v,j)=>sum+v*p.eye_to_scene_xyzw[j],0))-1)<=1e-8;
  });
  if(!same)throw Error('Restored parent cloud cameras changed');return restored;
}

export async function refineImageGroups(records,reconstruction,options){
 if(reconstruction.geographic_acceptance!==false)throw Error('Refinement requires conditional scene estimates');
 const groups=[];
 for(const record of records){
  const source=await options.load(record),observations=new Set(source.observations.map(s=>s.observation_sha256));
  const candidates=reconstruction.groups.flatMap(g=>g.candidates).map(c=>({...c,overlap:c.cameras.filter(camera=>observations.has(camera.observation_sha256)).length}));
  const parents=candidates.filter(c=>c.overlap>=3).sort((a,b)=>b.overlap-a.overlap||a.record.sha256.localeCompare(b.record.sha256)).slice(0,8);
  const initialParents=parents.map(c=>({...c,scene:{cameras:c.cameras,points:[]}}));
  const result=await reconstructImageGroups([record],{...options,initialParents});groups.push(...result.groups);
 }
 return {groups,geographic_acceptance:false};
}
