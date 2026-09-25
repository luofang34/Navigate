// Scheduling covers new observations without joining their coordinate frames.
function selectPaths(paths,limit){
 const remaining=paths.map(path=>({path,observations:new Set(path.nodes.flatMap(n=>(n.camera_fits??n.cameras).map(c=>c.observation_sha256)))}));
 const selected=[],covered=new Set();
 while(remaining.length&&selected.length<limit){
  let best=0,most=-1;
  for(const [index,item] of remaining.entries()){
   let count=0;for(const id of item.observations)if(!covered.has(id))count++;
   if(count>most){best=index;most=count}
  }
  const [item]=remaining.splice(best,1);selected.push(item.path);for(const id of item.observations)covered.add(id);
 }
 return {selected,deferred:remaining.map(item=>item.path)};
}

export function scenePaths(reconstruction,{maxPaths=32}={}){
 if(!Number.isInteger(maxPaths)||maxPaths<1||maxPaths>128)throw Error('Invalid scene path budget');
 const nodes=new Map(),parents=new Set();
 for(const group of reconstruction.groups)for(const candidate of group.candidates){
  if(nodes.has(candidate.record.sha256))throw Error('Repeated scene identity');
  nodes.set(candidate.record.sha256,candidate);if(candidate.parent_scene_sha256)parents.add(candidate.parent_scene_sha256);
 }
 const paths=[],visited=new Set();
 for(const [id,leaf] of nodes)if(!parents.has(id)){
  const chain=[],seen=new Set();let node=leaf;
  while(node){
   if(seen.has(node.record.sha256))throw Error('Cycle in scene lineage');seen.add(node.record.sha256);visited.add(node.record.sha256);chain.unshift(node);
   if(node.coordinate_frame!==leaf.coordinate_frame)throw Error('Scene lineage changed coordinate frame');
   const parent=node.parent_scene_sha256;node=parent?nodes.get(parent):null;if(parent&&!node)throw Error('Missing scene parent');
  }
  if(chain[0].record.sha256!==leaf.coordinate_frame)throw Error('Scene root does not identify its coordinates');
  paths.push({leaf_sha256:id,root:chain[0],nodes:chain,camera_count:new Set(chain.flatMap(n=>(n.camera_fits??n.cameras).map(c=>c.observation_sha256))).size});
 }
 if(visited.size!==nodes.size)throw Error('Cycle in scene lineage');
 const rank=path=>Number.isInteger(path.nodes.at(-1).continuation_rank)&&path.nodes.at(-1).continuation_rank>=0?path.nodes.at(-1).continuation_rank:Infinity;
 paths.sort((a,b)=>b.camera_count-a.camera_count||Number(Boolean(b.nodes.at(-1).camera_fits))-Number(Boolean(a.nodes.at(-1).camera_fits))||rank(a)-rank(b)||a.leaf_sha256.localeCompare(b.leaf_sha256));
 return selectPaths(paths,maxPaths);
}
