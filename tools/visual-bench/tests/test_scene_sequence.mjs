import assert from 'node:assert/strict';
import {reconstructImageGroups} from '../webapp/scene-sequence.js';
const records=[{sha256:'source-a'},{sha256:'source-b'}],stored=[],progress=[];
const load=async record=>({camera:{width:320,height:240},observations:[{observation_sha256:record.sha256}],graph:{observation_sha256:[record.sha256]}});
let solved=0;
const kernel={
 scene_seed_pairs:()=>JSON.stringify([[0,2],[1,3],[2,4]]),
 scene_proposals:(_camera,_graph,first,second)=>JSON.stringify([{seed:{camera_indices:[first,second],branch:first+1}}]),
 reconstruct_scene:(_camera,graph,seed)=>{solved++;const {branch}=JSON.parse(seed);return JSON.stringify({scene:{cameras:[{observation_sha256:JSON.parse(graph).observation_sha256[0],position_scene_units:[branch,0,0]}],points:[{feature_id:'1',observations:[{},{}]}]},source_camera_indices:[0],unresolved_camera_indices:[],final_cost:branch,geographic_acceptance:false})},
 seed_from_scene:(_graph,scene)=>JSON.stringify({camera_indices:[0,2],branch:JSON.parse(scene).cameras[0].position_scene_units[0]}),
 align_scene_groups:(source,target)=>JSON.stringify({scene:{...JSON.parse(source),cameras:JSON.parse(source).cameras.map(c=>({...c,position_scene_units:JSON.parse(target).cameras[0].position_scene_units}))},geographic_acceptance:false,observation_sha256:['shared']})
};
const options={kernel,load,maxInitialPairs:2,maxContinuations:2,progress:text=>progress.push(text),save:async text=>{stored.push(JSON.parse(text));return {sha256:'scene-'+stored.length}}};
const result=await reconstructImageGroups(records,options);
assert.equal(solved,4,'each group has fresh image initialization even when a parent pose is available');
assert.equal(result.geographic_acceptance,false);assert.equal(result.groups.length,2);
assert.deepEqual(result.groups[0].candidates.map(c=>c.coordinate_frame),['scene-1','scene-2']);
assert.deepEqual(result.groups[1].candidates.map(c=>c.coordinate_frame),['scene-1','scene-2','scene-1','scene-2']);
assert.deepEqual(result.groups[1].candidates.map(c=>c.cameras[0].position_scene_units),[[1,0,0],[2,0,0],[1,0,0],[2,0,0]],'alternatives are not averaged');
const scheduled=result.groups[1].candidates.filter(c=>c.continuation==='scheduled');
assert.equal(scheduled.length,2);
assert.equal(new Set(scheduled.map(c=>c.local_solution.sha256)).size,2,'one low-residual local solution cannot consume the complete continuation budget');
assert.equal(new Set(scheduled.map(c=>c.coordinate_frame)).size,2,'distinct parent coordinate alternatives retain continuation slots');
assert.equal(result.groups[1].candidates.filter(c=>c.continuation==='deferred').length,2,'bounded scheduling retains the unscheduled candidates');
assert.equal(stored.filter(s=>s.scene.points.length).length,4,'each local point cloud is stored once, separate from alignment alternatives');
assert.deepEqual(result.groups[1].candidates.map(c=>c.parent_scene_sha256),['scene-1','scene-2','scene-1','scene-2']);
assert.equal(stored[3].local_solution.sha256,'scene-3');assert.equal(stored[3].parent_scene_sha256,'scene-1');
assert.equal(stored.every(s=>s.geographic_acceptance===false),true);assert.equal(result.groups[0].attempts.at(-1).seed_pairs_examined,2);
assert.equal(stored[3].alignment.scene,undefined,'alignment metadata does not duplicate its scene');
assert.ok(progress.length>0);
const broken={...options,kernel:{...kernel,align_scene_groups(){throw Error('overlap lacks scale')}}};
const restarted=await reconstructImageGroups(records,broken);
assert.equal(restarted.groups[1].attempts.filter(a=>a.reason?.includes('overlap lacks scale')).length,4);
assert.ok(restarted.groups[1].candidates.every(c=>!restarted.groups[0].candidates.some(p=>p.coordinate_frame===c.coordinate_frame)),'failed alignment starts an unresolved component');
let saves=0;
await assert.rejects(reconstructImageGroups(records,{...options,save:async()=>{if(++saves===4)throw Error('OPFS full');return {sha256:'write-'+saves}}}),/OPFS full/);
const empty=await reconstructImageGroups(records,{...options,kernel:{...kernel,scene_seed_pairs:()=>JSON.stringify([])}});
assert.ok(empty.groups.every(g=>g.candidates.length===0),'unsupported groups remain explicit');
const fallback=await reconstructImageGroups(records,{...options,kernel:{...kernel,scene_seed_pairs:(_c,g)=>JSON.stringify(JSON.parse(g).observation_sha256[0]==='source-a'?[[0,2]]:[])}});
assert.equal(fallback.groups[1].candidates[0].coordinate_frame,fallback.groups[0].candidates[0].coordinate_frame,'shared estimates remain a fallback when image initialization has no result');
solved=0;
const bounded=await reconstructImageGroups([...records,{sha256:'source-c'}],options);
assert.equal(solved,6,'aligned alternatives do not multiply expensive reconstruction calls');
assert.equal(bounded.groups[2].candidates.length,4,'each fresh solution is compared only with the scheduled parents');
console.info('Independent group initialization, bounded continuations, saved alternatives, lineage, and storage failures passed.');

const clouds=new Map();let cloudWrites=0,pointChecks=0;
const pointOptions={...options,save:async text=>{const value=JSON.parse(text),record={sha256:'point-scene-'+(++cloudWrites)};clouds.set(record.sha256,value);return record},
 loadScene:async record=>{const value=clouds.get(record.sha256);if(!value)throw Error('Missing saved cloud');return structuredClone(value)},
 kernel:{...kernel,reconstruct_scene:(...args)=>{const value=JSON.parse(kernel.reconstruct_scene(...args));value.stage='local_scene_reconstruction';value.scene.cameras=value.scene.cameras.map(c=>({...c,fixed:false,eye_to_scene_xyzw:[0,0,0,1]}));return JSON.stringify(value)},
  align_scene_groups(){throw Error('stationary overlap')},
  restore_aligned_scene:scene=>scene,
  align_scene_groups_with_points:(source,target)=>{
   const a=JSON.parse(source),b=JSON.parse(target);assert.equal(a.points.length,1);assert.equal(b.points.length,1,'alignment loads the exact parent cloud, including aligned parent groups');pointChecks++;
   return JSON.stringify({geographic_acceptance:false,point_proposals:12,candidate_budget_exhausted:true,candidates:[1,2].map(label=>({scene:{...a,points:[]},scale:1,source_to_target_xyzw:[0,0,0,1],translation_target_scene_units:[0,0,0],geographic_acceptance:false,point_associations:[{source_feature_id:String(label),target_feature_id:'parent-'+label}]}))});
  }
 }};
const linked=await reconstructImageGroups([...records,{sha256:'source-c'}],pointOptions);
assert.equal(pointChecks,8);assert.equal(linked.groups[1].candidates.length,8);assert.equal(linked.groups[2].candidates.length,8,'both conditional scale proposals remain separate for each parent');
assert.ok(linked.groups[2].candidates.every(c=>linked.groups[0].candidates.some(p=>p.coordinate_frame===c.coordinate_frame)));
assert.equal(linked.groups[1].attempts.filter(a=>a.stage==='shared_scene_points'&&a.candidate_budget_exhausted).length,4);
assert.equal([...clouds.values()].filter(c=>c.stage==='local_scene_reconstruction').length,6,'each original local point cloud is stored once');
assert.ok([...clouds.values()].filter(c=>c.stage==='conditional_scene_alignment').every(c=>c.alignment.point_associations.length===1&&c.scene.points.length===0));
assert.equal(linked.geographic_acceptance,false);
await assert.rejects(reconstructImageGroups(records,{...pointOptions,loadScene:async()=>{throw Error('OPFS cloud is missing')}}),/OPFS cloud is missing/);
await assert.rejects(reconstructImageGroups(records,{...pointOptions,loadScene:async record=>{const value=await pointOptions.loadScene(record);value.scene.cameras[0].position_scene_units[0]+=1;return value}}),/parent scene cameras changed/);
console.info('Point-supported alignment retains alternative parent chains, exact saved clouds, explicit budgets, and storage failures.');

const parent=linked.groups[0].candidates[0],external={...parent,scene:{cameras:parent.cameras,points:[]}};
const resumed=await reconstructImageGroups([records[1]],{...pointOptions,initialParents:[external]});
assert.ok(resumed.groups[0].candidates.every(c=>c.parent_scene_sha256===parent.record.sha256),'resampling extends the verified exact parent rather than replacing its lineage');
await assert.rejects(reconstructImageGroups([records[1]],{...pointOptions,initialParents:[{...external,scene:{cameras:[],points:[]}}]}),/parent scene cameras changed/);

solved=0;
const completeAlternatives=await reconstructImageGroups([records[0]],{...options,maxInitialPairs:3,maxContinuations:1,
 kernel:{...kernel,reconstruct_scene:(...args)=>{const result=JSON.parse(kernel.reconstruct_scene(...args));result.final_cost=4-result.final_cost;return JSON.stringify(result)}}});
const complete=completeAlternatives.groups[0];
assert.equal(solved,3,'full camera coverage cannot stop other seed pairs in the search budget');
assert.equal(complete.candidates.length,3,'different complete solutions remain available');
assert.equal(complete.candidates.find(c=>c.continuation==='scheduled').cameras[0].position_scene_units[0],3,'a later complete solution can receive the continuation slot');
assert.equal(complete.attempts.at(-1).seed_pairs_examined,3);
assert.equal(complete.attempts.at(-1).seed_pairs_deferred,0);
assert.equal(completeAlternatives.geographic_acceptance,false);
console.info('Complete coverage preserves other seed checks and later solutions within the search budget.');
