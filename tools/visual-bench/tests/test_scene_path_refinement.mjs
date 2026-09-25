import assert from 'node:assert/strict';
import {refineScenePaths} from '../webapp/scene-path-refinement.js';
const poses=Array.from({length:4},(_,i)=>({observation_sha256:'frame-'+i,position_scene_units:[i,0,0],eye_to_scene_xyzw:[0,0,0,1],fixed:i===0}));
const observations=poses.map((p,i)=>({observation_sha256:p.observation_sha256,capture_time_ns:i*1e9,sequence:i}));
const point={feature_id:'9007199254740993',position_scene_units:[1,2,-4],observations:poses.map((_,camera_index)=>({camera_index,pixel:[4,5]}))};
const camera={width:10,height:10,fx:10,fy:10,cx:5,cy:5};
const root=id=>({record:{sha256:id},local_solution:{sha256:id},coordinate_frame:id,parent_scene_sha256:null,cameras:structuredClone(poses)});
const original={stage:'local_scene_reconstruction',geographic_acceptance:false,groups:[{source_group_sha256:'source',observations,candidates:[root('a'),root('b')]}]};
const frames=observations.map(o=>({...o,candidate_hypotheses:[]})),records=[{sha256:'source'}],before=JSON.stringify(original);let freed=0,selections=0;
class Graph {
 constructor(){}
 push(){}
 select(ids){selections++;return JSON.stringify({graph:{observation_sha256:JSON.parse(ids),tracks:[point]},association_sources:[{feature_id:point.feature_id,source_tracks:[{group_sha256:'source',feature_id:point.feature_id}]}]})}
 set_scene(){return 'scene-fit'}
 fit_camera(id,initial){return JSON.stringify({scene_sha256:'scene-fit',observation_sha256:id,geographic_acceptance:false,fit:id==='frame-2'?null:JSON.parse(initial)})}
 free(){freed++}
}
const saved=[];
const options={kernel:{SceneTrackGraph:Graph,
 initialize_scene_points:(_camera,_graph,scene,seeds)=>{assert.equal(JSON.parse(seeds)[0].feature_id,point.feature_id);return JSON.stringify({scene:{...JSON.parse(scene),points:[point]}})},
 triangulate_scene_points:(_camera,_graph,scene)=>JSON.stringify({scene:{...JSON.parse(scene),points:[point]},unresolved_feature_ids:[]}),
 refine_scene:(_camera,scene)=>JSON.stringify({scene:JSON.parse(scene),initial_cost:2,final_cost:1})},
 load:async()=>({camera,observations,graph:{}}),loadScene:async record=>({stage:'local_scene_reconstruction',geographic_acceptance:false,scene:{cameras:original.groups[0].candidates.find(c=>c.record.sha256===record.sha256).cameras,points:[point]}}),
 save:async text=>{saved.push(JSON.parse(text));return {sha256:'result-'+saved.length}},progress:()=>{}};
const result=await refineScenePaths(records,original,frames,options);
assert.equal(freed,2);assert.equal(result.groups.length,3);assert.equal(JSON.stringify(original),before);
assert.equal(result.groups[0],original.groups[0],'all original alternatives remain');
for(const g of result.groups.slice(1)){assert.equal(g.candidates[0].camera_fits.length,3);assert.ok(!g.candidates[0].camera_fits.some(c=>c.observation_sha256==='frame-2'))}
const dense=saved.filter(s=>s.method==='conditional_path_refinement');assert.equal(dense.length,2);
for(const s of dense){assert.deepEqual(s.unresolved_observation_sha256,['frame-2']);assert.equal(s.geographic_acceptance,false);assert.equal(s.camera_fit_evidence.length,4);assert.ok(s.source_scene_path_sha256.length===1);assert.ok(s.graph.association_sources.length===1)}
assert.notEqual(result.groups[1].candidates[0].coordinate_frame,result.groups[2].candidates[0].coordinate_frame,'separate source alternatives retain separate results');
const count=selections;assert.equal(await refineScenePaths(records,result,frames,options),result);assert.equal(selections,count,'reprocessing does not manufacture evidence or duplicate paths');
await assert.rejects(refineScenePaths(records,original,frames,{...options,save:async()=>{throw Error('Storage full')}}),/Storage full/);assert.equal(freed,3,'worker resources are released on save failure');
const failed=await refineScenePaths(records,original,frames,{...options,kernel:{...options.kernel,refine_scene(){throw Error('Unconstrained geometry')}}});
assert.equal(failed.groups.length,1);assert.equal(failed.path_refinement.attempts.length,2);assert.match(failed.path_refinement.attempts[0].reason,/Unconstrained/);
assert.equal(freed,5);assert.equal(JSON.stringify(original),before);
await assert.rejects(refineScenePaths(records,original,frames,{...options,loadScene:async()=>({geographic_acceptance:false,scene:{cameras:[]}})}),/cameras changed/);
console.info('Path refinement keeps alternatives and unsupported gaps, retains source evidence, rejects changed inputs, and releases resources on failure.');
