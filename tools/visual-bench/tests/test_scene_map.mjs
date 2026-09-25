import assert from 'node:assert/strict';
import {sceneMapPlans,appendScenePaths,registerScenePaths} from '../webapp/scene-map.js';
import {hypotheses} from '../webapp/hypotheses.js';
import {supportedPose,connectedSamples} from '../webapp/pose-playback.js';
import {sequenceTrack} from '../webapp/input-sequence.js';
import {resultSummary} from '../webapp/result-summary.js';
const camera=id=>({observation_sha256:id,position_scene_units:[0,0,0],eye_to_scene_xyzw:[0,0,0,1],fixed:false});
const node=(id,parent,cameras)=>({record:{sha256:id},coordinate_frame:'root',parent_scene_sha256:parent,cameras:cameras.map(camera)});
const root=node('root',null,['a','b']),left=node('left','root',['b','c']),right=node('right','root',['b','d']);
const reconstruction={groups:[{candidates:[root]},{candidates:[left,right]}]};
const frame=(id,index)=>({observation_sha256:id,sequence:index,capture_time_ns:index*1e9,accepted:false,candidate_hypotheses:id==='a'?[{candidate_id:9,accepted:true,inliers:30,map_manifest_sha256:'map'}]:[]});
const frames=['a','b','c','d'].map(frame);
const schedule=sceneMapPlans(reconstruction,frames);
assert.equal(schedule.plans.length,1);assert.equal(schedule.plans[0].paths.length,2);
assert.deepEqual(schedule.plans[0].paths.map(p=>p.nodes.map(n=>n.record.sha256)),[['root','left'],['root','right']]);
assert.deepEqual(sceneMapPlans(reconstruction,frames,{maxPaths:1}).deferred_leaf_sha256,['right']);
assert.equal(sceneMapPlans(reconstruction,frames.map(f=>({...f,candidate_hypotheses:[]}))).unregistered_root_sha256[0],'root');
assert.throws(()=>sceneMapPlans({groups:[{candidates:[node('child','missing',['c'])]}]},frames),/Missing scene parent/);
assert.throws(()=>sceneMapPlans({groups:[{candidates:[root,root]}]},frames),/Repeated scene identity/);
assert.throws(()=>sceneMapPlans({groups:[{candidates:[root,{...left,coordinate_frame:'other'}]}]},frames),/changed coordinate/);
const renderer={begin(){},select:()=>JSON.stringify({observation_sha256:'a'}),render_reference:async()=>new Uint8Array(48),
 register_scene:()=>JSON.stringify({observation_sha256:'a',map_manifest_sha256:'map',geographic_acceptance:false,candidates:[{inlier_indices:[0,1,2],point_fit_rms_m:.4}],candidate_budget_exhausted:false})};
let matches=0,writes=0;
const options={renderer,matcher:{matchImages:async()=>{matches++;return {pairs:[],backend_identity:'test'}}},
 kernel:{registered_scene_cameras:scene=>JSON.stringify(JSON.parse(scene).cameras.map(c=>({...c,position_enu_m:[1,2,3],eye_to_enu_xyzw:c.eye_to_scene_xyzw,latitude_deg:40,longitude_deg:-74})))},
 camera:{width:96,height:72},pack:{pack_id:'map'},navigationPrior:()=>({}),
 load:async()=>({scene:{cameras:root.cameras,points:[]},stage:'local_scene_reconstruction',geographic_acceptance:false}),
 save:async text=>{assert.equal(JSON.parse(text).source_scene.sha256,'root');writes++;return {sha256:'registration'}},progress:()=>{}};
const report=await registerScenePaths(schedule.plans[0],{gray:new Uint8Array(48)},{},options);
assert.equal(matches,1);assert.equal(writes,1);assert.equal(report.geographic_acceptance,false);
const paths=report.registrations[0].candidates[0].paths;
assert.deepEqual(paths.map(p=>p.cameras.map(c=>c.observation_sha256)),[['a','b','c'],['a','b','d']],'siblings sharing axes cannot be combined into one path');
assert.equal(appendScenePaths(frames,report.registrations),6);
assert.equal(appendScenePaths(frames,report.registrations),0,'reprocessing does not duplicate poses or confidence');
assert.ok(frames[0].candidate_hypotheses[0].track_id.includes('left'),'preview order retains the scheduled path order');
assert.equal(hypotheses(frames[0]).length,3,'map alternative remains alongside both scene paths');
assert.equal(frames[0].accepted,false);assert.equal(frames[0].decision,'scene_registration');
const leftA=frames[0].candidate_hypotheses.find(h=>h.track_id?.includes('left')),leftB=frames[1].candidate_hypotheses.find(h=>h.track_id?.includes('left'));
assert.ok(supportedPose(leftA));assert.ok(connectedSamples({time:0,h:leftA},{time:.2,h:leftB}));
const rightB=frames[1].candidate_hypotheses.find(h=>h.track_id?.includes('right'));
assert.equal(connectedSamples({time:0,h:leftA},{time:.2,h:rightB}),false);
assert.equal(leftA.tracking_supported,undefined);assert.equal(leftA.accepted,false);assert.equal(leftA.geographic_accuracy,'not_independently_measured');
const summary=resultSummary(frames[0],leftA);assert.match(summary.explanation,/estimated map alignment/);assert.equal(summary.metrics[0][0],'Registration links');assert.equal(summary.metrics[1][0],'Point fit RMS');
await assert.rejects(registerScenePaths(schedule.plans[0],{gray:[]},{},{...options,renderer:{...renderer,select:()=>JSON.stringify({observation_sha256:'different'})}}),/observation identity/);
assert.equal(matches,1,'identity mismatch stops before inference');
await assert.rejects(registerScenePaths(schedule.plans[0],{gray:[]},{},{...options,pack:{pack_id:'different'}}),/map identity/);
await assert.rejects(registerScenePaths(schedule.plans[0],{gray:[]},{},{...options,save:async()=>{throw Error('OPFS full')}}),/OPFS full/);
const failed=await registerScenePaths(schedule.plans[0],{gray:[]},{},{...options,renderer:{...renderer,register_scene(){throw Error('missing surface')}}});
assert.equal(failed.registrations.length,0);assert.match(failed.attempts[0].reason,/missing surface/);
console.info('Scene map lineage, alternatives, worker identities, missing surfaces, storage failures, and conditional preview passed.');

const limited=await registerScenePaths({...schedule.plans[0],preview_path_budget:1},{gray:[]},{},options);
assert.equal(limited.preview_paths,1);assert.equal(limited.deferred_preview_paths,1);
assert.equal(limited.registrations[0].candidates[0].paths.length,1);
const noDisplay=await registerScenePaths({...schedule.plans[0],preview_path_budget:0},{gray:[]},{},options);
assert.equal(noDisplay.registrations.length,1,'registration is stored even when its display expansion is deferred');
assert.equal(noDisplay.preview_paths,0);assert.equal(noDisplay.deferred_preview_paths,2);

const exported=sequenceTrack(frames);
assert.equal(exported.features.filter(f=>f.geometry.type==='LineString').length,2);
assert.equal(exported.features.filter(f=>f.properties.scene_supported&&f.properties.accepted).length,0);

const rejectedFrames=['a','b','c','d'].map(frame);
rejectedFrames[0].candidate_hypotheses=[{candidate_id:4,accepted:false,map_manifest_sha256:'map',
 reference_image_sha256:'reference',reference_depth_sha256:'depth',reference_pose:{position_enu_m:[1,2,3],eye_to_enu_xyzw:[0,0,0,1]},reason:'single-image geometry failed'}];
const fallback=sceneMapPlans(reconstruction,rejectedFrames).plans[0];
assert.equal(fallback.anchor.seeds[0].accepted,false);
assert.equal(fallback.anchor.seeds[0].initialization_stage,'reference_proposal');
assert.deepEqual(fallback.anchor.seeds[0].position_enu_m,[1,2,3]);
assert.equal(rejectedFrames[0].candidate_hypotheses[0].accepted,false);
const rejected=await registerScenePaths(fallback,{gray:[]},{},{...options,renderer:{...renderer,register_scene(){throw Error('too few scene links')}}});
assert.equal(rejected.registrations.length,0,'retrieval cannot bypass the scene geometric check');
rejectedFrames[0].candidate_hypotheses[0].reference_depth_sha256=null;
assert.equal(sceneMapPlans(reconstruction,rejectedFrames).plans.length,0,'untraced proposals cannot initialize registration');

assert.throws(()=>sceneMapPlans({groups:[{candidates:[node('cycle-a','cycle-b',['a']),node('cycle-b','cycle-a',['b'])]}]},frames),/Cycle in scene lineage/,'a component with no leaf cannot silently disappear');
const localLeft={...left,local_solution:{sha256:'left-cloud'}},laterReconstruction={groups:[{candidates:[root]},{candidates:[localLeft,right]}]};
const laterFrames=['a','b','c','d'].map(frame).map(f=>({...f,candidate_hypotheses:[]}));
laterFrames[2].candidate_hypotheses=[{candidate_id:12,accepted:true,inliers:42,map_manifest_sha256:'map'}];
const laterSchedule=sceneMapPlans(laterReconstruction,laterFrames);
assert.equal(laterSchedule.plans.length,1);const laterPlan=laterSchedule.plans[0];
assert.equal(laterPlan.source.record.sha256,'left');assert.equal(laterPlan.anchor.frame.observation_sha256,'c');
assert.deepEqual(laterPlan.paths.map(p=>p.leaf_sha256),['left']);
assert.deepEqual(laterSchedule.unregistered_leaf_sha256,['right'],'a sibling does not inherit a descendant registration');
assert.deepEqual(laterSchedule.unregistered_root_sha256,[],'the later group can anchor its exact chain back to the root');
const aligned={stage:'conditional_scene_alignment',geographic_acceptance:false,scene:{cameras:localLeft.cameras,points:[]},
 local_solution:localLeft.local_solution,parent_scene_sha256:'root',coordinate_frame:'root',source_group_sha256:'group',
 alignment:{scale:2,source_to_target_xyzw:[0,0,0,1],translation_target_scene_units:[0,0,0]}};
const cloud={stage:'local_scene_reconstruction',geographic_acceptance:false,source_group_sha256:'group',scene:{cameras:localLeft.cameras,points:[{feature_id:'17',position_scene_units:[1,2,3],observations:[]}]}};
let restored=0;
const laterOptions={...options,load:async record=>record.sha256==='left'?structuredClone(aligned):structuredClone(cloud),
 kernel:{...options.kernel,restore_aligned_scene:(scene,alignment)=>{restored++;assert.equal(JSON.parse(alignment).scale,2);const result=JSON.parse(scene);result.points[0].position_scene_units=[2,4,6];return JSON.stringify(result)}},
 renderer:{...renderer,select:()=>JSON.stringify({observation_sha256:'c'}),register_scene:(id,scene)=>{
  assert.deepEqual(JSON.parse(scene).points[0].position_scene_units,[2,4,6]);return JSON.stringify({...JSON.parse(renderer.register_scene()),observation_sha256:'c'});
 }},save:async text=>{const saved=JSON.parse(text);assert.equal(saved.source_scene.sha256,'left');assert.equal(saved.coordinate_frame,'root');return {sha256:'later-registration'}}};
const laterResult=await registerScenePaths(laterPlan,{gray:[]},{},laterOptions);
assert.equal(restored,1);assert.equal(laterResult.source_scene_sha256,'left');
assert.deepEqual(laterResult.registrations[0].candidates[0].paths[0].cameras.map(c=>c.observation_sha256),['a','b','c']);
assert.equal(appendScenePaths(laterFrames,laterResult.registrations),3);
assert.equal(laterFrames[3].candidate_hypotheses.length,0);
await assert.rejects(registerScenePaths({...laterPlan,paths:schedule.plans[0].paths},{gray:[]},{},laterOptions),/outside the scene path/);
await assert.rejects(registerScenePaths(laterPlan,{gray:[]},{},{...laterOptions,load:async record=>({...await laterOptions.load(record),parent_scene_sha256:'other'})}),/changed lineage/);
await assert.rejects(registerScenePaths(laterPlan,{gray:[]},{},{...laterOptions,load:async record=>({...await laterOptions.load(record),source_group_sha256:record.sha256})}),/changed source/);
await assert.rejects(registerScenePaths(laterPlan,{gray:[]},{},{...laterOptions,kernel:{...laterOptions.kernel,restore_aligned_scene:()=>JSON.stringify({cameras:[camera('wrong')],points:[]})}}),/differ from the saved alignment/);
assert.equal(sceneMapPlans(reconstruction,frames).plans.length,1,'one source observation is not scheduled repeatedly through shared scene groups');
console.info('Later scene anchors retain their exact cloud, parent chain, and deferred sibling paths; corrupt or cyclic lineage is rejected.');

const segment=(id,observations)=>({...node(id,null,observations),coordinate_frame:id});
const coverageFrames=['p','q','r','s','t','u'].map(frame);
const longAlternatives=Array.from({length:8},(_,i)=>segment('long-'+i,['p','q','r','s']));
const facade=segment('facade',['t','u']);
const limitedCoverage={groups:[{candidates:[...longAlternatives,facade]}]};
const beforeCoverage=JSON.stringify(limitedCoverage);
const coverageSchedule=sceneMapPlans(limitedCoverage,coverageFrames,{maxPaths:2});
assert.deepEqual(coverageSchedule.unregistered_leaf_sha256,['long-0','facade'],'a short new interval receives work before duplicate long-path alternatives');
assert.deepEqual(coverageSchedule.unregistered_root_sha256,['long-0','facade'],'unsupported registration remains explicit for each separate scene');
assert.equal(coverageSchedule.deferred_leaf_sha256.length,7);
assert.ok(!coverageSchedule.deferred_leaf_sha256.includes('facade'));
assert.equal(coverageSchedule.plans.length,0,'scheduling new coverage does not manufacture map evidence');
assert.equal(JSON.stringify(limitedCoverage),beforeCoverage,'coverage scheduling preserves every alternative and camera estimate');
console.info('Registration work covers separate short intervals before repeated long-path alternatives.');

const ranked={groups:[{candidates:[root]},{candidates:[{...left,continuation_rank:1},{...right,continuation_rank:0}]}]};
const rankedBefore=JSON.stringify(ranked),rankedPlan=sceneMapPlans(ranked,['a','b','c','d'].map(frame),{maxPaths:1});
assert.equal(rankedPlan.plans[0].paths[0].leaf_sha256,'right','equal-coverage paths retain the solver work order instead of a hash order');
assert.deepEqual(rankedPlan.deferred_leaf_sha256,['left']);
assert.equal(JSON.stringify(ranked),rankedBefore);assert.equal(rankedPlan.geographic_acceptance,false);
console.info('Preview scheduling retains solver work order without discarding other paths or accepting geography.');

const anchorFrames=Array.from({length:11},(_,i)=>({observation_sha256:'anchor-'+i,sequence:i,capture_time_ns:i*1e9,accepted:false,
 candidate_hypotheses:[{candidate_id:0,accepted:true,inliers:i===5?500:Math.abs(i-5)===1?400:30,map_manifest_sha256:'map'}]}));
const anchorScene={groups:[{candidates:[segment('anchors',anchorFrames.map(f=>f.observation_sha256))]}]};
const anchorInputs=JSON.stringify([anchorScene,anchorFrames]);
const distributed=sceneMapPlans(anchorScene,anchorFrames,{maxAnchors:3});
assert.deepEqual(distributed.plans.map(p=>p.anchor.index),[5,0,10],'registration checks cover the sequence instead of spending all slots on neighboring high-inlier frames');
assert.equal(JSON.stringify([anchorScene,anchorFrames]),anchorInputs,'spreading work changes neither poses nor image evidence');
assert.equal(distributed.geographic_acceptance,false);
anchorFrames[0].candidate_hypotheses=[{candidate_id:0,accepted:false,map_manifest_sha256:'map',reference_image_sha256:'image',reference_depth_sha256:'depth',reference_pose:{position_enu_m:[0,0,0],eye_to_enu_xyzw:[0,0,0,1]}}];
const geometricAnchors=sceneMapPlans(anchorScene,anchorFrames,{maxAnchors:3});
assert.deepEqual(geometricAnchors.plans.map(p=>p.anchor.index),[5,10,1],'sequence spacing does not promote a rejected pose over available geometric seeds');
console.info('Bounded map-registration checks span source observations while retaining geometry priority and all alternatives.');

const fullFits=Array.from({length:300},(_,i)=>({...camera('dense-'+i),position_scene_units:[i,0,0]}));
const refinedRoot={record:{sha256:'refined'},coordinate_frame:'refined',parent_scene_sha256:null,cameras:fullFits.slice(0,2),camera_fits:fullFits};
const refinedFrames=fullFits.map((c,i)=>frame(c.observation_sha256,i));refinedFrames[0].candidate_hypotheses=[{accepted:true,map_manifest_sha256:'map'}];
const refinedPlan=sceneMapPlans({groups:[{candidates:[refinedRoot]}]},refinedFrames).plans[0];
const storedRefinement={stage:'local_scene_reconstruction',method:'conditional_path_refinement',geographic_acceptance:false,scene:{cameras:refinedRoot.cameras,points:[]},camera_fits:fullFits,camera_fit_scene_sha256:'fit-scene',
 camera_fit_evidence:fullFits.map(c=>({scene_sha256:'fit-scene',geographic_acceptance:false,observation_sha256:c.observation_sha256,fit:{position_scene_units:c.position_scene_units,eye_to_scene_xyzw:c.eye_to_scene_xyzw}}))};
const batches=[];
const refinedOptions={...options,load:async()=>structuredClone(storedRefinement),save:async()=>({sha256:'dense-registration'}),
 renderer:{...renderer,select:()=>JSON.stringify({observation_sha256:'dense-0'}),register_scene:()=>JSON.stringify({...JSON.parse(renderer.register_scene()),observation_sha256:'dense-0'})},
 kernel:{registered_scene_cameras:scene=>{const cameras=JSON.parse(scene).cameras;batches.push(cameras.length);assert.ok(cameras.length<=129);return options.kernel.registered_scene_cameras(scene)}}};
const full=await registerScenePaths(refinedPlan,{gray:[]},{},refinedOptions);
assert.deepEqual(batches,[129,129,42]);assert.equal(full.registrations[0].candidates[0].paths[0].cameras.length,300);
assert.equal(appendScenePaths(refinedFrames,full.registrations),300);assert.ok(refinedFrames.every(f=>!f.accepted));
const changedFits=structuredClone(storedRefinement);changedFits.camera_fits[2].position_scene_units[0]++;
await assert.rejects(registerScenePaths(refinedPlan,{gray:[]},{},{...refinedOptions,load:async()=>changedFits}),/preview cameras changed/);
const changedSupport=structuredClone(storedRefinement);changedSupport.camera_fit_evidence[2].scene_sha256='unrelated';
await assert.rejects(registerScenePaths(refinedPlan,{gray:[]},{},{...refinedOptions,load:async()=>changedSupport}),/evidence changed/);
const unsupported=structuredClone(storedRefinement);unsupported.camera_fit_evidence[2].fit=null;
await assert.rejects(registerScenePaths(refinedPlan,{gray:[]},{},{...refinedOptions,load:async()=>unsupported}),/support changed/);
console.info('Full-frame refined paths use bounded transforms and reject changed camera evidence before preview.');
