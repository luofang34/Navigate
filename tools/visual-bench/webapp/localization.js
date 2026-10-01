import {refineReferenceCandidates,recoverRejectedCandidates} from './candidate-refinement.js';
import {similarityCandidates} from './retrieval-recovery.js';
import {cropPoseCandidates} from './retrieval-poses.js';
import {refineScenePaths} from './scene-path-refinement.js';
import {registerScenePaths} from './scene-map.js';
import {diverseShortlist} from './shortlist.js';
import {TemporalSearch,refineCandidates,localMapContinuations} from './temporal-search.js';
import {headingAngles} from './matching-options.js';
import {LocalMatcher} from './inference/local.js';
import {ReferencePack} from './reference-pack.js';
import {read,saveImageTracks,loadImageTracks,saveSceneReconstruction,loadSceneReconstruction,saveSceneRegistration} from './storage.js';
import {SequenceImages} from './sequence-images.js';
import {reconstructImageGroups,refineImageGroups} from './scene-sequence.js';
import {rotate,gray,consumeOriginalPixels} from './observation.js';
import {localPosition,geographicPosition} from './geography.js';
const STRONG_PROPOSAL_INLIERS=30;
// Buffered frames that catch a stale pose up to the live frame check the map less often, so tracking
// gains on the source; the pose still receives a map check every few buffered hops.
const CATCH_UP_MAP_FACTOR=3;
// A search around the last supported pose compares as many image pairs per map crop as the search of
// the whole prior, never fewer than enough to fill the refined candidate set.
export function searchBudget({shortlist,candidates},crops,priorCrops){
  if(!priorCrops)return shortlist;
  return Math.min(shortlist,Math.max(candidates*3,Math.ceil(shortlist*crops/priorCrops)));
}
export class LocalizationPipeline {
  constructor(matcher=new LocalMatcher(),options={},referenceSearch=null){this.matcher=matcher;this.referenceSearch=referenceSearch;this.temporal=new TemporalSearch();this.options={scales:[1],shortlist:24,candidates:3,refinements:2,mapMotionFraction:0,...options}}
  beginSequence({maxFrames=64}={}){this.temporal=new TemporalSearch();this.priorCrops=null;this.imageSequence?.close();this.imageSequence=new SequenceImages({create:()=>new this.SceneTracks(JSON.stringify(this.camera)),matcher:this.matcher,save:saveImageTracks,camera:this.camera,maxFrames})}
  async finishSequence(){return await this.imageSequence?.finish()??[]}
  async reconstructSequence(groups,progress){return reconstructImageGroups(groups,{kernel:this.sceneKernel,load:loadImageTracks,loadScene:loadSceneReconstruction,save:saveSceneReconstruction,progress})}
  async refineScenePaths(groups,reconstruction,frames,progress){return refineScenePaths(groups,reconstruction,frames,{kernel:this.sceneKernel,load:loadImageTracks,loadScene:loadSceneReconstruction,save:saveSceneReconstruction,progress})}
  async refineScenes(groups,reconstruction,progress){return refineImageGroups(groups,reconstruction,{kernel:this.sceneKernel,load:loadImageTracks,loadScene:loadSceneReconstruction,save:saveSceneReconstruction,progress})}
  async observeScene(image,prior,sequence,expected,progress){
    if(image.width!==this.camera.width||image.height!==this.camera.height)throw Error('Scene image calibration changed');
    this.renderer.begin(image.gray,JSON.stringify(this.navigationPrior(prior)),sequence,Math.round(image.time*1e9));
    const frame=JSON.parse(this.renderer.select());
    if(expected&&frame.observation_sha256!==expected)throw Error('Scene source pixels do not match the saved observation');
    await this.imageSequence.observe(image,frame.observation_sha256,sequence,progress);
    return {...frame,accepted:false,decision:'search_deferred',candidate_hypotheses:[],requested_time_s:image.requested_time_s,timing_scope:image.timing,
      reason:'Extra scene sample; geographic search was not run.',execution:this.matcher.diagnostics?.()};
  }
  async registerScene(plan,image,prior,progress){return registerScenePaths(plan,image,prior,{renderer:this.renderer,matcher:this.matcher,kernel:this.sceneKernel,camera:this.camera,pack:this.pack,navigationPrior:p=>this.navigationPrior(p),load:loadSceneReconstruction,save:saveSceneRegistration,progress})}
  async initialize(pack,camera,progress){const {default:init,Preview,propose,propose_nadir,SceneTracks,...sceneKernel}=await import('./wasm/navigate_visual_preview.js');await init();this.propose=propose;this.proposeNadir=propose_nadir;this.SceneTracks=SceneTracks;this.sceneKernel=sceneKernel;this.pack=pack;this.camera=camera;await this.matcher.initialize(progress);progress('Reading verified imagery and terrain…');this.references=await ReferencePack.open(pack);this.renderer=await Preview.create(JSON.stringify(pack),JSON.stringify(camera),read,false)}
  // After lost tracking the camera is near its last supported pose. The search area grows with
  // elapsed time at the maximum expected speed and never exceeds the navigation prior.
  // Reference features depend only on the map, the prior and the camera, so a host can compute them
  // before the first frame arrives; the first area search then reads them from the feature cache.
  async prepareReferences(prior,progress){
    const crops=this.references.crops(prior,this.camera,this.options.scales);
    await this.matcher.prepareReferences?.(crops,progress);return crops.length;
  }
  reacquisitionArea(prior,seeds,captureTimeNs){
    const last=this.temporal.previous;if(!seeds.length||!last||!Number.isFinite(last.capture_time_ns))return prior;
    const seconds=Math.max(0,(captureTimeNs-last.capture_time_ns)/1e9),radius=Math.min(prior.radius_m,(this.options.reacquisitionRadiusM??40)+(this.options.maximumSpeedMps??25)*seconds);
    if(radius>=prior.radius_m)return prior;
    // The prior height keeps the crop plan inside the limit the prior itself was checked against.
    return {...prior,...geographicPosition(this.pack,seeds[0].position_enu_m.slice(0,2)),radius_m:radius,search_scope:'last supported pose'};
  }
  async estimate(image,prior,sequence,progress){const start=performance.now();consumeOriginalPixels(image);const terrain=this.references.elevation(prior.latitude,prior.longitude);const position=localPosition(this.pack,prior.latitude,prior.longitude,terrain+prior.agl_m);
    const navigationPrior=this.navigationPrior(prior,position);this.renderer.begin(image.gray,JSON.stringify(navigationPrior),sequence,Math.round(image.time*1e9));
    const observation=JSON.parse(this.renderer.select()).observation_sha256;
    const adjacent=image.timing!=='still image'?await this.imageSequence?.observe(image,observation,sequence,progress):null;
    this.temporal.lastAttempt=[];
    const temporalEnabled=this.options.temporal!==false&&image.timing!=='still image';
    const captureTimeNs=Math.round(image.time*1e9),seeds=temporalEnabled?this.temporal.seeds(observation,position,prior.radius_m):[];
    // Supported tracking and map checks carry the pose; a timed area search is an explicit host choice.
    let regionalDue=!temporalEnabled||(this.options.regionalIntervalSeconds!=null&&this.temporal.regionalDue(captureTimeNs,this.options.regionalIntervalSeconds))||(!seeds.length&&this.temporal.regionalDue(captureTimeNs,this.options.recoveryIntervalSeconds??5)),relativeFallback=null,localCheck=null,localFallback=null;
    if(seeds.length){
      const relative=await this.temporal.track(this.renderer,this.matcher,this.camera,seeds,image,observation,progress,adjacent?.reference===this.temporal.previous?.observation?adjacent.result:undefined,{alternatives:!image.catch_up});
      let report=relative;
      // A failed catch-up hop retries a closer buffered frame, which costs less than a map check.
      if((!relative&&!image.catch_up)||this.temporal.mapDue(captureTimeNs,(this.options.mapIntervalSeconds??5)*(image.catch_up?CATCH_UP_MAP_FACTOR:1),this.options.mapMotionFraction?{candidates:relative.candidate_hypotheses.filter(h=>h.tracking_supported),camera:this.camera,agl_m:prior.agl_m,fraction:this.options.mapMotionFraction}:undefined)){
        progress('Checking camera hypotheses against map imagery…');
        const initials=relative?relative.candidate_hypotheses.filter(h=>h.tracking_supported):seeds;
        localCheck=await refineCandidates(this.renderer,this.matcher,this.camera,initials,image,observation,this.options.refinements,progress,{acceptedPasses:2});
        if(localCheck.candidate_hypotheses.some(h=>h.accepted))report=localFallback=await this.localUpdate(localCheck,initials,relative,image,prior,progress);
      }
      if(report&&!regionalDue){
        this.temporal.remember(report,image,{map:Boolean(localCheck)});
        return {...report,...(localCheck?{local_map_check:localCheck}:{}),navigation_prior:navigationPrior,requested_time_s:image.requested_time_s,timing_scope:image.timing,anchor_lat_lon:this.pack.anchor_lat_lon,
          retrieval:{algorithm:report.decision==='relative_tracking'?'conditional_camera_tracking':'previous_pose_seeds_then_current_image_geometry',shortlist_pairs:0,search_scope:'previous hypotheses only; unexamined geographic alternatives remain',stage:'retrieval_only',searched_pairs:0,map_crops:0,pose_candidates:seeds.length,evaluated_candidates:seeds.length},
          execution:this.matcher.diagnostics?.()??{execution:'adapter does not report device metrics'},stage_ms:{retrieval:0,learned_matching:0,geometry:performance.now()-start},processing_ms:performance.now()-start};
      }
      relativeFallback=relative;
      if(!report)regionalDue||=this.temporal.recoveryDue(captureTimeNs,this.options.recoveryIntervalSeconds??5);
      this.renderer.begin(image.gray,JSON.stringify(navigationPrior),sequence,Math.round(image.time*1e9));
    }
    if(!regionalDue){
      const attempted=JSON.parse(this.renderer.select()),report={...attempted,accepted:false,decision:'search_deferred',candidate_hypotheses:[],local_attempt:localCheck??attempted,
        reason:'Tracking did not produce a supported pose. Geographic search was not run for this sampled frame.',
        navigation_prior:navigationPrior,requested_time_s:image.requested_time_s,timing_scope:image.timing,anchor_lat_lon:this.pack.anchor_lat_lon,
        retrieval:{algorithm:'scheduled_geographic_search',stage:'not_run',last_search_time_ns:this.temporal.lastRegionalSearchNs,search_scope:'deferred until the geographic search interval; this observation has no supported pose',searched_pairs:0,map_crops:0,pose_candidates:0,evaluated_candidates:0},
        execution:this.matcher.diagnostics?.()??{execution:'adapter does not report device metrics'},processing_ms:performance.now()-start};
      // The last supported pose stays the tracking reference until the recovery interval ends.
      if(seeds.length)this.temporal.markLost(captureTimeNs);else this.temporal.remember(report,image,{map:Boolean(localCheck)});
      return report;
    }
    progress('Searching the map area…');
    if(this.referenceSearch){
      const orientations=headingAngles(8).map(angle=>[0,0,Math.sin(angle*Math.PI/360),Math.cos(angle*Math.PI/360)]),retrieved=await this.referenceSearch.propose(image,prior,this.camera,this.references,progress,{orientations});
      const retrievalMs=performance.now()-start,checked=await refineReferenceCandidates(this.renderer,this.matcher,this.camera,retrieved.candidates,image,observation,progress,{candidateLimit:this.options.referenceCandidates??128});
      const recovered=await this.recoverCandidates(checked,image,observation,progress);
      const result=this.temporal.reacquired(recovered,relativeFallback,localFallback);this.temporal.remember(result,image,{regional:true});
      return {...result,navigation_prior:navigationPrior,anchor_lat_lon:this.pack.anchor_lat_lon,requested_time_s:image.requested_time_s,timing_scope:image.timing,
        retrieval:{algorithm:'indexed_references_then_rendered_geometry',stage:'retrieval_only',search_scope:retrieved.scope,backend_identity:retrieved.backend_identity,input_image:retrieved.input_image,reference_ids:retrieved.reference_ids,unsupported_references:retrieved.unsupported_references,pose_candidates:retrieved.candidates.length,evaluated_candidates:checked.verification_work.evaluated_candidate_ids.length,unexamined_candidate_ids:checked.verification_work.unexamined_candidates.map(h=>h.candidate_id)},
        execution:{matching:this.matcher.diagnostics?.(),retrieval:this.referenceSearch.diagnostics?.()},stage_ms:{retrieval:retrievalMs,verification:performance.now()-start-retrievalMs},processing_ms:performance.now()-start};
    }
    const temporalMs=performance.now()-start;let stoppedEarly=null;
    const search=this.reacquisitionArea(prior,seeds,captureTimeNs),searchPosition=localPosition(this.pack,search.latitude,search.longitude,this.references.elevation(search.latitude,search.longitude)+search.agl_m);
    const crops=this.references.crops(search,this.camera,this.options.scales),candidates=[],groundMatches=[];let searched=0;
    const coarseWidth=Math.round(image.width/Math.max(image.width,image.height)*80)*8,coarseHeight=Math.round(image.height/Math.max(image.width,image.height)*80)*8;
    const coarse=gray(image.canvas,coarseWidth,coarseHeight);
    const queries=headingAngles(this.options.headings).map(angle=>{const rotated=rotate(coarse,angle),unrotate=rotated.unrotate;rotated.unrotate=p=>{const q=unrotate(p);return [(q[0]+.5)*image.width/coarseWidth-.5,(q[1]+.5)*image.height/coarseHeight-.5]};return {image:rotated,angle,key:`${observation}/angle/${angle}`}});
    searched=crops.length*queries.length;
    if(search===prior)this.priorCrops=crops.length;
    const budget=searchBudget(this.options,crops.length,search===prior?crops.length:this.priorCrops);
    const indices=this.matcher.retrievePairs?await this.matcher.retrievePairs(crops,queries,(this.options.diverse?crops.length*queries.length:budget),progress):queries.flatMap((_,q)=>crops.map((_,r)=>({reference_index:r,query_index:q})));
    const selected=this.options.diverse?diverseShortlist(indices,crops,queries,budget):indices;
    const shortlist=selected.map(({reference_index:r,query_index:q})=>({crop:crops[r],rotated:queries[q].image,angle:queries[q].angle,keys:{reference:crops[r].key,query:queries[q].key,progress}}));
    const retrievalEnd=performance.now();
    for(const [index,{crop,rotated,angle,keys}] of shortlist.entries()){
      progress(`Comparing images ${index+1}/${shortlist.length} · ${candidates.length} proposals`);const {pairs}=await this.matcher.matchImages(crop.image,rotated,keys);if(pairs.length<12)continue;
      const ground=pairs.flatMap(p=>{const world=crop.world(p.reference);return world?[{world,query:rotated.unrotate(p.query)}]:[]});const proposal=JSON.parse(this.propose(JSON.stringify(this.camera),JSON.stringify(ground)));
      if(this.options.similarityCandidates)groundMatches.push({ground,source_index:index});
      if(!proposal.retrieved||proposal.retrieval_inliers<10)continue;
      if(Math.hypot(...proposal.position_enu_m.map((v,i)=>v-position[i]))>prior.radius_m||Math.hypot(...proposal.position_enu_m.slice(0,2).map((v,i)=>v-searchPosition[i]))>search.radius_m)continue;
      candidates.push({...proposal,angle,crop:crop.key,crop_center:crop.world([(crop.image.width-1)/2,(crop.image.height-1)/2])});
      // The shortlist is in rank order and only the strongest `candidates` proposals are refined.
      if(candidates.filter(c=>c.retrieval_inliers>=STRONG_PROPOSAL_INLIERS).length>=this.options.candidates){stoppedEarly=index+1;break}
    }
    const matchingEnd=performance.now();
    candidates.sort((a,b)=>b.retrieval_inliers-a.retrieval_inliers);
    const chosen=candidates.slice(0,this.options.candidates);
    let checked=await refineCandidates(this.renderer,this.matcher,this.camera,chosen,image,observation,this.options.refinements,progress,{recoveryAllowed:false});
    let cropSeeds=[];
    if(!checked.candidate_hypotheses.some(h=>h.accepted)&&this.options.cropSeedRegions){
      cropSeeds=cropPoseCandidates(candidates,position,prior,this.options.cropSeedRegions);
      checked=await refineCandidates(this.renderer,this.matcher,this.camera,cropSeeds,image,observation,this.options.refinements,progress,{recoveryAllowed:false,firstCandidateId:chosen.length});
    }
    checked=await this.recoverCandidates(checked,image,observation,progress);
    let similaritySeeds=[];
    if(!checked.candidate_hypotheses.some(h=>h.accepted)&&this.options.similarityCandidates){
      progress('Checking extra camera seeds…');
      similaritySeeds=similarityCandidates(groundMatches,this.proposeNadir,this.camera,position,prior.radius_m);
      const firstCandidateId=chosen.length+cropSeeds.length;
      const extra=await refineReferenceCandidates(this.renderer,this.matcher,this.camera,similaritySeeds,image,observation,progress,{candidateLimit:this.options.similarityCandidates,firstCandidateId});
      checked={...checked,...extra,seed_verification_work:extra.verification_work};
    }
    const result=this.temporal.reacquired(checked,relativeFallback,localFallback);
    this.temporal.remember(result,image,{regional:true});
    return {...result,...(localCheck?{local_map_check:localCheck}:{}),tracking_attempts:this.temporal.lastAttempt,navigation_prior:navigationPrior,requested_time_s:image.requested_time_s,timing_scope:image.timing,anchor_lat_lon:this.pack.anchor_lat_lon,retrieval:{algorithm:'descriptor_shortlist_then_regional_planar_proposals',shortlist_pairs:shortlist.length,compared_pairs:stoppedEarly??shortlist.length,search_area:{latitude:search.latitude,longitude:search.longitude,radius_m:search.radius_m,scope:search.search_scope??'navigation prior'},shortlist_stop:stoppedEarly?`${this.options.candidates} proposals with at least ${STRONG_PROPOSAL_INLIERS} retrieval inliers`:'shortlist exhausted',search_scope:'bounded retrieval; unexamined alternatives can remain',stage:'retrieval_only',searched_pairs:searched,map_crops:crops.length,pose_candidates:candidates.length+cropSeeds.length+similaritySeeds.length,similarity_pose_candidates:similaritySeeds.length,similarity_pose_scope:'auxiliary near-nadir seeds; final pose attitude is unrestricted',crop_pose_candidates:cropSeeds.length,crop_pose_scope:'auxiliary near-nadir seeds; final pose attitude is unrestricted',evaluated_candidates:chosen.length+cropSeeds.length+(checked.seed_verification_work?.evaluated_candidate_ids.length??0)},execution:this.matcher.diagnostics?.()??{execution:"adapter does not report device metrics"},stage_ms:{temporal_attempt:temporalMs,retrieval:retrievalEnd-start-temporalMs,learned_matching:matchingEnd-retrievalEnd,geometry:performance.now()-matchingEnd},processing_ms:performance.now()-start};
  }
  async recoverCandidates(checked,image,observation,progress){
    if(!this.options.recoveryCandidates)return checked;
    return {...checked,...await recoverRejectedCandidates(this.renderer,this.matcher,this.camera,image,observation,progress,{limit:this.options.recoveryCandidates,passes:2})};
  }
  async localUpdate(checked,seeds,relative,image,prior,progress){
    if(!this.options.localMotionCheck||!relative)return this.temporal.label(checked,seeds,relative);
    const previous=this.temporal.previous,tracked=relative.candidate_hypotheses.filter(h=>h.tracking_supported),mapped=checked.candidate_hypotheses.filter(h=>h.accepted);
    const candidates=[...tracked,...mapped].map((h,candidate_id)=>({...h,candidate_id}));
    const motion=await this.checkMotion({report:previous.report,image:previous.image},checked,image,prior,candidates,progress);
    const permitted=localMapContinuations(motion.checks,tracked,mapped,seeds);
    return {...this.temporal.label(checked,seeds,relative,permitted),local_motion_check:{...motion,minimum_support_retention:.5,
      policy:'conditional continuity only; retain incompatible map alternatives without moving the existing camera path',
      candidate_sources:candidates.map((h,i)=>({check_candidate_id:i,source:i<tracked.length?'relative':'map',source_candidate_id:(i<tracked.length?tracked[i]:mapped[i-tracked.length]).candidate_id}))}};
  }
  navigationPrior(prior,position){
    position??=localPosition(this.pack,prior.latitude,prior.longitude,this.references.elevation(prior.latitude,prior.longitude)+prior.agl_m);
    return {pose:{position_enu_m:position,eye_to_enu_xyzw:[0,0,0,1]},position_radius_m:prior.radius_m,attitude_radius_rad:Math.PI};
  }
  async trackFrom(reference,image,prior,sequence,progress){
    const start=performance.now(),navigationPrior=this.navigationPrior(prior),previous=reference.report;
    for(const pixels of [reference.image,image])if(pixels.width!==this.camera.width||pixels.height!==this.camera.height)throw Error('Tracking image calibration differs from the active camera');
    const supported=previous.candidate_hypotheses.filter(h=>h.accepted||h.tracking_supported);
    if(!this.pack.pack_id||supported.some(h=>h.map_manifest_sha256!==this.pack.pack_id))throw Error('Tracking reference uses different or unknown map data');
    this.renderer.begin(reference.image.gray,JSON.stringify(navigationPrior),previous.sequence,previous.capture_time_ns);
    if(JSON.parse(this.renderer.select()).observation_sha256!==previous.observation_sha256)throw Error('Tracking reference pixels do not match the saved observation');
    const temporal=new TemporalSearch();temporal.remember(previous,reference.image);
    this.renderer.begin(image.gray,JSON.stringify(navigationPrior),sequence,Math.round(image.time*1e9));
    const observation=JSON.parse(this.renderer.select()).observation_sha256,seeds=temporal.seeds(observation,navigationPrior.pose.position_enu_m,prior.radius_m);
    if(!seeds.length)return null;
    const tracked=await temporal.track(this.renderer,this.matcher,this.camera,seeds,image,observation,progress);
    const report=tracked??{...JSON.parse(this.renderer.select()),accepted:false,decision:'rejected',candidate_hypotheses:temporal.lastAttempt??[],reason:'No relative camera hypothesis passed the geometric checks.'};
    return {...report,navigation_prior:navigationPrior,requested_time_s:image.requested_time_s,timing_scope:image.timing,anchor_lat_lon:this.pack.anchor_lat_lon,
      retrieval:{algorithm:'conditional_camera_tracking',stage:'retrieval_only',search_scope:'supplied reference hypotheses only; no independent geographic search',searched_pairs:0,map_crops:0,pose_candidates:seeds.length,evaluated_candidates:seeds.length},
      execution:this.matcher.diagnostics?.()??{execution:'adapter does not report device metrics'},processing_ms:performance.now()-start};
  }
  async checkMotion(reference,observation,image,prior,candidates,progress){
    const previous=reference.report,navigationPrior=this.navigationPrior(prior),seeds=previous.candidate_hypotheses.filter(h=>h.accepted||h.tracking_supported);
    for(const pixels of [reference.image,image])if(pixels.width!==this.camera.width||pixels.height!==this.camera.height)throw Error('Motion check image calibration differs from the active camera');
    if(!this.pack.pack_id||[...seeds,...candidates].some(h=>h.map_manifest_sha256!==this.pack.pack_id))throw Error('Motion check uses different or unknown map data');
    this.renderer.begin(reference.image.gray,JSON.stringify(navigationPrior),previous.sequence,previous.capture_time_ns);
    if(JSON.parse(this.renderer.select()).observation_sha256!==previous.observation_sha256)throw Error('Motion check reference pixels do not match the saved observation');
    this.renderer.begin(image.gray,JSON.stringify(navigationPrior),observation.sequence,Math.round(image.time*1e9));
    if(JSON.parse(this.renderer.select()).observation_sha256!==observation.observation_sha256)throw Error('Motion check pixels do not match the saved observation');
    progress('Checking camera motion…');
    const {pairs,backend_identity}=await this.matcher.matchImages(reference.image,image,{reference:previous.observation_sha256+'/query',query:observation.observation_sha256+'/query',stage:'tracking',progress}),checks=[];
    for(const [id,seed] of seeds.entries()){
      await this.renderer.render_reference(id,JSON.stringify(seed));
      const stamp={candidate_id:id,sequence:previous.sequence,capture_time_ns:previous.capture_time_ns};
      for(const candidate of candidates){
        const result=JSON.parse(this.renderer.check_tracking_pose(reference.image.gray,JSON.stringify(stamp),JSON.stringify(candidate),JSON.stringify(pairs),backend_identity));
        checks.push({...result,reference_candidate_id:seed.candidate_id,candidate_id:candidate.candidate_id});
      }
    }
    return {checks,observation_sha256:observation.observation_sha256,reference_observation_sha256:previous.observation_sha256,
      evidence_correlation:'unknown; candidate and reference poses share observations and map data; checks are not independent evidence'};
  }
  async refineAt(observation,image,prior,candidates,progress){
    const start=performance.now(),navigationPrior=this.navigationPrior(prior);
    if(image.width!==this.camera.width||image.height!==this.camera.height)throw Error('Refinement image calibration differs from the active camera');
    if(!this.pack.pack_id||candidates.some(h=>h.map_manifest_sha256!==this.pack.pack_id))throw Error('Refinement seeds use different or unknown map data');
    this.renderer.begin(image.gray,JSON.stringify(navigationPrior),observation.sequence,Math.round(image.time*1e9));
    const identity=JSON.parse(this.renderer.select()).observation_sha256;
    if(identity!==observation.observation_sha256)throw Error('Refinement pixels do not match the saved observation');
    const checked=await refineCandidates(this.renderer,this.matcher,this.camera,candidates,image,identity,this.options.refinements,progress);
    const accepted=checked.candidate_hypotheses.filter(h=>h.accepted);
    return {...checked,accepted:false,decision:accepted.length?'unresolved':'rejected',
      reason:'Current-image map checks from estimated pose seeds. Unexamined geographic alternatives remain.',
      evidence_correlation:'unknown; shared map data and estimated search seeds; repeated checks add no independent confidence',
      search_seeds:candidates.map(h=>({observation_sha256:h.observation_sha256,candidate_id:h.candidate_id,tracking_anchor:h.tracking_anchor,role:'estimated pose initializes current-image geometry'})),
      navigation_prior:navigationPrior,requested_time_s:image.requested_time_s,timing_scope:image.timing,anchor_lat_lon:this.pack.anchor_lat_lon,
      retrieval:{algorithm:'pose_seeds_then_current_image_geometry',stage:'retrieval_only',search_scope:'supplied reference hypotheses only; no independent geographic search',searched_pairs:0,map_crops:0,pose_candidates:candidates.length,evaluated_candidates:candidates.length},
      execution:this.matcher.diagnostics?.()??{execution:'adapter does not report device metrics'},processing_ms:performance.now()-start};
  }
  close(){this.referenceSearch?.close();this.imageSequence?.close();this.matcher?.close();this.renderer?.free()}
}
