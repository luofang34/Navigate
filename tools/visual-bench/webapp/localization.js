import {diverseShortlist} from './shortlist.js';
import {TemporalSearch,refineCandidates} from './temporal-search.js';
import {headingAngles} from './matching-options.js';
import init,{Preview,propose} from './wasm/navigate_visual_preview.js';
import {LocalMatcher} from './inference/local.js';
import {ReferencePack} from './reference-pack.js';
import {read} from './storage.js';
import {rotate,gray} from './observation.js';
import {localPosition} from './geography.js';
export class LocalizationPipeline {
  constructor(matcher=new LocalMatcher(),options={}){this.matcher=matcher;this.temporal=new TemporalSearch();this.options={scales:[1],shortlist:24,candidates:3,refinements:2,...options}}
  async initialize(pack,camera,progress){await init();this.pack=pack;this.camera=camera;await this.matcher.initialize(progress);progress('Reading verified imagery and terrain…');this.references=await ReferencePack.open(pack);this.renderer=await Preview.create(JSON.stringify(pack),JSON.stringify(camera),read,false)}
  async estimate(image,prior,sequence,progress){const start=performance.now();const terrain=this.references.elevation(prior.latitude,prior.longitude);const position=localPosition(this.pack,prior.latitude,prior.longitude,terrain+prior.agl_m);
    const navigationPrior={pose:{position_enu_m:position,eye_to_enu_xyzw:[0,0,0,1]},position_radius_m:prior.radius_m,attitude_radius_rad:Math.PI};this.renderer.begin(image.gray,JSON.stringify(navigationPrior),sequence,Math.round(image.time*1e9));
    const observation=JSON.parse(this.renderer.select()).observation_sha256;
    this.temporal.lastAttempt=[];
    const seeds=this.options.temporal===false?[]:this.temporal.seeds(observation,position,prior.radius_m);
    if(seeds.length){
      const relative=await this.temporal.track(this.renderer,this.matcher,this.camera,seeds,image,observation,progress);
      let report=relative;
      if(!relative||this.temporal.previous.steps%5===4){
        progress('Checking camera hypotheses against map imagery…');
        const initials=relative?relative.candidate_hypotheses.filter(h=>h.tracking_supported):seeds;
        const checked=await refineCandidates(this.renderer,this.matcher,this.camera,initials,image,observation,Math.min(2,this.options.refinements),progress);
        if(checked.candidate_hypotheses.some(h=>h.accepted))report=this.temporal.label(checked,seeds);
      }
      if(report){
        this.temporal.remember(report,image);
        return {...report,navigation_prior:navigationPrior,requested_time_s:image.requested_time_s,timing_scope:image.timing,anchor_lat_lon:this.pack.anchor_lat_lon,
          retrieval:{algorithm:relative?'conditional_camera_tracking':'previous_pose_seeds_then_current_image_geometry',shortlist_pairs:0,search_scope:'previous hypotheses only; unexamined geographic alternatives remain',stage:'retrieval_only',searched_pairs:0,map_crops:0,pose_candidates:seeds.length,evaluated_candidates:seeds.length},
          execution:this.matcher.diagnostics?.()??{execution:'adapter does not report device metrics'},stage_ms:{retrieval:0,learned_matching:0,geometry:performance.now()-start},processing_ms:performance.now()-start};
      }
      this.renderer.begin(image.gray,JSON.stringify(navigationPrior),sequence,Math.round(image.time*1e9));
    }
    const temporalMs=performance.now()-start;const crops=this.references.crops(prior,this.camera,this.options.scales),candidates=[];let searched=0;
    const coarseWidth=Math.round(image.width/Math.max(image.width,image.height)*80)*8,coarseHeight=Math.round(image.height/Math.max(image.width,image.height)*80)*8;
    const coarse=gray(image.canvas,coarseWidth,coarseHeight);
    const queries=headingAngles(this.options.headings).map(angle=>{const rotated=rotate(coarse,angle),unrotate=rotated.unrotate;rotated.unrotate=p=>{const q=unrotate(p);return [(q[0]+.5)*image.width/coarseWidth-.5,(q[1]+.5)*image.height/coarseHeight-.5]};return {image:rotated,angle,key:`${observation}/angle/${angle}`}});
    searched=crops.length*queries.length;
    const indices=this.matcher.retrievePairs?await this.matcher.retrievePairs(crops,queries,(this.options.diverse?crops.length*queries.length:this.options.shortlist),progress):queries.flatMap((_,q)=>crops.map((_,r)=>({reference_index:r,query_index:q})));
    const selected=this.options.diverse?diverseShortlist(indices,crops,queries,this.options.shortlist):indices;
    const shortlist=selected.map(({reference_index:r,query_index:q})=>({crop:crops[r],rotated:queries[q].image,angle:queries[q].angle,keys:{reference:crops[r].key,query:queries[q].key}}));
    const retrievalEnd=performance.now();
    for(const [index,{crop,rotated,angle,keys}] of shortlist.entries()){
      progress(`Comparing images ${index+1}/${shortlist.length} · ${candidates.length} proposals`);const {pairs}=await this.matcher.matchImages(crop.image,rotated,keys);if(pairs.length<12)continue;
      const ground=pairs.flatMap(p=>{const world=crop.world(p.reference);return world?[{world,query:rotated.unrotate(p.query)}]:[]});const proposal=JSON.parse(propose(JSON.stringify(this.camera),JSON.stringify(ground)));
      if(!proposal.retrieved||proposal.retrieval_inliers<10)continue;
      if(Math.hypot(...proposal.position_enu_m.map((v,i)=>v-position[i]))>prior.radius_m)continue;
      candidates.push({...proposal,angle,crop:crop.key});
    }
    const matchingEnd=performance.now();
    candidates.sort((a,b)=>b.retrieval_inliers-a.retrieval_inliers);
    const chosen=candidates.slice(0,this.options.candidates);
    const result=await refineCandidates(this.renderer,this.matcher,this.camera,chosen,image,observation,this.options.refinements,progress);
    this.temporal.remember(result,image);
    return {...result,tracking_attempts:this.temporal.lastAttempt,navigation_prior:navigationPrior,requested_time_s:image.requested_time_s,timing_scope:image.timing,anchor_lat_lon:this.pack.anchor_lat_lon,retrieval:{algorithm:'descriptor_shortlist_then_regional_planar_proposals',shortlist_pairs:shortlist.length,search_scope:'bounded retrieval; unexamined alternatives can remain',stage:'retrieval_only',searched_pairs:searched,map_crops:crops.length,pose_candidates:candidates.length,evaluated_candidates:chosen.length},execution:this.matcher.diagnostics?.()??{execution:"adapter does not report device metrics"},stage_ms:{temporal_attempt:temporalMs,retrieval:retrievalEnd-start-temporalMs,learned_matching:matchingEnd-retrievalEnd,geometry:performance.now()-matchingEnd},processing_ms:performance.now()-start};
  }
  close(){this.matcher?.close();this.renderer?.free()}
}
