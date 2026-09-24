export class TemporalSearch {
  constructor(){this.previous=null}
  seeds(observation,position,radius){
    if(!this.previous||this.previous.observation===observation)return [];
    return this.previous.candidates.filter(c=>Math.hypot(...c.position_enu_m.map((v,i)=>v-position[i]))<=radius);
  }
  remember(report,image){
    const candidates=report.candidate_hypotheses.filter(h=>h.accepted||h.tracking_supported).map(h=>({
      candidate_id:h.candidate_id,position_enu_m:[...h.position_enu_m],eye_to_enu_xyzw:[...h.eye_to_enu_xyzw],
      tracking_anchor:h.tracking_anchor??{observation_sha256:report.observation_sha256,candidate_id:h.candidate_id,map_manifest_sha256:h.map_manifest_sha256},
    }));
    const steps=report.decision==='relative_tracking'?((this.previous?.steps??0)+1)>>>0:0;
    this.previous=candidates.length?{observation:report.observation_sha256,candidates,steps,
      image:image?{gray:image.gray.slice(),width:image.width,height:image.height}:null,
      sequence:report.sequence,capture_time_ns:report.capture_time_ns}:null;
  }
  async track(renderer,matcher,camera,seeds,image,observation,progress){
    if(!this.previous?.image||typeof renderer.track!=='function')return null;
    progress('Tracking between camera frames…');
    const keys={reference:`${this.previous.observation}/query`,query:`${observation}/query`,stage:'refinement',progress};
    const verify=async({pairs,backend_identity})=>{
      const proposals=[];
      for(const [id,seed] of seeds.entries()){
        await renderer.render_reference(id,JSON.stringify(seed));
        const prior={candidate_id:id,sequence:this.previous.sequence,capture_time_ns:this.previous.capture_time_ns};
        const result=JSON.parse(renderer.track(this.previous.image.gray,JSON.stringify(prior),JSON.stringify(pairs),backend_identity));
        proposals.push({...result,tracking_anchor:seed.tracking_anchor,parent_candidate_id:seed.candidate_id});
      }
      this.lastAttempt=proposals;return proposals;
    };
    let proposals=await verify(await matcher.matchImages(this.previous.image,image,keys));
    if(!proposals.some(h=>h.tracking_supported)&&matcher.matchAlternatives){
      for await(const matches of matcher.matchAlternatives(this.previous.image,image,keys)){
        proposals=await verify(matches);if(proposals.some(h=>h.tracking_supported))break;
      }
    }
    if(!proposals.some(h=>h.tracking_supported))return null;
    return {...JSON.parse(renderer.select()),accepted:false,decision:'relative_tracking',candidate_hypotheses:proposals,
      reason:'Relative camera tracking. This pose depends on the initial map hypothesis and rendered terrain depth.',
      evidence_correlation:'unknown; tracking inherits the anchor pose and shared map evidence'};
  }
  label(report,seeds){
    const ids=report.candidate_hypotheses.filter(h=>h.accepted).map(h=>h.candidate_id);
    return {...report,accepted:false,decision:'unresolved',unresolved_candidate_ids:ids,
      reason:'Tracked geometric hypotheses. The full reference area was not searched again.',
      search_seeds:{observation_sha256:this.previous.observation,candidate_ids:seeds.map(c=>c.candidate_id),role:'estimated poses used only to initialize current-image geometry'},
      evidence_correlation:'unknown; consecutive observations share map data and estimated search seeds'};
  }
}

export function poseConverged(first,second){
  const distance=Math.hypot(...first.position_enu_m.map((v,i)=>v-second.position_enu_m[i]));
  const dot=Math.abs(first.eye_to_enu_xyzw.reduce((sum,v,i)=>sum+v*second.eye_to_enu_xyzw[i],0));
  return distance<.1&&2*Math.acos(Math.min(1,dot))<.001;
}

export async function refineCandidates(renderer,matcher,camera,candidates,image,observation,passes,progress){
  for(let id=0;id<candidates.length;id++){
    let candidate=candidates[id];
    for(let pass=0;pass<passes;pass++){
      progress(`Checking camera pose · candidate ${id+1}/${candidates.length} · refinement ${pass+1}`);
      const pixels=await renderer.render_reference(id,JSON.stringify(candidate));
      const {pairs,backend_identity}=await matcher.matchImages({gray:pixels,width:camera.width,height:camera.height},image,{query:`${observation}/query`,stage:'refinement',progress});
      const report=JSON.parse(renderer.refine(id,JSON.stringify(pairs),backend_identity));
      const next=report.accepted?report:report.refinement_proposal;
      if(!next||poseConverged(candidate,next))break;
      candidate=next;
    }
  }
  return JSON.parse(renderer.select());
}
