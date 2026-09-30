import {refineCandidates} from './temporal-search.js';
export async function refineReferenceCandidates(renderer,matcher,camera,candidates,image,observation,progress,{candidateLimit=128,refineLimit=12,passes=3,firstCandidateId=0}={}){
 if(!Number.isInteger(candidateLimit)||candidateLimit<1||candidateLimit>128||!Number.isInteger(refineLimit)||refineLimit<0||refineLimit>128||!Number.isInteger(passes)||passes<1||passes>3||candidates.length>128)throw Error('Invalid reference refinement budget');
 if(!Number.isSafeInteger(firstCandidateId)||firstCandidateId<0||firstCandidateId+candidates.length>2**32)throw Error('Invalid candidate ID range');
 const selected=candidates.slice(0,candidateLimit),evaluated=new Set(selected.map((_,id)=>firstCandidateId+id));
 const work={render_ms:0,matching_ms:0,geometry_ms:0,attempts:0,candidate_limit:candidateLimit,evaluated_candidate_ids:[...evaluated],
  unexamined_candidates:candidates.slice(candidateLimit).map((pose,index)=>({candidate_id:firstCandidateId+candidateLimit+index,pose})),
  search_scope:'Only evaluated candidates have geometric checks. Unexamined poses remain geographic alternatives.',
  scope:'wall-clock stage time; includes host work and device waits'};
 const first=await refineCandidates(renderer,matcher,camera,selected,image,observation,1,progress,{recoveryAllowed:false,firstCandidateId,work});
 if(passes===1)return {...first,verification_work:work};
 const proposals=first.candidate_hypotheses.filter(h=>evaluated.has(h.candidate_id)).map(h=>({id:h.candidate_id,pose:h.accepted?h:h.refinement_proposal})).filter(h=>h.pose);
 proposals.sort((a,b)=>(b.pose.spatial_support??0)-(a.pose.spatial_support??0)||a.id-b.id);
 for(const {id,pose} of proposals.slice(0,refineLimit))await refineCandidates(renderer,matcher,camera,[pose],image,observation,passes-1,progress,{recoveryAllowed:false,firstCandidateId:id,work});
 return {...JSON.parse(renderer.select()),verification_work:work};
}

export async function recoverRejectedCandidates(renderer,matcher,camera,image,observation,progress,{limit=0,passes=2}={}){
 if(!Number.isInteger(limit)||limit<0||limit>16||!Number.isInteger(passes)||passes<1||passes>3)throw Error('Invalid rejected-candidate recovery budget');
 const report=JSON.parse(renderer.select());
 if(!limit||report.candidate_hypotheses.some(h=>h.accepted))return report;
 const selected=report.candidate_hypotheses.filter(h=>h.refinement_proposal).sort((a,b)=>(b.refinement_proposal.spatial_support??0)-(a.refinement_proposal.spatial_support??0)||a.candidate_id-b.candidate_id).slice(0,limit);
 if(!selected.length)return report;
 const work={render_ms:0,matching_ms:0,geometry_ms:0,attempts:0,candidate_ids:selected.map(h=>h.candidate_id),scope:'Same observation and map evidence; recovery does not add independent confidence'};
 for(const h of selected)await refineCandidates(renderer,matcher,camera,[h.refinement_proposal],image,observation,passes,progress,{initialRecovery:true,firstCandidateId:h.candidate_id,work});
 return {...JSON.parse(renderer.select()),recovery_work:work};
}
