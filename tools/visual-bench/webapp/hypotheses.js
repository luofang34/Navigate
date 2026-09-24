export function hypotheses(frame) {
  if (frame.candidate_hypotheses) return frame.candidate_hypotheses.filter(h => h.accepted || h.tracking_supported);
  return frame.accepted ? [frame] : [];
}
export function frameLabel(frame) {
  if(frame.decision==='search_deferred')return 'No pose · search deferred';
  if(frame.decision==='relative_tracking')return 'Relative tracking';
  return frame.decision === 'unresolved' ? 'Unresolved alternatives' : frame.accepted ? 'Unique among evaluated' : 'Rejected';
}

export function branchKey(frame,h){return h.track_id??JSON.stringify(h.tracking_anchor??{observation_sha256:h.anchor_observation_sha256??frame.observation_sha256,candidate_id:h.candidate_id,map_manifest_sha256:h.map_manifest_sha256})}
