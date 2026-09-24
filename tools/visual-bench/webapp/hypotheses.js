export function hypotheses(frame) {
  if (frame.candidate_hypotheses) return frame.candidate_hypotheses.filter(h => h.accepted || h.tracking_supported);
  return frame.accepted ? [frame] : [];
}
export function frameLabel(frame) {
  if(frame.decision==='relative_tracking')return 'Relative tracking';
  return frame.decision === 'unresolved' ? 'Unresolved alternatives' : frame.accepted ? 'Unique among evaluated' : 'Rejected';
}
