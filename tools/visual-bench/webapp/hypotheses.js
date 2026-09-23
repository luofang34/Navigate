export function hypotheses(frame) {
  if (frame.candidate_hypotheses) return frame.candidate_hypotheses.filter(h => h.accepted);
  return frame.accepted ? [frame] : [];
}
export function frameLabel(frame) {
  return frame.decision === 'unresolved' ? 'Unresolved alternatives' : frame.accepted ? 'Unique among evaluated' : 'Rejected';
}
