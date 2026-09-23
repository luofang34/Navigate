import {hypotheses, frameLabel} from './hypotheses.js';

export function missionSummary(frames) {
  const geometric = frames.reduce((sum, frame) => sum + hypotheses(frame).length, 0);
  const unique = frames.filter(frame => frame.accepted && frame.decision !== 'unresolved').length;
  return `${geometric} geometric hypotheses · ${unique} unique among evaluated · ${frames.length} frames saved locally`;
}

export function resultSummary(frame, hypothesis) {
  const fixed = (value, places) => Number.isFinite(value) ? value.toFixed(places) : 'Unknown';
  const reasons=[...new Set((frame.candidate_hypotheses||[]).filter(h=>!h.accepted&&h.reason).map(h=>h.reason))];
  const retrieval=frame.retrieval;
  const noCandidate=retrieval?.map_crops>0&&retrieval.pose_candidates===0&&retrieval.evaluated_candidates===0&&!(frame.candidate_hypotheses?.length);
  const rejection=noCandidate
    ? 'The searched imagery produced no pose candidate. Geometric checks did not run. Check the reference area and camera settings.'
    : [frame.reason||'No evaluated candidate passed the geometric checks.',...reasons].join(' ');
  const count=value=>Number.isInteger(value)&&value>=0?String(value):'Unknown';
  return {
    title: hypothesis ? frameLabel(frame) : noCandidate ? 'No location candidate found' : 'Visual observation rejected',
    location: hypothesis ? `${fixed(hypothesis.latitude_deg, 7)}, ${fixed(hypothesis.longitude_deg, 7)}` : '',
    explanation: hypothesis
      ? 'Geometric support is not a calibrated probability of the correct location.'
      : rejection,
    metrics: hypothesis ? [
      ['Geometric inliers', Number.isFinite(hypothesis.inliers) ? String(hypothesis.inliers) : 'Unknown'],
      ['Reprojection RMS', `${fixed(hypothesis.reprojection_rms_px, 2)} px`],
      ['Absolute accuracy', 'Not measured'],
    ] : frame.retrieval ? [
      ['Reference crops', count(retrieval.map_crops)],
      ['Pose candidates', count(retrieval.pose_candidates)],
      ['Candidates checked', count(retrieval.evaluated_candidates)],
    ] : [],
  };
}

export function executionSummary(frame){
  const observed=frame.execution?.feature_gpu_dispatches>0&&frame.execution?.matching_gpu_dispatches>0;
  const time=Number.isFinite(frame.processing_ms)?(frame.processing_ms/1000).toFixed(1)+' s':'Time unavailable';
  return (observed?'GPU work observed in this worker':'GPU work not verified')+' · '+time+' · CPU fallback can occur';
}
