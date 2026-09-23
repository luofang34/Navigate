import {hypotheses, frameLabel} from './hypotheses.js';

export function missionSummary(frames) {
  const geometric = frames.reduce((sum, frame) => sum + hypotheses(frame).length, 0);
  const unique = frames.filter(frame => frame.accepted && frame.decision !== 'unresolved').length;
  return `${geometric} geometric hypotheses · ${unique} unique among evaluated · ${frames.length} frames saved locally`;
}

export function resultSummary(frame, hypothesis) {
  const fixed = (value, places) => Number.isFinite(value) ? value.toFixed(places) : 'Unknown';
  return {
    title: hypothesis ? frameLabel(frame) : 'Visual observation rejected',
    location: hypothesis ? `${fixed(hypothesis.latitude_deg, 7)}, ${fixed(hypothesis.longitude_deg, 7)}` : '',
    explanation: hypothesis
      ? 'Geometric support is not a calibrated probability of the correct location.'
      : frame.reason || 'No evaluated candidate passed the geometric checks.',
    metrics: hypothesis ? [
      ['Geometric inliers', Number.isFinite(hypothesis.inliers) ? String(hypothesis.inliers) : 'Unknown'],
      ['Reprojection RMS', `${fixed(hypothesis.reprojection_rms_px, 2)} px`],
      ['Absolute accuracy', 'Not measured'],
    ] : [],
  };
}

export function executionSummary(frame){
  const observed=frame.execution?.feature_gpu_dispatches>0&&frame.execution?.matching_gpu_dispatches>0;
  const time=Number.isFinite(frame.processing_ms)?(frame.processing_ms/1000).toFixed(1)+' s':'Time unavailable';
  return (observed?'GPU work observed in this worker':'GPU work not verified')+' · '+time+' · CPU fallback can occur';
}
