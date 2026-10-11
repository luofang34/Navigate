use super::*;

#[test]
fn assignment_requires_mutual_match_and_model_threshold() -> Result<(), InferenceError> {
    let a = [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]];
    let b = [[20.0, 40.0], [60.0, 80.0], [100.0, 120.0]];
    let scores = [0.8_f32, 0.2, 0.01, 0.7, 0.6, 0.01, 0.01, 0.02, 0.09].map(f32::ln);
    let pairs = assignment_pairs(&scores, &a, &b, false)?;
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].reference.x, 2.0);
    assert_eq!(pairs[0].query.x, 20.0);
    Ok(())
}

#[test]
fn assignment_rejects_invalid_values_and_handles_empty_evidence() -> Result<(), InferenceError> {
    assert!(assignment_pairs(&[], &[[0.0; 2]], &[[0.0; 2]], false).is_err());
    assert!(assignment_pairs(&[f32::NAN], &[[0.0; 2]], &[[0.0; 2]], false).is_err());
    assert!(assignment_pairs(&[], &[], &[[0.0; 2]], false)?.is_empty());
    assert!(assignment_pairs(&[], &[[0.0; 2]], &[], false)?.is_empty());
    Ok(())
}

#[test]
fn probability_scores_preserve_mutual_assignment_and_threshold() -> Result<(), InferenceError> {
    let a = [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]];
    let b = [[20.0, 40.0], [60.0, 80.0], [100.0, 120.0]];
    let scores = [0.8_f32, 0.2, 0.0, 0.7, 0.6, 0.01, 0.01, 0.02, 0.1];
    let probabilities = assignment_pairs(&scores, &a, &b, true)?;
    let logs = assignment_pairs(
        &scores.map(|s| s.max(f32::MIN_POSITIVE).ln()),
        &a,
        &b,
        false,
    )?;
    assert_eq!(probabilities.len(), 1);
    assert_eq!(logs.len(), probabilities.len());
    assert_eq!(logs[0].reference, probabilities[0].reference);
    assert_eq!(logs[0].query, probabilities[0].query);
    for invalid in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
        assert!(assignment_pairs(&[invalid], &a[..1], &b[..1], true).is_err());
    }
    Ok(())
}
