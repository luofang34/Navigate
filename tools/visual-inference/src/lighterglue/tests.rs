use super::*;

#[test]
fn assignment_requires_mutual_match_and_model_threshold() -> Result<(), InferenceError> {
    let a = [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]];
    let b = [[20.0, 40.0], [60.0, 80.0], [100.0, 120.0]];
    let scores = [0.8_f32, 0.2, 0.01, 0.7, 0.6, 0.01, 0.01, 0.02, 0.09].map(f32::ln);
    let pairs = assignment_pairs(&scores, &a, &b)?;
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].reference.x, 2.0);
    assert_eq!(pairs[0].query.x, 20.0);
    Ok(())
}

#[test]
fn assignment_rejects_invalid_values_and_handles_empty_evidence() -> Result<(), InferenceError> {
    assert!(assignment_pairs(&[], &[[0.0; 2]], &[[0.0; 2]]).is_err());
    assert!(assignment_pairs(&[f32::NAN], &[[0.0; 2]], &[[0.0; 2]]).is_err());
    assert!(assignment_pairs(&[], &[], &[[0.0; 2]])?.is_empty());
    assert!(assignment_pairs(&[], &[[0.0; 2]], &[])?.is_empty());
    Ok(())
}
