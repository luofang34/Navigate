use super::*;

#[test]
fn compact_output_retains_mutuality_threshold_and_index_validation() -> Result<(), InferenceError> {
    let a = [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]];
    let b = [[20.0, 40.0], [60.0, 80.0], [100.0, 120.0]];
    let result = pairs(&[0, 0, 2], &[0, 1, 2], &[0.8, 0.7, 0.1], &a, &b)?;
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].reference.x, 2.0);
    assert_eq!(result[0].query.x, 20.0);
    for bad in [-1, 3, i64::MAX] {
        assert!(pairs(&[bad, 0, 2], &[0, 1, 2], &[0.8, 0.7, 0.1], &a, &b).is_err());
        assert!(pairs(&[0, 0, 2], &[bad, 1, 2], &[0.8, 0.7, 0.1], &a, &b).is_err());
    }
    assert!(pairs(&[0, 0, 2], &[0, 1, 2], &[f32::NAN, 0.7, 0.1], &a, &b).is_err());
    assert!(pairs(&[0, 0], &[0, 1, 2], &[0.8, 0.7], &a, &b).is_err());
    Ok(())
}

#[test]
fn padding_never_produces_correspondences() -> Result<(), InferenceError> {
    let a = [[2.0, 4.0]];
    let b = [[20.0, 40.0], [60.0, 80.0]];
    let result = pairs(&[1, 0, 2], &[1, 0, 2], &[0.8, 0.9, 0.9], &a, &b)?;
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].query.x, 60.0);
    assert!(pairs(&[2, 0, 1], &[1, 2, 0], &[0.9; 3], &a, &b)?.is_empty());
    Ok(())
}

#[test]
fn scoring_retains_exact_pair_order_and_raw_mutual_assignment_values() -> Result<(), InferenceError>
{
    let a = [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]];
    let b = [[20.0, 40.0], [60.0, 80.0], [100.0, 120.0]];
    let rows = [2, 1, 0];
    let columns = [2, 1, 0];
    let scores = [0.25, 0.1, 0.875];
    let plain = pairs(&rows, &columns, &scores, &a, &b)?;
    let scored = map_pairs(
        &rows,
        &columns,
        &scores,
        &a,
        &b,
        |pair, assignment_score| crate::ScoredPixelMatch {
            pair,
            assignment_score,
        },
    )?;
    assert_eq!(plain.len(), scored.len());
    for (a, b) in plain.iter().zip(&scored) {
        assert_eq!(a.reference, b.pair.reference);
        assert_eq!(a.query, b.pair.query);
    }
    assert_eq!(
        scored
            .iter()
            .map(|v| v.assignment_score)
            .collect::<Vec<_>>(),
        vec![0.25, 0.875]
    );
    Ok(())
}
