use super::{StageLatencies, percentile};
use serde_json::{Map, Value};

#[test]
fn nearest_rank_percentiles_of_one_to_one_hundred() {
    let samples: Vec<f64> = (1..=100).rev().map(f64::from).collect();
    assert_eq!(percentile(&samples, 50.0), Some(50.0));
    assert_eq!(percentile(&samples, 95.0), Some(95.0));
    assert_eq!(percentile(&samples, 99.0), Some(99.0));
    assert_eq!(percentile(&samples, 100.0), Some(100.0));
    assert_eq!(percentile(&samples, 0.0), Some(1.0));
}

#[test]
fn small_sets_use_the_ceiling_rank() {
    let samples = [30.0, 10.0, 20.0, 40.0];
    assert_eq!(percentile(&samples, 50.0), Some(20.0));
    assert_eq!(percentile(&samples, 51.0), Some(30.0));
    assert_eq!(percentile(&samples, 95.0), Some(40.0));
    assert_eq!(percentile(&[7.5], 99.0), Some(7.5));
}

#[test]
fn empty_and_non_finite_samples_give_none() {
    assert_eq!(percentile(&[], 50.0), None);
    assert_eq!(percentile(&[f64::NAN], 50.0), None);
    assert_eq!(percentile(&[f64::NAN, 3.0], 50.0), Some(3.0));
}

#[test]
fn summary_has_every_key_and_null_for_empty_stages() {
    let latencies = StageLatencies {
        warm_track_ms: vec![1.0, 2.0, 3.0],
        anchor_ms: vec![100.0],
        seg_refresh_ms: Vec::new(),
    };
    let mut summary = Map::new();
    latencies.write_summary(&mut summary);
    assert_eq!(summary["warm_track_p50_ms"], 2.0);
    assert_eq!(summary["warm_track_p95_ms"], 3.0);
    assert_eq!(summary["warm_track_p99_ms"], 3.0);
    assert_eq!(summary["anchor_p50_ms"], 100.0);
    assert_eq!(summary["seg_refresh_p50_ms"], Value::Null);
}
