#![allow(clippy::expect_used)]
use super::*;

#[test]
fn variable_frame_times_are_preserved_and_duplicates_reject() {
    let camera = CameraModel {
        width: 640,
        height: 480,
        fx: 550.0,
        fy: 550.0,
        cx: 319.5,
        cy: 239.5,
    };
    let mut frames: Vec<_> = ["4.500000", "4.540000", "4.650000"]
        .into_iter()
        .map(|time| ProbeFrame {
            width: 640,
            height: 480,
            best_effort_timestamp_time: Some(time.into()),
        })
        .collect();
    assert_eq!(
        timeline(&frames, camera).expect("valid timeline"),
        vec![0, 40_000_000, 150_000_000]
    );
    frames[2].best_effort_timestamp_time = Some("4.540000".into());
    assert!(timeline(&frames, camera).is_err());
    frames[2].best_effort_timestamp_time = None;
    assert!(timeline(&frames, camera).is_err());
    frames[2].best_effort_timestamp_time = Some("broken-time".into());
    assert!(matches!(
        timeline(&frames, camera),
        Err(BenchError::Timestamp { frame: 2, .. })
    ));
}
