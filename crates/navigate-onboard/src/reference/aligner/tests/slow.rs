//! The start position of a receiver that declares a time-correlated
//! error: the one fix that starts the state holds that error in full.

use super::rig::{Rig, at};
use super::{ATTITUDE_DEG, feed};
use crate::{AlignmentStatus, ErrorBudget, GnssCorrelatedError, Navigator};

/// The position variances, north, east and down, of the state that a
/// vehicle at rest aligns to, with a receiver that declares `slow`,
/// before a fix updates it.
fn start_variances(slow: Option<GnssCorrelatedError>) -> [f64; 3] {
    let mut rig = Rig::new(ATTITUDE_DEG, 71, 1.0);
    let b = rig.budgets;
    let mut navigator = rig.navigator_with_gnss(ErrorBudget::Gnss {
        horizontal_sigma_m: b.gnss_horizontal_m,
        vertical_sigma_m: b.gnss_vertical_m,
        velocity_sigma_m_s: b.gnss_velocity_m_s,
        correlated: slow,
    });
    let mut sink = Vec::new();
    // The window closes at step 200 on the fix of step 100; the fix of
    // step 200 would update the state, so it is not given.
    feed(&mut navigator, &mut rig, 1..=199, &mut sink);
    navigator.ingest(&rig.increment(200), &mut sink).unwrap();
    assert!(matches!(
        navigator.alignment_status(),
        Some(AlignmentStatus::Aligned { .. })
    ));
    let (north, east, down) = navigator.solution(at(200)).unwrap().position_cov.diagonal();
    [north, east, down]
}

/// The start position of a receiver with a slow error states the white
/// and the slow parts together: the slow part of one fix does not
/// average out, and the filter's correlated-error block starts from its
/// own stationary variance.
#[test]
fn a_slow_gnss_error_widens_the_start_position() {
    let slow = GnssCorrelatedError {
        tau_s: 60.0,
        horizontal_sigma_m: 4.0,
        vertical_sigma_m: 6.0,
    };
    let white = start_variances(None);
    let with_slow = start_variances(Some(slow));
    let added = [16.0, 16.0, 36.0];
    for ((w, s), a) in white.iter().zip(with_slow).zip(added) {
        assert!((s - w - a).abs() < 1e-6 * s, "{w} {s} {a}");
    }
}
