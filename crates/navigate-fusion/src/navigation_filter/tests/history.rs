//! Delayed-measurement history and out-of-sequence replay.

use navigate_contract::DurationNanos;

use super::*;
use crate::navigation_filter::Core;
use crate::observation::MeasurementKind;

const GNSS: u32 = 1;
const VISUAL: u32 = 2;
const VELOCITY: u32 = 3;

fn fix(
    source: u32,
    sequence: u32,
    at_ms: u64,
    north_m: f64,
    east_m: f64,
    variance_m2: f64,
) -> Observation {
    let plane = LocalTangentPlane::new(origin()).expect("plausible origin");
    let class = if source == VISUAL {
        SensorClass::VisualLandmark
    } else {
        SensorClass::Gnss
    };
    Observation::new(
        stamp(source, 1, sequence, at_ms),
        ObservationValue::PositionFix {
            position: plane.from_ned(&NedOffset::new(north_m, east_m, 0.0)),
            covariance: SymmetricCov3::from_diagonal(variance_m2, variance_m2, variance_m2),
        },
        SourceComposition::of(class),
    )
}

fn velocity(sequence: u32, at_ms: u64, north_mps: f64) -> Observation {
    Observation::new(
        stamp(VELOCITY, 1, sequence, at_ms),
        ObservationValue::VelocityFix {
            velocity: NedVelocity::new(north_mps, 0.0, 0.0),
            covariance: SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
        },
        SourceComposition::of(SensorClass::Inertial),
    )
}

fn run(config: FusionConfig, script: &[(Observation, u64)]) -> NavigationFilter {
    let mut f = NavigationFilter::new(config, CLOCK);
    for (obs, now_ms) in script {
        assert_accepted(f.ingest(obs, t(*now_ms)));
    }
    f
}

fn core(f: &NavigationFilter) -> Core {
    f.core.expect("initialized")
}

fn assert_cores_match(a: &Core, b: &Core, tolerance: f64) {
    assert_eq!(a.plane, b.plane);
    assert_eq!(a.state_at, b.state_at);
    let dx = (a.state.x - b.state.x).abs().max();
    let dp = (a.state.p - b.state.p).abs().max();
    assert!(dx <= tolerance, "mean differs by {dx}");
    assert!(dp <= tolerance, "covariance differs by {dp}");
}

/// A wide velocity prior (100 m/s 1-sigma) admits the first 70 m/s
/// velocity fix.
fn flying_config() -> FusionConfig {
    FusionConfig {
        initial_velocity_variance_m2_per_s2: 1e4,
        ..FusionConfig::default()
    }
}

/// A vehicle at 70 m/s north: a 20 m 1-sigma first fix and tight
/// velocity fixes every 100 ms until `end_ms`.
fn flying_script(end_ms: u64) -> Vec<(Observation, u64)> {
    let mut script = vec![(fix(GNSS, 1, 0, 0.0, 0.0, 400.0), 0)];
    for step in 0..=end_ms / 100 {
        let at_ms = step * 100;
        let sequence = u32::try_from(step).expect("short script") + 1;
        script.push((velocity(sequence, at_ms, 70.0), at_ms));
    }
    script
}

#[test]
fn an_out_of_order_script_matches_the_in_order_script() {
    let a = (fix(GNSS, 1, 0, 0.0, 0.0, 25.0), 0);
    let b = (velocity(1, 100, 10.0), 100);
    let c = fix(VISUAL, 1, 200, 2.5, 3.0, 16.0);
    let d = (velocity(2, 250, 10.0), 250);
    let e = (fix(GNSS, 2, 300, 3.2, -1.0, 25.0), 300);
    let config = FusionConfig::default();
    let in_order = run(config, &[a, b, (c, 200), d, e]);
    let delayed = run(config, &[a, b, d, e, (c, 350)]);
    assert_cores_match(&core(&in_order), &core(&delayed), 1e-11);
    assert_eq!(in_order.history.len(), delayed.history.len());
    let count = |f: &NavigationFilter, kind| f.innovation_stats(kind).map(|s| s.admitted_count);
    for kind in [MeasurementKind::PositionFix, MeasurementKind::VelocityFix] {
        assert_eq!(count(&in_order, kind), Some(2));
        assert_eq!(count(&delayed, kind), Some(2));
    }
}

#[test]
fn a_delayed_fix_applies_at_its_acquisition_time_without_lag_bias() {
    let config = flying_config();
    // The fix reads the true position at 600 ms: 42 m north. It arrives
    // 400 ms late, when the vehicle is at 70 m.
    let late = fix(VISUAL, 1, 600, 42.0, 0.0, 25.0);
    let mut delayed = flying_script(1_000);
    delayed.push((late, 1_000));
    let delayed = run(config, &delayed);
    let north_m = core(&delayed).state.x[0];
    assert!((north_m - 70.0).abs() < 1.0, "{north_m}");

    let mut ordered = flying_script(1_000);
    let position = ordered
        .iter()
        .position(|(obs, _)| obs.stamp.acquired_at > t(600))
        .expect("later entries");
    ordered.insert(position, (late, 600));
    assert_cores_match(&core(&run(config, &ordered)), &core(&delayed), 1e-9);

    // Applied at the arrival time, the same fix pulls the state back by
    // most of v·dt = 70 m/s · 0.4 s = 28 m.
    let lagged = fix(VISUAL, 1, 1_000, 42.0, 0.0, 25.0);
    let mut naive = flying_script(1_000);
    naive.push((lagged, 1_000));
    let naive_north_m = core(&run(config, &naive)).state.x[0];
    assert!(70.0 - naive_north_m > 20.0, "{naive_north_m}");
}

#[test]
fn a_delayed_outlier_leaves_state_and_history_untouched() {
    let mut f = run(flying_config(), &flying_script(1_000));
    let before = core(&f);
    let history_before = format!("{:?}", f.history);
    let outlier = fix(VISUAL, 1, 600, 42.0, 1_000.0, 25.0);
    assert!(matches!(
        rejection(f.ingest(&outlier, t(1_000))),
        RejectionReason::InnovationGate { .. }
    ));
    assert_cores_match(&core(&f), &before, 0.0);
    assert_eq!(format!("{:?}", f.history), history_before);
    assert_eq!(f.rejections().innovation_gate, 1);
    assert_eq!(
        f.innovation_stats(MeasurementKind::PositionFix)
            .map(|s| (s.admitted_count, s.consecutive_gate_rejections)),
        Some((0, 1))
    );
    // The source's sequence slot stays open for a valid delayed fix.
    assert_accepted(f.ingest(&fix(VISUAL, 1, 600, 42.0, 0.0, 25.0), t(1_000)));
    assert_eq!(
        f.innovation_stats(MeasurementKind::PositionFix)
            .map(|s| (s.admitted_count, s.consecutive_gate_rejections)),
        Some((1, 0))
    );
}

#[test]
fn evicted_history_refuses_with_outside_history() {
    let config = FusionConfig {
        history_capacity: 2,
        ..flying_config()
    };
    let mut f = run(config, &flying_script(400));
    // Entries kept: updates at 300 and 400. The oldest rewind point is
    // the state at 200 ms.
    assert_eq!(f.history.len(), 2);
    let before = core(&f);
    let reason = rejection(f.ingest(&fix(VISUAL, 1, 150, 10.5, 0.0, 25.0), t(450)));
    assert_eq!(
        reason,
        RejectionReason::OutsideHistory {
            age: DurationNanos::from_millis(250),
            depth: DurationNanos::from_millis(200),
        }
    );
    assert_eq!(f.rejections().outside_history, 1);
    assert_cores_match(&core(&f), &before, 0.0);
    // Inside the retained depth, the same slot is admitted.
    assert_accepted(f.ingest(&fix(VISUAL, 1, 250, 17.5, 0.0, 25.0), t(450)));
    assert_eq!(f.history.len(), 2);
}

#[test]
fn delayed_fixes_before_initialization_or_without_history_are_refused() {
    let mut f = filter();
    assert_accepted(f.ingest(&fix(GNSS, 1, 1_000, 0.0, 0.0, 25.0), t(1_000)));
    assert_accepted(f.ingest(&velocity(1, 1_200, 0.0), t(1_200)));
    let reason = rejection(f.ingest(&fix(VISUAL, 1, 900, 0.0, 0.0, 25.0), t(1_200)));
    assert!(matches!(
        reason,
        RejectionReason::OutsideHistory { age, .. } if age == DurationNanos::from_millis(300)
    ));

    let config = FusionConfig {
        history_capacity: 0,
        ..flying_config()
    };
    let mut f = run(config, &flying_script(100));
    assert_eq!(f.history.len(), 0);
    assert_eq!(
        rejection(f.ingest(&fix(VISUAL, 1, 50, 3.5, 0.0, 25.0), t(100))),
        RejectionReason::OutsideHistory {
            age: DurationNanos::from_millis(50),
            depth: DurationNanos::from_nanos(0),
        }
    );
    assert_eq!(f.rejections().outside_history, 1);
}

#[test]
fn replay_across_a_reanchor_matches_the_in_order_script() {
    // A 200 m re-anchor distance makes the replay cross re-anchors.
    let config = FusionConfig {
        reanchor_distance_m: 200.0,
        ..flying_config()
    };
    let late = fix(VISUAL, 1, 2_800, 196.0, 0.0, 25.0);
    let mut delayed = flying_script(3_200);
    delayed.push((late, 3_200));
    let delayed = run(config, &delayed);
    assert_ne!(core(&delayed).plane.origin(), origin(), "re-anchored");

    let mut ordered = flying_script(3_200);
    let position = ordered
        .iter()
        .position(|(obs, _)| obs.stamp.acquired_at > t(2_800))
        .expect("later entries");
    ordered.insert(position, (late, 2_800));
    assert_cores_match(&core(&run(config, &ordered)), &core(&delayed), 1e-9);
}
