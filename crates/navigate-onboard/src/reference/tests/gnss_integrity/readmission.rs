//! Deterministic spoofing exclusion and re-admission tests: step jumps,
//! moderate gate-edge offsets with spoofed Doppler ("first rejected then
//! dragged"), pulsing spoofers while excluded, and post-exclusion
//! `accepting_since` reset.

use super::*;
use crate::AlignmentStatus;

fn gnss_fix_with_doppler(step: u64, north_m: f64, north_mps: f64) -> SensorInput {
    let lat = LAT.to_radians() + north_m / 6_364_000.0;
    SensorInput {
        sensor: GNSS,
        acquired_at: at(step),
        delivered_at: at(step),
        data: SensorData::GnssFix {
            position: GeodeticPosition::new(lat, LON.to_radians(), HEIGHT),
            velocity: Some(NedVelocity::new(north_mps, 0.0, 0.0)),
            position_cov: None,
            velocity_cov: None,
        },
    }
}

fn step_run(
    navigator: &mut ReferenceNavigator,
    seconds: u64,
    fix_for_second: impl Fn(u64) -> (f64, f64),
    witness_every: Option<u64>,
    sink: &mut Vec<AcceptanceRecord>,
) -> Vec<Second> {
    let mut seen = Vec::new();
    for step in 1..=seconds * RATE {
        navigator.ingest(&rest_increment(step), sink).unwrap();
        if !step.is_multiple_of(RATE) {
            continue;
        }
        let second = step / RATE;
        if let Some(period) = witness_every
            && second.is_multiple_of(period)
        {
            navigator.ingest(&witness(second), sink).unwrap();
        }
        let (north_offset_m, north_vel_mps) = fix_for_second(second);
        navigator
            .ingest(
                &gnss_fix_with_doppler(step, north_offset_m, north_vel_mps),
                sink,
            )
            .unwrap();
        seen.push(Second {
            second,
            status: navigator.gnss_integrity().unwrap(),
            solution: navigator.solution(at(step)).unwrap(),
        });
    }
    seen
}

#[test]
fn step_spoof_withholds_doppler_excludes_and_cleanly_readmits_with_healthy_gating() {
    let mut navigator = checked(true);
    let mut sink = Vec::new();
    // Nominal GNSS for seconds 1..=4, +80 m step + +25 m/s spoofed Doppler
    // for seconds 5..=19, spoofing removed from second 20 onward.
    let seen = step_run(
        &mut navigator,
        35,
        |second| {
            if (5..20).contains(&second) {
                (80.0, 25.0)
            } else {
                (0.0, 0.0)
            }
        },
        Some(5),
        &mut sink,
    );

    assert!(seen.iter().take(4).all(|s| {
        s.status.state == GnssIntegrityState::Consistent
            && s.solution
                .integrity
                .contributing
                .contains(SensorClass::Gnss)
    }));

    // Seconds 5 and 6 are rejected at the innovation gate before the third
    // exceedance excludes the receiver on second 7; their spoofed Doppler
    // velocities must be withheld rather than offered to the filter.
    let excluded = first(&seen, GnssIntegrityState::Excluded).unwrap();
    assert_eq!(excluded.second, 7);
    let withheld_vel = sink
        .iter()
        .filter(|r| {
            r.measurement == "gnss_velocity"
                && r.outcome == Outcome::Unused("gnss_position_rejected")
        })
        .count();
    assert_eq!(withheld_vel, 2, "rejected positions must withhold Doppler");

    // The solution is never dragged by the +80 m / +25 m/s step spoofer.
    let worst = seen
        .iter()
        .map(|s| north_m(&s.solution).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1.0, "step spoof dragged solution by {worst} m");

    // Spoofing stops at second 20; after readmission_s (10 s) with witness
    // fixes, GNSS recovers cleanly at second 30.
    let readmitted = seen
        .iter()
        .skip(19)
        .find(|s| s.status.state == GnssIntegrityState::Consistent)
        .expect("GNSS must re-admit after spoofing ends");
    assert_eq!(readmitted.second, 30);
    let final_sec = seen.last().unwrap();
    assert_eq!(final_sec.status.state, GnssIntegrityState::Consistent);
    assert_eq!(final_sec.status.fault, None);
    assert_eq!(final_sec.solution.integrity.quality, SolutionQuality::Good);
    assert!(
        final_sec
            .solution
            .integrity
            .contributing
            .contains(SensorClass::Gnss)
    );
    assert!(final_sec.solution.integrity.horizontal_1sigma_m < 1.5);
    assert!(matches!(
        navigator.alignment_status(),
        Some(AlignmentStatus::Aligned { suspect: false, .. })
    ));

    // Every post-readmission GNSS position and velocity is accepted.
    for second in 30..=35 {
        let stamp = at(second * RATE);
        for measurement in ["gnss_position", "gnss_velocity"] {
            let record = sink
                .iter()
                .find(|r| r.acquired_at == stamp && r.measurement == measurement)
                .expect("record exists");
            assert_eq!(record.outcome, Outcome::Accepted, "{second} {measurement}");
        }
    }
}

#[test]
fn moderate_gate_edge_spoof_is_not_first_rejected_then_dragged_and_readmits() {
    let mut navigator = checked(true);
    let mut sink = Vec::new();
    // +10.5 m north + +8 m/s Doppler sits above main's 3D innovation gate
    // (16.27) and caution_chi2 (13.82), yet below exclusion_chi2 (27.63)
    // once reference_cov + gnss_cov ≈ 4.5 m² (chi2 ≈ 24.5). Without Doppler
    // withholding and sustained gated-disagreement exclusion, main would
    // first reject the position, get dragged by Doppler / covariance
    // growth, and accept the spoofed offset.
    let seen = step_run(
        &mut navigator,
        40,
        |second| {
            if (5..22).contains(&second) {
                (10.5, 8.0)
            } else {
                (0.0, 0.0)
            }
        },
        Some(5),
        &mut sink,
    );

    let doubted = first(&seen, GnssIntegrityState::Disagreeing).expect("must enter Disagreeing");
    let excluded = first(&seen, GnssIntegrityState::Excluded).expect("must enter Excluded");
    assert!(doubted.second < excluded.second);
    assert!(
        excluded.second <= 10,
        "excluded promptly at {}",
        excluded.second
    );

    let worst = seen
        .iter()
        .map(|s| north_m(&s.solution).abs())
        .fold(0.0, f64::max);
    assert!(
        worst < 1.0,
        "moderate spoof must never drag the filter: worst={worst} m"
    );

    // Once spoofing stops at second 22, GNSS cleanly re-admits at second 32.
    let readmitted = seen
        .iter()
        .skip(21)
        .find(|s| s.status.state == GnssIntegrityState::Consistent)
        .expect("clean re-admission");
    assert_eq!(readmitted.second, 32);
    assert!(
        seen.last()
            .unwrap()
            .solution
            .integrity
            .contributing
            .contains(SensorClass::Gnss)
    );
}

#[test]
fn pulsing_spoofer_while_excluded_cannot_readmit_or_inject_a_spoofed_fix() {
    let mut navigator = checked(true);
    let mut sink = Vec::new();
    // Exclude via +60 m step on seconds 5..=10, then pulse +40 m every 3rd
    // second (two honest fixes, one spoofed spike) from second 11..=45.
    let seen = step_run(
        &mut navigator,
        45,
        |second| {
            if (5..=10).contains(&second) {
                (60.0, 0.0)
            } else if second > 10 && second.is_multiple_of(3) {
                (40.0, 0.0)
            } else {
                (0.0, 0.0)
            }
        },
        Some(5),
        &mut sink,
    );

    let excluded_at = first(&seen, GnssIntegrityState::Excluded).unwrap().second;
    assert!(
        seen.iter()
            .filter(|s| s.second >= excluded_at)
            .all(|s| s.status.state == GnssIntegrityState::Excluded),
        "pulsing spoofer must remain Excluded"
    );
}

#[test]
fn accepting_since_resets_across_exclusion_so_first_readmitted_fix_keeps_reference() {
    let mut navigator = checked(true);
    navigator
        .set_gnss_integrity(Some(GnssIntegrityConfig {
            readmission_s: 10.0,
            // Force `useless = bound_m > alert_limit_m` to evaluate whether
            // `settled` triggers prematurely on the first re-admitted fix
            // once `alert_limit_m` is tightened after re-admission.
            ..GnssIntegrityConfig::default()
        }))
        .unwrap();
    let mut sink = Vec::new();
    // Nominal GNSS for 15 s (> readmission_s, so accepting_since was
    // settled before spoofing), spoof on 16..=25, honest from 26..=40.
    let seen = step_run(
        &mut navigator,
        40,
        |second| {
            if (16..=25).contains(&second) {
                (60.0, 0.0)
            } else {
                (0.0, 0.0)
            }
        },
        Some(5),
        &mut sink,
    );

    let readmitted = seen
        .iter()
        .skip(25)
        .find(|s| s.status.state == GnssIntegrityState::Consistent)
        .expect("re-admitted at second 36");
    assert_eq!(readmitted.second, 36);
    // The reference stayed anchored at its original start, not overwritten
    // on the first re-admitted fix.
    assert_eq!(readmitted.status.reference_anchored_at, Some(at(1)));
}
