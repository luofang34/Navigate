//! The rest window by its stamps: a gap clears it, a window of too few
//! increments is not rest, evidence from before the window does not count,
//! and a window without a position waits for one.

use navigate_contract::{NedVelocity, SymmetricCov3};

use super::rig::{GNSS, Rig, at};
use super::{ATTITUDE_DEG, feed, waiting};
use crate::{AlignmentNeed, AlignmentPhase, AlignmentStatus, ErrorBudget, Navigator, SensorData};

/// Half a second without inertial increments clears the window: the state
/// comes two seconds after the gap, not two seconds of increments after
/// the start.
#[test]
fn a_gap_in_the_increments_clears_the_window() {
    let mut rig = Rig::new(ATTITUDE_DEG, 41, 1.0);
    let mut navigator = rig.navigator();
    let mut sink = Vec::new();
    feed(&mut navigator, &mut rig, 1..=100, &mut sink);
    feed(&mut navigator, &mut rig, 151..=349, &mut sink);
    assert!(navigator.solution(at(349)).is_none());
    feed(&mut navigator, &mut rig, 350..=350, &mut sink);
    let Some(AlignmentStatus::Aligned { record, .. }) = navigator.alignment_status() else {
        panic!("aligned: {:?}", navigator.alignment_status());
    };
    assert_eq!(record.at, at(350));
}

/// One increment that covers the whole window has no scatter to judge
/// rest by.
#[test]
fn a_window_of_too_few_increments_is_not_rest() {
    let mut rig = Rig::new(ATTITUDE_DEG, 43, 1.0);
    let mut navigator = rig.navigator();
    let mut sink = Vec::new();
    for step in [100, 150] {
        navigator.ingest(&rig.magnetic(step), &mut sink).unwrap();
    }
    navigator.ingest(&rig.gnss(100, true).0, &mut sink).unwrap();
    let mut long = rig.increment(200);
    if let SensorData::ImuIncrement {
        dt_s,
        delta_theta_rad,
        delta_velocity_m_s,
    } = &mut long.data
    {
        let scale = 2.0 / *dt_s;
        *dt_s = 2.0;
        *delta_theta_rad = delta_theta_rad.map(|v| v * scale);
        *delta_velocity_m_s = delta_velocity_m_s.map(|v| v * scale);
    }
    navigator.ingest(&long, &mut sink).unwrap();
    assert!(navigator.solution(at(200)).is_none());
    assert_eq!(
        waiting(&navigator),
        (AlignmentPhase::Levelling, AlignmentNeed::RestSamples)
    );
}

/// A GNSS fix acquired before the window, and delivered in it, says
/// nothing of rest in the window.
#[test]
fn evidence_from_before_the_window_does_not_count() {
    let mut rig = Rig::new(ATTITUDE_DEG, 47, 1.0);
    let mut navigator = rig.navigator();
    let mut sink = Vec::new();
    let mut late = rig.gnss(50, true).0;
    for step in 101..=300 {
        navigator.ingest(&rig.increment(step), &mut sink).unwrap();
        navigator.ingest(&rig.magnetic(step), &mut sink).unwrap();
        if step == 150 {
            late.delivered_at = at(step);
            navigator.ingest(&late, &mut sink).unwrap();
        }
    }
    assert!(navigator.solution(at(300)).is_none());
    assert_eq!(
        waiting(&navigator),
        (AlignmentPhase::Levelling, AlignmentNeed::RestEvidence)
    );
}

/// A receiver whose suite states no position error and whose fixes carry
/// no position covariance says that the vehicle is at rest, and gives no
/// position: the alignment waits for one.
#[test]
fn a_window_without_a_position_waits_for_one() {
    let mut rig = Rig::new(ATTITUDE_DEG, 53, 1.0);
    let mut navigator = rig.navigator_with_gnss(ErrorBudget::Declared);
    let mut sink = Vec::new();
    let s = rig.budgets.gnss_velocity_m_s;
    for step in 1..=200 {
        navigator.ingest(&rig.increment(step), &mut sink).unwrap();
        navigator.ingest(&rig.magnetic(step), &mut sink).unwrap();
        if step.is_multiple_of(100) {
            let mut fix = rig.gnss(step, true).0;
            if let SensorData::GnssFix {
                position_cov,
                velocity_cov,
                ..
            } = &mut fix.data
            {
                *position_cov = None;
                *velocity_cov = Some(SymmetricCov3::from_diagonal(s * s, s * s, s * s));
            }
            navigator.ingest(&fix, &mut sink).unwrap();
        }
    }
    assert!(navigator.solution(at(200)).is_none());
    assert_eq!(
        waiting(&navigator),
        (AlignmentPhase::Position, AlignmentNeed::PositionSource)
    );
}

/// A receiver that says the vehicle moves makes the window not rest.
#[test]
fn a_moving_receiver_is_not_rest() {
    let mut rig = Rig::new(ATTITUDE_DEG, 59, 1.0);
    let mut navigator = rig.navigator();
    let mut sink = Vec::new();
    for step in 1..=200 {
        navigator.ingest(&rig.increment(step), &mut sink).unwrap();
        navigator.ingest(&rig.magnetic(step), &mut sink).unwrap();
        if step == 100 {
            let mut moving = rig.gnss(step, false).0;
            if let SensorData::GnssFix { velocity, .. } = &mut moving.data {
                *velocity = Some(NedVelocity::new(4.0, 0.0, 0.0));
            }
            assert_eq!(moving.sensor, GNSS);
            navigator.ingest(&moving, &mut sink).unwrap();
        }
    }
    assert_eq!(
        waiting(&navigator),
        (AlignmentPhase::Levelling, AlignmentNeed::Rest)
    );
}
