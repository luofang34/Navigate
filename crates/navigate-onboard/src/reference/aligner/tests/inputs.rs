//! Inputs before the state: matched to the kind of their sensor first, as
//! after it; the stated field of the place; and a known position, which
//! gives a position and no rest.

use super::rig::{GNSS, MAGNETOMETER, Rig, SURVEY, at};
use super::{ATTITUDE_DEG, feed, waiting};
use crate::{AlignmentNeed, AlignmentPhase, Navigator, OnboardError, SensorInput};

/// `input` as if the sensor `sensor` gave it.
fn from(sensor: navigate_contract::SourceId, input: SensorInput) -> SensorInput {
    SensorInput { sensor, ..input }
}

#[test]
fn data_of_another_kind_is_refused_before_the_state() {
    let mut rig = Rig::new(ATTITUDE_DEG, 61, 1.0);
    let mut navigator = rig.navigator();
    let mut sink = Vec::new();
    let refused =
        |result: Result<(), OnboardError>| matches!(result, Err(OnboardError::WrongData { .. }));
    for step in 1..=200 {
        navigator.ingest(&rig.increment(step), &mut sink).unwrap();
        navigator.ingest(&rig.magnetic(step), &mut sink).unwrap();
        if step.is_multiple_of(50) {
            // A fix from the magnetometer and a surveyed point from the
            // receiver would each say that the vehicle is at rest and
            // where it is.
            let fix = from(MAGNETOMETER, rig.gnss(step, true).0);
            assert!(refused(navigator.ingest(&fix, &mut sink)));
            let survey = from(GNSS, Rig::surveyed(step));
            assert!(refused(navigator.ingest(&survey, &mut sink)));
            let field = from(SURVEY, rig.magnetic(step));
            assert!(refused(navigator.ingest(&field, &mut sink)));
        }
    }
    assert!(navigator.solution(at(200)).is_none());
    assert_eq!(
        waiting(&navigator),
        (AlignmentPhase::Levelling, AlignmentNeed::RestEvidence)
    );
    // The same inputs from their own sensors align it.
    feed(&mut navigator, &mut rig, 201..=400, &mut sink);
    assert!(navigator.solution(at(400)).is_some());
}

/// Feeds 2 s of increments and the magnetometer, a GNSS fix with a
/// velocity at 1 s, and what `extra` gives at each step.
fn window_with(
    navigator: &mut crate::ReferenceNavigator,
    rig: &mut Rig,
    mut extra: impl FnMut(u64, &mut Rig) -> Option<SensorInput>,
) {
    let mut sink = Vec::new();
    for step in 1..=200 {
        navigator.ingest(&rig.increment(step), &mut sink).unwrap();
        navigator.ingest(&rig.magnetic(step), &mut sink).unwrap();
        if step == 100 {
            navigator
                .ingest(&rig.gnss(step, true).0, &mut sink)
                .unwrap();
        }
        if let Some(input) = extra(step, rig) {
            navigator.ingest(&input, &mut sink).unwrap();
        }
    }
}

/// The heading needs the field that the host states for the place, and a
/// measured field of another strength is disturbed, not a heading.
#[test]
fn the_heading_needs_the_stated_field_and_its_strength() {
    let mut rig = Rig::new(ATTITUDE_DEG, 67, 1.0);
    let suite = rig.suite(Some(rig.budgets.mag_disturbance_rad));
    let mut untold = crate::ReferenceNavigator::new(suite, super::rig::CLOCK, None).unwrap();
    window_with(&mut untold, &mut rig, |_, _| None);
    assert_eq!(
        waiting(&untold),
        (AlignmentPhase::Heading, AlignmentNeed::HeadingSource)
    );
    // A steel hangar adds 8 µT along the field.
    let mut rig = Rig::new(ATTITUDE_DEG, 67, 1.0);
    rig.hard_iron_t = rig.body_to_ned.inverse() * nalgebra::Vector3::new(4.4e-6, 0.0, 6.7e-6);
    let mut navigator = rig.navigator();
    window_with(&mut navigator, &mut rig, |_, _| None);
    assert_eq!(
        waiting(&navigator),
        (AlignmentPhase::Heading, AlignmentNeed::FieldStrength)
    );
    // A trait object reaches the reference and the configuration.
    let mut boxed: Box<dyn Navigator> = Box::new(rig.navigator());
    let mut unusable = rig.reference();
    unusable.intensity_t = 0.0;
    assert!(boxed.set_magnetic_reference(unusable).is_err());
    assert!(boxed.set_magnetic_reference(rig.reference()).is_ok());
    let mut config = crate::AlignmentConfig::default();
    assert!(boxed.set_alignment_config(config).is_ok());
    config.window_s = f64::NAN;
    assert!(boxed.set_alignment_config(config).is_err());
}

/// A known position gives the position of a window and never its rest:
/// with a receiver that gives no position it completes the window, and
/// without the receiver's velocity the window waits for rest evidence.
#[test]
fn a_known_position_gives_a_position_and_no_rest() {
    let known = |step: u64, _: &mut Rig| {
        (step == 150).then(|| SensorInput {
            sensor: SURVEY,
            acquired_at: at(step),
            delivered_at: at(step),
            data: crate::SensorData::KnownPosition {
                position: navigate_contract::GeodeticPosition::new(
                    super::rig::LAT_DEG.to_radians(),
                    super::rig::LON_DEG.to_radians(),
                    super::rig::HEIGHT_M,
                ),
                horizontal_sigma_m: 3.0,
                vertical_sigma_m: 5.0,
            },
        })
    };
    let mut rig = Rig::new(ATTITUDE_DEG, 73, 1.0);
    let mut navigator = rig.navigator_with_gnss(crate::ErrorBudget::Declared);
    let s = rig.budgets.gnss_velocity_m_s;
    let mut sink = Vec::new();
    for step in 1..=200 {
        navigator.ingest(&rig.increment(step), &mut sink).unwrap();
        navigator.ingest(&rig.magnetic(step), &mut sink).unwrap();
        if step == 100 {
            let mut fix = rig.gnss(step, true).0;
            if let crate::SensorData::GnssFix {
                position_cov,
                velocity_cov,
                ..
            } = &mut fix.data
            {
                *position_cov = None;
                *velocity_cov = Some(navigate_contract::SymmetricCov3::from_diagonal(
                    s * s,
                    s * s,
                    s * s,
                ));
            }
            navigator.ingest(&fix, &mut sink).unwrap();
        }
        if let Some(input) = known(step, &mut rig) {
            navigator.ingest(&input, &mut sink).unwrap();
        }
    }
    let Some(crate::AlignmentStatus::Aligned { record, .. }) = navigator.alignment_status() else {
        panic!("aligned: {:?}", navigator.alignment_status());
    };
    assert_eq!(record.position, Some(crate::PositionSource::KnownPosition));
    // The known position is no evidence of rest.
    let mut rig = Rig::new(ATTITUDE_DEG, 79, 1.0);
    let mut navigator = rig.navigator();
    let mut sink = Vec::new();
    for step in 1..=200 {
        navigator.ingest(&rig.increment(step), &mut sink).unwrap();
        navigator.ingest(&rig.magnetic(step), &mut sink).unwrap();
        if let Some(input) = known(step, &mut rig) {
            navigator.ingest(&input, &mut sink).unwrap();
        }
    }
    assert_eq!(
        waiting(&navigator),
        (AlignmentPhase::Levelling, AlignmentNeed::RestEvidence)
    );
}
