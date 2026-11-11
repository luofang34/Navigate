//! Before the state, an input is matched to the kind of its sensor first,
//! as after it: data of another kind is refused and never reaches the
//! alignment.

use super::rig::{GNSS, MAGNETOMETER, Rig, SURVEY, at};
use super::{ATTITUDE_DEG, feed, waiting};
use crate::reference::aligner::names;
use crate::{AlignmentPhase, Navigator, OnboardError, SensorInput};

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
        (AlignmentPhase::Levelling, names::REST_EVIDENCE)
    );
    // The same inputs from their own sensors align it.
    feed(&mut navigator, &mut rig, 201..=400, &mut sink);
    assert!(navigator.solution(at(400)).is_some());
}
