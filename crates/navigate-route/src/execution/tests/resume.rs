//! Resuming an execution from its stored sequencing state.

use super::*;

/// A four-leg dogleg route along and off the equator.
fn route() -> [Waypoint; 5] {
    [
        wp("W0", 0.0, 0.0),
        wp("W1", 0.0, 0.3),
        wp("W2", 0.3, 0.3),
        wp("W3", 0.3, 0.6),
        wp("W4", 0.0, 0.6),
    ]
}

fn plan(waypoints: &[Waypoint]) -> FlightPlan<'_> {
    FlightPlan::new(
        id("mission"),
        PlanRole::Mission,
        Ellipsoid::WGS84,
        waypoints,
    )
}

/// Positions along the route at 1 km spacing, the samples a vehicle that
/// flies the legs gives the sequencer.
fn samples(waypoints: &[Waypoint]) -> impl Iterator<Item = GeodeticPosition> + '_ {
    let steps = 40_u32;
    waypoints.windows(2).flat_map(move |pair| {
        let (a, b) = (pair[0].position, pair[1].position);
        (0..=steps).map(move |i| {
            let t = f64::from(i) / f64::from(steps);
            GeodeticPosition::new(
                a.latitude_rad + (b.latitude_rad - a.latitude_rad) * t,
                a.longitude_rad + (b.longitude_rad - a.longitude_rad) * t,
                0.0,
            )
        })
    })
}

#[test]
fn a_resumed_execution_continues_step_for_step_like_the_original() {
    let waypoints = route();
    let config = ExecutionConfig::default();
    let mut original = PlanExecution::new(plan(&waypoints), config).expect("valid plan");
    let (mut index, mut complete) = (0, false);
    let mut events = 0;
    for position in samples(&waypoints) {
        let mut resumed =
            PlanExecution::resume(plan(&waypoints), config, index, complete).expect("resume");
        let expected = original.advance(&position, 60.0);
        let got = resumed.advance(&position, 60.0);
        assert_eq!(got, expected, "at {position:?}");
        assert_eq!(resumed, original);
        events += usize::from(expected != SequenceEvent::None);
        (index, complete) = (resumed.active_index(), resumed.is_complete());
    }
    assert_eq!(events, waypoints.len());
    assert!(complete);
}

#[test]
fn an_index_past_the_plan_is_refused() {
    let waypoints = route();
    let refused = PlanExecution::resume(plan(&waypoints), ExecutionConfig::default(), 5, false);
    assert!(matches!(
        refused,
        Err(PlanActivationError::InvalidResumeState {
            active_index: 5,
            waypoints: 5,
            ..
        })
    ));
    let early_complete =
        PlanExecution::resume(plan(&waypoints), ExecutionConfig::default(), 2, true);
    assert!(matches!(
        early_complete,
        Err(PlanActivationError::InvalidResumeState { complete: true, .. })
    ));
}

#[test]
fn a_completed_execution_resumes_completed() {
    let waypoints = route();
    let mut resumed = PlanExecution::resume(plan(&waypoints), ExecutionConfig::default(), 4, true)
        .expect("resume");
    assert!(resumed.is_complete());
    assert_eq!(resumed.active_leg(), None);
    assert_eq!(
        resumed.advance(&waypoints[4].position, 60.0),
        SequenceEvent::None
    );
    assert_eq!(resumed.active_index(), 4);
}

#[test]
fn a_plan_that_does_not_activate_does_not_resume() {
    let waypoints = [wp("W0", 0.0, 0.0), wp("W1", 0.0, 0.0001)];
    let refused = PlanExecution::resume(plan(&waypoints), ExecutionConfig::default(), 1, false);
    assert!(matches!(
        refused,
        Err(PlanActivationError::CaptureRadiusExceedsLeg { .. })
    ));
}
