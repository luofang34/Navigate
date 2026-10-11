use super::*;
use std::convert::Infallible;

struct LinearTarget {
    epoch: Epoch,
    position: [f64; 3],
    velocity: [f64; 3],
}

impl Ephemeris for LinearTarget {
    type Error = Infallible;

    fn barycentric_state(&self, state: State) -> Result<State, Self::Error> {
        Ok(state)
    }

    fn body_state(&self, _body: i32, epoch: Epoch) -> Result<State, Self::Error> {
        let seconds = (epoch - self.epoch).to_seconds();
        Ok(State {
            position_m: std::array::from_fn(|i| self.position[i] + seconds * self.velocity[i]),
            velocity_m_s: self.velocity,
            epoch,
            frame: FrameId::SSB_J2000,
        })
    }
}

fn observer() -> State {
    State {
        position_m: [0.0; 3],
        velocity_m_s: [0.0; 3],
        epoch: Epoch::from_tdb_seconds(0.0),
        frame: FrameId::SSB_J2000,
    }
}

fn target() -> LinearTarget {
    LinearTarget {
        epoch: observer().epoch,
        position: [LIGHT_SPEED_M_S, 0.0, 0.0],
        velocity: [0.0; 3],
    }
}

#[test]
fn reception_evaluates_a_moving_target_at_emission() {
    let provider = LinearTarget {
        velocity: [30_000.0, 0.0, 0.0],
        ..target()
    };
    let sight = observe_body(&provider, observer(), 499, Correction::VacuumReception).unwrap();
    let expected = LIGHT_SPEED_M_S / (LIGHT_SPEED_M_S + 30_000.0);
    assert!((sight.light_time.unwrap().to_seconds() - expected).abs() < 2e-9);
    assert!((sight.target_epoch - observer().epoch).to_seconds() < -0.99);
    let geometric = observe_body(&provider, observer(), 499, Correction::Geometric).unwrap();
    assert_eq!(geometric.light_time, None);
    assert_eq!(geometric.target_epoch, geometric.received_at);
    assert_eq!(geometric.coordinate_range_m, LIGHT_SPEED_M_S);
}

#[test]
fn orbital_velocity_changes_the_observed_direction() {
    for speed in [3_074.0, 7_800.0] {
        let observer = State {
            velocity_m_s: [0.0, speed, 0.0],
            ..observer()
        };
        let sight = observe_body(&target(), observer, 10, Correction::VacuumReception).unwrap();
        let beta = speed / LIGHT_SPEED_M_S;
        assert!((sight.direction[1] - beta).abs() < 1e-14);
        assert!((sight.direction[0] - (1.0 - beta * beta).sqrt()).abs() < 1e-14);
        assert_eq!(sight.received_at, observer.epoch);
    }
}

#[test]
fn lorentz_aberration_has_an_inverse_for_arbitrary_directions() {
    let direction = Vector3::new(1.0, 2.0, -3.0).normalize();
    let beta = Vector3::new(0.1, -0.2, 0.05);
    assert!((aberrate(aberrate(direction, beta), -beta) - direction).norm() < 1e-14);
}

#[test]
fn location_has_no_earth_radius_or_altitude_limit() {
    for position in [[42_164_000.0, 0.0, 0.0], [2e11, -1e11, 5e10]] {
        let observer = State {
            position_m: position,
            ..observer()
        };
        let provider = LinearTarget {
            position: [position[0], position[1] + 1e9, position[2]],
            ..target()
        };
        let sight = observe_body(&provider, observer, 301, Correction::VacuumReception).unwrap();
        assert_eq!(sight.direction, [0.0, 1.0, 0.0]);
    }
}

#[test]
fn rejects_unknown_frame_from_provider_and_nonfinite_state() {
    let invalid_frame = State {
        frame: FrameId {
            center: 499,
            orientation: 1,
        },
        ..observer()
    };
    assert!(matches!(
        observe_body(&target(), invalid_frame, 10, Correction::Geometric),
        Err(ObservationError::ProviderContract { .. })
    ));
    let invalid = State {
        position_m: [f64::NAN, 0.0, 0.0],
        ..observer()
    };
    assert!(matches!(
        observe_body(&target(), invalid, 10, Correction::Geometric),
        Err(ObservationError::InvalidState { .. })
    ));
}

#[test]
fn rejects_light_speed_and_coincident_observer() {
    let invalid = State {
        velocity_m_s: [0.0, LIGHT_SPEED_M_S, 0.0],
        ..observer()
    };
    assert!(matches!(
        observe_body(&target(), invalid, 10, Correction::VacuumReception),
        Err(ObservationError::InvalidState { .. })
    ));
    let provider = LinearTarget {
        position: [0.0; 3],
        ..target()
    };
    assert!(matches!(
        observe_body(&provider, observer(), 10, Correction::Geometric),
        Err(ObservationError::Range { .. })
    ));
}

#[test]
fn utc_reception_uses_continuous_tdb_across_a_leap_second() {
    let reception: Epoch = "2017-01-01T00:00:00 UTC".parse().unwrap();
    let observer = State {
        epoch: reception,
        ..observer()
    };
    let provider = LinearTarget {
        epoch: reception.to_time_scale(TimeScale::TDB),
        ..target()
    };
    let sight = observe_body(&provider, observer, 10, Correction::VacuumReception).unwrap();
    assert_eq!(sight.received_at.time_scale, TimeScale::TDB);
    assert!(((sight.received_at - sight.target_epoch).to_seconds() - 1.0).abs() < 2e-9);
    let tai_observer = State {
        epoch: reception.to_time_scale(TimeScale::TAI),
        ..observer
    };
    assert_eq!(
        sight,
        observe_body(&provider, tai_observer, 10, Correction::VacuumReception).unwrap()
    );
}

#[test]
fn missing_kernel_coverage_is_an_error() {
    let almanac = anise::almanac::Almanac::default();
    let provider = crate::AniseEphemeris::new(&almanac, &[]);
    assert!(matches!(
        observe_body(&provider, observer(), 499, Correction::VacuumReception),
        Err(ObservationError::Ephemeris { target: 499, .. })
    ));
}

#[test]
fn a_failed_light_time_iteration_does_not_return_a_direction() {
    let provider = LinearTarget {
        velocity: [0.9 * LIGHT_SPEED_M_S, 0.0, 0.0],
        ..target()
    };
    assert!(matches!(
        observe_body(&provider, observer(), 10, Correction::VacuumReception),
        Err(ObservationError::LightTime { .. })
    ));
}
