//! The solution sequence advances only when the published solution
//! changes, for both kinds of solution of one filter.

use super::*;
use crate::observation::{Observation, ObservationValue};

#[test]
fn two_ticks_of_one_state_give_one_solution() {
    let mut filter = surface_filter(
        InertialConfig::default(),
        PlanetaryBody::EARTH_WGS84,
        earth_origin(),
    );
    let rates = rest_rates(&filter);
    assert_accepted(filter.ingest_imu(&imu(1, 100, rates, 0.1), t(100)));
    let first = filter.tick(t(150)).expect("published");
    assert_eq!(filter.tick(t(150)).expect("published"), first);
    let later = filter.tick(t(160)).expect("published");
    assert_eq!(later.stamp.sequence, first.stamp.sequence.next());
}

#[test]
fn a_change_of_kind_advances_the_sequence() {
    let mut filter = surface_filter(
        InertialConfig::default(),
        PlanetaryBody::EARTH_WGS84,
        earth_origin(),
    );
    let rates = rest_rates(&filter);
    assert_accepted(filter.ingest_imu(&imu(1, 100, rates, 0.1), t(100)));
    let surface = filter.tick(t(100)).expect("published");
    let space = filter.tick_space(t(100)).expect("published");
    assert_eq!(filter.tick_space(t(100)).expect("published"), space);
    let surface_again = filter.tick(t(100)).expect("published");
    let sequences = [surface, surface_again].map(|s| s.stamp.sequence.get());
    assert_eq!(sequences, [0, 2]);
    assert_eq!(space.stamp.sequence.get(), 1);
}

#[test]
fn an_adopted_estimate_continues_the_epoch_and_the_sequence_of_the_adopter() {
    let origin = earth_origin();
    let config = InertialConfig::default();
    let mut adopter = surface_filter(config, PlanetaryBody::EARTH_WGS84, origin);
    let mut other = surface_filter(config, PlanetaryBody::EARTH_WGS84, origin);
    let rates = rest_rates(&adopter);
    assert_accepted(adopter.ingest_imu(&imu(1, 100, rates, 0.1), t(100)));
    assert_accepted(other.ingest_imu(&imu(1, 100, rates, 0.1), t(100)));
    // Only the adopter takes a fix 10 m north, and it publishes twice.
    let north = GeodeticPosition::new(
        origin.latitude_rad + 10.0 / 6_371_000.0,
        origin.longitude_rad,
        origin.altitude_m,
    );
    let fix = Observation::new(
        stamp(7, 1, 100),
        ObservationValue::PositionFix {
            position: north,
            covariance: SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
        },
        gnss(),
    );
    assert_accepted(adopter.ingest(&fix, t(100)));
    let moved = adopter.tick(t(100)).expect("published");
    let later = adopter.tick(t(110)).expect("published");
    assert_eq!(later.stamp.sequence, moved.stamp.sequence.next());
    assert!(moved.position.latitude_rad > origin.latitude_rad + 1e-6);
    let rejections = *adopter.rejections();

    adopter.adopt_estimate(&other);
    let adopted = adopter.tick(t(110)).expect("published");
    let source = other.tick(t(110)).expect("published");
    // The state and its covariance are those of the other filter.
    assert_eq!(adopted.position, source.position);
    assert_eq!(adopted.position_cov, source.position_cov);
    assert_eq!(adopted.velocity, source.velocity);
    // The stamps go on from those of the adopter.
    assert_eq!(adopted.stamp.epoch, later.stamp.epoch);
    assert_eq!(adopted.stamp.sequence, later.stamp.sequence.next());
    assert_eq!(*adopter.rejections(), rejections);
    // The adopter then steps as the other filter does.
    assert_accepted(adopter.ingest_imu(&imu(2, 200, rates, 0.1), t(200)));
    assert_accepted(other.ingest_imu(&imu(2, 200, rates, 0.1), t(200)));
    let (a, b) = (adopter.tick(t(200)), other.tick(t(200)));
    assert_eq!(a.map(|s| s.position), b.map(|s| s.position));
}
