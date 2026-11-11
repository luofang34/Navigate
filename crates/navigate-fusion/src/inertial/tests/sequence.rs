//! The solution sequence advances only when the published solution
//! changes, for both kinds of solution of one filter.

use super::*;

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
