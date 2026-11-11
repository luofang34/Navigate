//! The error-state layout.

use crate::inertial::{CORE_STATES, StateLayout};

#[test]
fn the_full_layout_has_every_block_after_the_core() {
    let full = StateLayout::FULL;
    assert_eq!(full.dimension(), 22);
    assert!(full.fits(22));
    assert!(!full.fits(21));
    assert_eq!(
        [
            full.position(),
            full.velocity(),
            full.attitude(),
            full.accel_bias(),
            full.gyro_bias()
        ],
        [0, 3, 6, 9, 12]
    );
    assert_eq!(full.alt_bias(), Some(CORE_STATES));
    assert_eq!(full.mag_bias(), Some(16));
    assert_eq!(full.dmc(), Some(17));
    assert_eq!(full.clock(), Some(20));
}

#[test]
fn the_filter_of_today_is_built_with_the_full_layout() {
    let filter = crate::inertial::InertialNavigationFilter::new_surface(
        crate::inertial::InertialConfig::default(),
        super::CLOCK,
        navigate_astrodynamics::PlanetaryBody::EARTH_WGS84,
    );
    assert_eq!(filter.config.layout, StateLayout::FULL);
}
