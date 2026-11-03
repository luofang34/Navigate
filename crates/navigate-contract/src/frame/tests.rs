use super::{Ellipsoid, FrameId, naif};

#[test]
fn preset_frames_carry_their_naif_codes() {
    assert_eq!(FrameId::SSB_J2000, FrameId::new(0, 1));
    assert_eq!(FrameId::SUN_J2000, FrameId::new(10, 1));
    assert_eq!(FrameId::EARTH_J2000, FrameId::new(399, 1));
    assert_eq!(FrameId::EARTH_ITRF93, FrameId::new(399, 13_000));
    assert_eq!(FrameId::MOON_J2000, FrameId::new(301, 1));
    assert_eq!(FrameId::MOON_PA, FrameId::new(301, 31_000));
    assert_eq!(FrameId::MOON_ME, FrameId::new(301, 31_001));
    assert_eq!(FrameId::MARS_J2000, FrameId::new(499, 1));
    assert_eq!(FrameId::MARS_IAU, FrameId::new(499, 10_014));
    assert_eq!(naif::SSB, 0);
    assert_eq!(naif::SUN, 10);
    assert_eq!(naif::MOON, 301);
    assert_eq!(naif::EARTH, 399);
    assert_eq!(naif::MARS, 499);
}

#[test]
fn lunar_frames_with_one_center_are_different_frames() {
    assert_eq!(
        FrameId::MOON_ME.center_naif_id,
        FrameId::MOON_PA.center_naif_id
    );
    assert_ne!(FrameId::MOON_ME, FrameId::MOON_PA);
    assert_ne!(FrameId::MOON_ME, FrameId::MOON_J2000);
    assert_ne!(Ellipsoid::MOON_ME, Ellipsoid::MOON_PA);
}

#[test]
fn only_j2000_orientations_are_inertial() {
    for frame in [
        FrameId::SSB_J2000,
        FrameId::SUN_J2000,
        FrameId::EARTH_J2000,
        FrameId::MOON_J2000,
        FrameId::MARS_J2000,
    ] {
        assert!(frame.is_inertial(), "{frame:?}");
    }
    for frame in [
        FrameId::EARTH_ITRF93,
        FrameId::MOON_PA,
        FrameId::MOON_ME,
        FrameId::MARS_IAU,
    ] {
        assert!(!frame.is_inertial(), "{frame:?}");
    }
}

#[test]
fn preset_ellipsoids_carry_their_defining_parameters() {
    assert_eq!(Ellipsoid::WGS84.frame, FrameId::EARTH_ITRF93);
    assert_eq!(Ellipsoid::WGS84.semi_major_axis_m, 6_378_137.0);
    assert_eq!(Ellipsoid::WGS84.flattening, 1.0 / 298.257_223_563);
    assert_eq!(Ellipsoid::GRS80.frame, FrameId::EARTH_ITRF93);
    assert_eq!(Ellipsoid::GRS80.flattening, 1.0 / 298.257_222_101);
    assert_ne!(Ellipsoid::WGS84, Ellipsoid::GRS80);
    assert_eq!(Ellipsoid::MOON_ME.frame, FrameId::MOON_ME);
    assert_eq!(Ellipsoid::MOON_ME.semi_major_axis_m, 1_737_400.0);
    assert_eq!(Ellipsoid::MOON_ME.flattening, 0.0);
    assert_eq!(Ellipsoid::MOON_PA.frame, FrameId::MOON_PA);
    assert_eq!(Ellipsoid::MOON_PA.flattening, 0.0);
    assert_eq!(Ellipsoid::MARS_IAU.frame, FrameId::MARS_IAU);
    assert_eq!(Ellipsoid::MARS_IAU.semi_major_axis_m, 3_396_190.0);
    let mars_b = Ellipsoid::MARS_IAU.semi_major_axis_m * (1.0 - Ellipsoid::MARS_IAU.flattening);
    assert!((mars_b - 3_376_200.0).abs() < 1e-6, "{mars_b}");
}

#[test]
fn preset_ellipsoids_are_plausible() {
    for ellipsoid in [
        Ellipsoid::WGS84,
        Ellipsoid::GRS80,
        Ellipsoid::MOON_ME,
        Ellipsoid::MOON_PA,
        Ellipsoid::MARS_IAU,
    ] {
        assert!(ellipsoid.is_plausible(), "{ellipsoid:?}");
    }
}

#[test]
fn implausible_ellipsoids_are_refused() {
    let frame = FrameId::MOON_ME;
    let refused = [
        Ellipsoid::new(frame, 0.0, 0.0),
        Ellipsoid::new(frame, -1.0, 0.0),
        Ellipsoid::new(frame, f64::NAN, 0.0),
        Ellipsoid::new(frame, f64::INFINITY, 0.0),
        Ellipsoid::new(frame, 1_000.0, -0.01),
        Ellipsoid::new(frame, 1_000.0, 1.0),
        Ellipsoid::new(frame, 1_000.0, f64::NAN),
        Ellipsoid::sphere(FrameId::MOON_J2000, 1_737_400.0),
    ];
    for ellipsoid in refused {
        assert!(!ellipsoid.is_plausible(), "{ellipsoid:?}");
    }
}

#[test]
fn a_custom_sphere_has_zero_flattening() {
    let custom = Ellipsoid::sphere(FrameId::new(401, 401), 11_100.0);
    assert_eq!(custom.flattening, 0.0);
    assert!(custom.is_plausible());
}
