//! The Earth rotation angle, the polar motion matrix, and the transport
//! velocity of a rotating frame.

use core::f64::consts::TAU;

use navigate_contract::{PhysicalInstant, TimeScale};

use super::*;

fn ut1(seconds: i64, attoseconds: i64) -> PhysicalInstant {
    PhysicalInstant::from_j2000(TimeScale::Ut1, seconds, attoseconds).expect("in range")
}

fn close3(a: [f64; 3], b: [f64; 3], tolerance: f64) -> bool {
    a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= tolerance)
}

#[test]
fn the_earth_rotation_angle_matches_the_iau_reference_values() {
    // At J2000.0 UT1 the angle is 0.7790572732640 turns.
    let at_j2000 = era_rad(ut1(0, 0)).expect("UT1");
    assert!((at_j2000 - 0.779_057_273_264 * TAU).abs() < 1e-15);
    // ERFA test value: era00(2400000.5, 54388.0) = 0.4022837240028158102
    // rad. MJD 54388.0 is 2843.5 days of UT1 after J2000.0.
    let erfa = era_rad(ut1(2_843 * 86_400 + 43_200, 0)).expect("UT1");
    assert!((erfa - 0.402_283_724_002_815_8).abs() < 1e-12, "{erfa}");
    // One UT1 day turns the Earth by one turn and 0.00273781191135448.
    let day = era_rad(ut1(86_400, 0)).expect("UT1");
    let expected = (0.779_057_273_264 + 0.002_737_811_911_354_48) * TAU;
    assert!((day - expected).abs() < 1e-12, "{day}");
    // A half second before J2000.0 the angle is still in [0, 2π).
    let before = era_rad(ut1(-1, 500_000_000_000_000_000)).expect("UT1");
    assert!((0.0..TAU).contains(&before));
    assert!(matches!(
        era_rad(PhysicalInstant::j2000(TimeScale::Tt)),
        Err(TimeScaleError::WrongScale { .. })
    ));
}

#[test]
fn the_polar_motion_matrix_matches_erfa_without_the_tio_locator() {
    // ERFA pom00(2.55060238e-7, 1.860359247e-6, sp) gives the matrix from
    // the TIRS to the ITRS, which is the transpose of W. The TIO locator
    // sp changes only the (0, 1) and (1, 0) elements by about 1e-11.
    let w = polar_motion_matrix(2.550_602_38e-7, 1.860_359_247e-6);
    let rpom = [
        [0.999_999_999_999_967_5, 0.0, 0.255_060_237_999_997_2e-6],
        [0.0, 0.999_999_999_998_269_5, -0.186_035_924_699_886_6e-5],
        [
            -0.255_060_237_974_121_5e-6,
            0.186_035_924_700_241_4e-5,
            0.999_999_999_998_237,
        ],
    ];
    for i in 0..3 {
        for j in 0..3 {
            if i + j == 1 {
                assert!(w[j][i].abs() < 1e-12, "({i}, {j})");
            } else {
                assert!((w[j][i] - rpom[i][j]).abs() < 1e-15, "({i}, {j})");
            }
        }
    }
}

#[test]
fn a_geostationary_vehicle_is_at_rest_in_the_body_fixed_frame() {
    let rate = EARTH_ROTATION_RATE_RAD_PER_S;
    let radius = 42_164_000.0;
    for angle in [0.0, 1.0, 4.0] {
        let inertial_to_fixed = FrameTransform::axial_spin(angle, rate);
        let longitude = 0.7_f64;
        let inertial_angle = angle + longitude;
        let r = [
            radius * inertial_angle.cos(),
            radius * inertial_angle.sin(),
            0.0,
        ];
        // The inertial velocity of a point fixed to the Earth is ω × r.
        let v = [-rate * r[1], rate * r[0], 0.0];
        let (r_fixed, v_fixed) = inertial_to_fixed.transform_state(r, v);
        assert!(close3(v_fixed, [0.0; 3], 1e-9), "{v_fixed:?}");
        let expected = [radius * longitude.cos(), radius * longitude.sin(), 0.0];
        assert!(close3(r_fixed, expected, 1e-6), "{r_fixed:?}");
        // The inverse gives the inertial state back.
        let (r_back, v_back) = inertial_to_fixed
            .inverse()
            .transform_state(r_fixed, v_fixed);
        assert!(close3(r_back, r, 1e-6) && close3(v_back, v, 1e-9));
    }
}

#[test]
fn composed_transforms_add_their_angular_velocities() {
    let first = FrameTransform::axial_spin(0.3, 1e-4);
    let second = FrameTransform {
        dcm_source_to_target: polar_motion_matrix(0.2, -0.1),
        angular_rate_source_rad_s: [2e-5, -1e-5, 3e-5],
    };
    let both = first.then(&second);
    let r = [7_000_000.0, -1_200_000.0, 300_000.0];
    let v = [10.0, 7_500.0, -20.0];
    let (r1, v1) = first.transform_state(r, v);
    let (r2, v2) = second.transform_state(r1, v1);
    let (r12, v12) = both.transform_state(r, v);
    assert!(
        close3(r12, r2, 1e-6) && close3(v12, v2, 1e-9),
        "{v12:?} vs {v2:?}"
    );
    let identity = both.then(&both.inverse());
    let (r_same, v_same) = identity.transform_state(r, v);
    assert!(close3(r_same, r, 1e-6) && close3(v_same, v, 1e-9));
    assert_eq!(
        FrameTransform::IDENTITY.transform_state(r, v),
        (r, v),
        "the identity changes nothing"
    );
}

#[test]
fn the_celestial_to_terrestrial_transform_spins_with_the_earth_rotation_angle() {
    let orientation = EarthOrientation {
        ut1_minus_tai_s: -32.0,
        polar_motion_xp_rad: 0.0,
        polar_motion_yp_rad: 0.0,
    };
    let at = ut1(2_843 * 86_400 + 43_200, 0);
    let transform = FrameTransform::celestial_to_terrestrial(at, &orientation).expect("UT1");
    let era = era_rad(at).expect("UT1");
    // The CIRS x axis is at longitude −ERA in the ITRS.
    let (x_itrs, _) = transform.transform_state([1.0, 0.0, 0.0], [0.0; 3]);
    assert!(
        close3(x_itrs, [era.cos(), -era.sin(), 0.0], 1e-15),
        "{x_itrs:?}"
    );
    assert_eq!(
        transform.angular_rate_source_rad_s,
        [0.0, 0.0, EARTH_ROTATION_RATE_RAD_PER_S]
    );
    let wild = EarthOrientation {
        polar_motion_xp_rad: 1.0,
        ..orientation
    };
    assert_eq!(
        FrameTransform::celestial_to_terrestrial(at, &wild),
        Err(TimeScaleError::ImplausibleEarthOrientation)
    );
}
