//! The general change of frame: the Jacobian, the relative rate, and the
//! lossless round trip between the rotating and the inertial frame of
//! the Earth.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use navigate_contract::{AttitudeQuaternion, SymmetricCov3};
use navigate_geodesy::BodyEllipsoid;

use super::*;
use crate::inertial::FrameChange;
use crate::inertial::frame_change::{frame_change_jacobian, transformed_nominal};
use crate::inertial::nominal::{DMC, ErrorMat, N, POS, VEL, to_contract};
use crate::observation::{Observation, ObservationValue};
use crate::rejection::RejectionReason;
use navigate_astrodynamics::{SpaceEnvironment, SpaceGravityModel};

/// The rotation angle of the Earth at the change, in radians.
const EARTH_ANGLE_RAD: f64 = 1.234_5;

/// The inertial frame of the Earth with the pole along `z`.
fn earth_inertial() -> SpaceEnvironment {
    let central = PlanetaryBody::EARTH_WGS84.central_gravity();
    SpaceEnvironment::inertial(
        FrameId::EARTH_J2000,
        SpaceGravityModel::CentralBody(central),
    )
}

/// The rotation about the common pole from the rotating axes to the
/// inertial axes at the rotation angle `angle_rad`.
fn rotating_to_inertial(angle_rad: f64) -> AttitudeQuaternion {
    to_contract(&UnitQuaternion::from_axis_angle(
        &Vector3::z_axis(),
        angle_rad,
    ))
}

fn general_change() -> FrameChange {
    FrameChange {
        new_env: earth_inertial(),
        new_surface_ellipsoid: None,
        origin_in_old_m: [1.0e5, -2.0e5, 3.0e4],
        origin_velocity_in_old_mps: [12.0, -3.0, 40.0],
        orientation_old_to_new: to_contract(&UnitQuaternion::from_euler_angles(0.3, -0.7, 1.9)),
        relative_angular_rate_in_old_rad_s: [2e-3, -1e-3, 7.3e-5],
    }
}

/// Step size per error component for the finite differences.
fn epsilon(index: usize) -> f64 {
    match index {
        0..=2 => 1.0,
        3..=5 => 1e-3,
        6..=8 => 1e-6,
        _ => 1e-6,
    }
}

#[test]
fn the_jacobian_matches_finite_differences_of_the_change() {
    let change = general_change();
    let x = InertialNominal {
        p: Vector3::new(6.4e6, 1.0e5, 2.0e6),
        v: Vector3::new(100.0, -300.0, 7_000.0),
        q: UnitQuaternion::from_euler_angles(-0.4, 0.2, 2.5),
        b_a: Vector3::new(1e-3, -2e-3, 5e-4),
        b_g: Vector3::new(1e-5, 2e-5, -3e-5),
        b_alt: 4.0,
        b_mag: 2e-9,
        a_dmc: Vector3::new(3e-7, -1e-7, 2e-7),
        clock_bias_m: 120.0,
        clock_drift_mps: 0.4,
    };
    let next = transformed_nominal(&x, &change);
    let mut numeric = ErrorMat::zeros();
    for j in 0..N {
        let mut dx = ErrorVec::zeros();
        dx[j] = epsilon(j);
        let plus = transformed_nominal(&x.inject(&dx), &change);
        let minus = transformed_nominal(&x.inject(&(-dx)), &change);
        let column = (error_between(&plus, &next) - error_between(&minus, &next)) / (2.0 * dx[j]);
        numeric.set_column(j, &column);
    }
    let analytic = frame_change_jacobian(&change);
    let error = (analytic - numeric).abs().max();
    assert!(error < 1e-6, "{error}\n{}", analytic - numeric);
    // The rate couples the position error into the velocity error.
    let coupling = analytic.fixed_view::<3, 3>(VEL, POS).abs().max();
    assert!(coupling > 1e-3, "{coupling}");
    // The empirical acceleration turns with the axes.
    let dmc = analytic.fixed_view::<3, 3>(DMC, DMC).into_owned();
    assert!((dmc - Matrix3::identity()).abs().max() > 0.1);
}

#[test]
fn the_relative_rate_comes_from_the_two_environments() {
    let rotating = SpaceEnvironment::from_body(PlanetaryBody::EARTH_WGS84);
    let omega = PlanetaryBody::EARTH_WGS84.rotation_rate_rad_per_s;
    let orientation = rotating_to_inertial(EARTH_ANGLE_RAD);
    let out = FrameChange::between_environments(
        &rotating,
        earth_inertial(),
        None,
        [0.0; 3],
        [0.0; 3],
        orientation,
    );
    assert_eq!(out.relative_angular_rate_in_old_rad_s, [0.0, 0.0, -omega]);
    let back = FrameChange::between_environments(
        &earth_inertial(),
        rotating,
        Some(BodyEllipsoid::WGS84),
        [0.0; 3],
        [0.0; 3],
        rotating_to_inertial(-EARTH_ANGLE_RAD),
    );
    let rate = Vector3::from(back.relative_angular_rate_in_old_rad_s);
    assert!(
        (rate - Vector3::new(0.0, 0.0, omega)).norm() < 1e-20,
        "{rate}"
    );
    assert!(out.is_plausible() && back.is_plausible());
}

/// A surface filter at 28.4°N that has propagated for five seconds with
/// a fix, so the position and the velocity errors are correlated.
fn propagated_surface_filter() -> InertialNavigationFilter {
    let origin = GeodeticPosition::new(28.4_f64.to_radians(), -80.6_f64.to_radians(), 3.0);
    let mut filter = surface_filter(
        InertialConfig::default(),
        PlanetaryBody::EARTH_WGS84,
        origin,
    );
    let rates = rest_rates(&filter);
    for k in 1..=5_u32 {
        let at_ms = u64::from(k) * 1_000;
        assert_accepted(filter.ingest_imu(&imu(k, at_ms, rates, 1.0), t(at_ms)));
    }
    let fix = Observation::new(
        stamp(1, 1, 5_000),
        ObservationValue::PositionFix {
            position: origin,
            covariance: SymmetricCov3::from_diagonal(4.0, 4.0, 9.0),
        },
        gnss(),
    );
    assert_accepted(filter.ingest(&fix, t(5_000)));
    filter
}

#[test]
fn a_round_trip_between_the_rotating_and_the_inertial_frame_is_lossless() {
    let mut filter = propagated_surface_filter();
    let before = *core(&filter);
    let cross = before.p.fixed_view::<3, 3>(VEL, POS).abs().max();
    assert!(cross > 1e-6, "{cross}");
    let rotating = filter.environment();
    let to_inertial = FrameChange::between_environments(
        &rotating,
        earth_inertial(),
        None,
        [0.0; 3],
        [0.0; 3],
        rotating_to_inertial(EARTH_ANGLE_RAD),
    );
    filter.change_frame(to_inertial, t(5_000)).expect("valid");
    let inertial = *core(&filter);
    assert_eq!(filter.frame(), FrameId::EARTH_J2000);
    assert!(filter.tick(t(5_000)).is_none());
    let space = filter.tick_space(t(5_000)).expect("each regime");
    assert_eq!(space.frame, FrameId::EARTH_J2000);
    // The inertial velocity of a point at rest on the Earth is `ω × r`,
    // and its error is `δv + ω × δp`.
    let c = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), EARTH_ANGLE_RAD)
        .to_rotation_matrix()
        .into_inner();
    let w = Vector3::new(0.0, 0.0, PlanetaryBody::EARTH_WGS84.rotation_rate_rad_per_s);
    let expected_v = c * (before.nominal.v + w.cross(&before.nominal.p));
    assert!((inertial.nominal.v - expected_v).norm() < 1e-9);
    assert!(inertial.nominal.v.norm() > 400.0, "{}", inertial.nominal.v);
    let block = |p: &ErrorMat, row: usize, col: usize| p.fixed_view::<3, 3>(row, col).into_owned();
    let wx: Matrix3<f64> = w.cross_matrix();
    let p_vv = block(&before.p, VEL, VEL)
        + wx * block(&before.p, POS, VEL)
        + block(&before.p, VEL, POS) * wx.transpose()
        + wx * block(&before.p, POS, POS) * wx.transpose();
    let expected_vv = c * p_vv * c.transpose();
    let vv_error = (block(&inertial.p, VEL, VEL) - expected_vv).abs().max();
    assert!(vv_error < 1e-12 * expected_vv.abs().max(), "{vv_error}");
    let back = FrameChange::between_environments(
        &earth_inertial(),
        rotating,
        Some(BodyEllipsoid::WGS84),
        [0.0; 3],
        [0.0; 3],
        rotating_to_inertial(-EARTH_ANGLE_RAD),
    );
    filter.change_frame(back, t(5_000)).expect("valid");
    let after = *core(&filter);
    let scale = before.p.abs().max();
    assert!((after.p - before.p).abs().max() < 1e-12 * scale);
    assert!((after.nominal.p - before.nominal.p).norm() < 1e-12 * before.nominal.p.norm());
    assert!((after.nominal.v - before.nominal.v).norm() < 1e-12);
    assert!(after.nominal.q.angle_to(&before.nominal.q) < 1e-12);
    assert!(filter.tick(t(5_000)).is_some());
}

#[test]
fn implausible_changes_are_refused_and_counted() {
    let mut filter = propagated_surface_filter();
    let state = *core(&filter);
    let mut change = general_change();
    change.new_surface_ellipsoid = Some(BodyEllipsoid::MOON_PA);
    assert_eq!(
        filter.change_frame(change, t(5_000)),
        Err(RejectionReason::FrameMismatch {
            expected: FrameId::EARTH_J2000,
            got: FrameId::MOON_PA
        })
    );
    let mut change = general_change();
    change.orientation_old_to_new = AttitudeQuaternion::new(2.0, 0.0, 0.0, 0.0);
    assert_eq!(
        filter.change_frame(change, t(5_000)),
        Err(RejectionReason::NonFiniteValue)
    );
    let mut change = general_change();
    change.relative_angular_rate_in_old_rad_s[1] = f64::NAN;
    assert_eq!(
        filter.change_frame(change, t(5_000)),
        Err(RejectionReason::NonFiniteValue)
    );
    assert!(matches!(
        filter.change_frame(general_change(), t(4_000)),
        Err(RejectionReason::AcquisitionTimeRegression { .. })
    ));
    assert_eq!(core(&filter), &state);
    assert_eq!(filter.rejections().frame_mismatch, 1);
    assert_eq!(filter.rejections().non_finite_value, 2);
    let mut fresh = InertialNavigationFilter::new_surface(
        InertialConfig::default(),
        CLOCK,
        PlanetaryBody::EARTH_WGS84,
    );
    assert_eq!(
        fresh.change_frame(general_change(), t(0)),
        Err(RejectionReason::NotInitialized)
    );
    // A reset keeps the frame of the last change.
    let mut moved = propagated_surface_filter();
    let change = FrameChange::between_environments(
        &moved.environment(),
        earth_inertial(),
        None,
        [0.0; 3],
        [0.0; 3],
        rotating_to_inertial(0.0),
    );
    moved.change_frame(change, t(5_000)).expect("valid");
    let mut initialized = moved.clone();
    moved.reset();
    assert_eq!(moved.frame(), FrameId::EARTH_J2000);
    assert_eq!(moved.surface_ellipsoid(), None);
    // An initialization without a reset also stays in the new frame.
    assert_eq!(
        initialized.initialize_surface(
            earth_origin(),
            NedVelocity::new(0.0, 0.0, 0.0),
            identity_attitude(),
            SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
            SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
            SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
            t(6_000),
            gnss(),
        ),
        Err(RejectionReason::UnsupportedMeasurement {
            kind: crate::observation::MeasurementKind::PositionFix
        })
    );
    initialized
        .initialize_space(
            navigate_contract::CartesianPosition::new(7.0e6, 0.0, 0.0),
            navigate_contract::CartesianVelocity::new(0.0, 7_500.0, 0.0),
            identity_attitude(),
            SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
            SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
            SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
            t(6_000),
            gnss(),
        )
        .expect("plausible");
    assert_eq!(initialized.frame(), FrameId::EARTH_J2000);
    assert_eq!(core(&initialized).regime.env, earth_inertial());
}

#[test]
fn orbital_rtn_uncertainty_is_frame_invariant_between_rotating_and_inertial_frames() {
    use navigate_contract::{CartesianPosition, CartesianVelocity, SpaceUncertainty};

    let rotating = SpaceEnvironment::from_body(PlanetaryBody::EARTH_WGS84);
    let omega = Vector3::new(0.0, 0.0, PlanetaryBody::EARTH_WGS84.rotation_rate_rad_per_s);
    let r_rot = Vector3::new(7_000_000.0, 0.0, 0.0);
    let v_inertial_in_rot = Vector3::new(0.0, 0.0, 1_500.0);
    let v_rot = v_inertial_in_rot - omega.cross(&r_rot);

    let mut filter =
        InertialNavigationFilter::new_space(InertialConfig::default(), CLOCK, rotating);
    filter
        .initialize_space(
            CartesianPosition::new(r_rot.x, r_rot.y, r_rot.z),
            CartesianVelocity::new(v_rot.x, v_rot.y, v_rot.z),
            identity_attitude(),
            SymmetricCov3::from_diagonal(9.0, 4.0, 100.0),
            SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
            SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
            t(0),
            gnss(),
        )
        .expect("plausible");

    let sol_rot = filter.tick_space(t(0)).expect("initialized");
    let to_inertial = FrameChange::between_environments(
        &rotating,
        earth_inertial(),
        None,
        [0.0; 3],
        [0.0; 3],
        rotating_to_inertial(EARTH_ANGLE_RAD),
    );
    filter.change_frame(to_inertial, t(0)).expect("valid");
    let sol_inertial = filter.tick_space(t(0)).expect("initialized");

    match (
        sol_rot.integrity.uncertainty,
        sol_inertial.integrity.uncertainty,
    ) {
        (
            SpaceUncertainty::OrbitalRtn {
                radial_1sigma_m: r1,
                along_track_1sigma_m: t1,
                cross_track_1sigma_m: n1,
                ..
            },
            SpaceUncertainty::OrbitalRtn {
                radial_1sigma_m: r2,
                along_track_1sigma_m: t2,
                cross_track_1sigma_m: n2,
                ..
            },
        ) => {
            assert!((r1 - 3.0).abs() < 1e-12 && (r1 - r2).abs() < 1e-12);
            assert!((t1 - 10.0).abs() < 1e-12 && (t1 - t2).abs() < 1e-12);
            assert!((n1 - 2.0).abs() < 1e-12 && (n1 - n2).abs() < 1e-12);
        }
        other => panic!("expected OrbitalRtn in both frames, got {other:?}"),
    }
}
