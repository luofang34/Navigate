//! Cislunar propagation, the change from `EARTH_J2000` to `MOON_J2000`,
//! and the replay of a delayed fix through the change.

use nalgebra::Vector3;
use navigate_contract::{
    CartesianPosition, CartesianVelocity, DurationNanos, FrameId, SymmetricCov3,
};

use super::*;
use crate::inertial::estimate::InertialCore;
use crate::observation::{Observation, ObservationValue};
use crate::rejection::RejectionReason;
use navigate_astrodynamics::{
    CentralBodyGravity, PointMassBody, SpaceEnvironment, SpaceGravityModel, ThirdBodySet,
};

const MOON_POSITION_M: [f64; 3] = [384_400_000.0, 0.0, 0.0];
const MOON_VELOCITY_MPS: [f64; 3] = [0.0, 1_022.0, 0.0];
const RECENTER_MS: u64 = 10_000;

fn earth_env() -> SpaceEnvironment {
    SpaceEnvironment::inertial(
        FrameId::EARTH_J2000,
        SpaceGravityModel::MultiBody {
            central: CentralBodyGravity {
                gm_m3_per_s2: 3.986_004_418e14,
                j2: 1.082_626_68e-3,
                equatorial_radius_m: 6_378_137.0,
                pole_unit_in_frame: [0.0, 0.0, 1.0],
            },
            third_bodies: ThirdBodySet::from_slice(&[PointMassBody {
                naif_id: 301,
                gm_m3_per_s2: 4.904_869_5e12,
                position_in_frame_m: MOON_POSITION_M,
            }])
            .expect("one body"),
        },
    )
}

fn moon_env() -> SpaceEnvironment {
    SpaceEnvironment::inertial(
        FrameId::MOON_J2000,
        SpaceGravityModel::MultiBody {
            central: CentralBodyGravity {
                gm_m3_per_s2: 4.904_869_5e12,
                j2: 2.032_3e-4,
                equatorial_radius_m: 1_737_400.0,
                pole_unit_in_frame: [0.0, 0.0, 1.0],
            },
            third_bodies: ThirdBodySet::from_slice(&[PointMassBody {
                naif_id: 399,
                gm_m3_per_s2: 3.986_004_418e14,
                position_in_frame_m: MOON_POSITION_M.map(|c| -c),
            }])
            .expect("one body"),
        },
    )
}

fn config() -> InertialConfig {
    InertialConfig {
        fusion: crate::config::FusionConfig {
            staleness_bound: DurationNanos::from_millis(30_000),
            ..InertialConfig::default().fusion
        },
        ..InertialConfig::default()
    }
}

fn space_filter() -> InertialNavigationFilter {
    let mut filter = InertialNavigationFilter::new_space(config(), CLOCK, earth_env());
    filter
        .initialize_space(
            CartesianPosition::new(326_000_000.0, 8_000_000.0, 1_000_000.0),
            CartesianVelocity::new(800.0, 300.0, 0.0),
            identity_attitude(),
            SymmetricCov3::from_diagonal(1e4, 1e4, 1e4),
            SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
            SymmetricCov3::from_diagonal(1e-4, 1e-4, 1e-4),
            t(0),
            gnss(),
        )
        .expect("plausible");
    filter
}

fn coast_imu(k: u32) -> ImuSample {
    let at_ms = u64::from(k) * 1_000;
    let frame = if at_ms <= RECENTER_MS {
        FrameId::EARTH_J2000
    } else {
        FrameId::MOON_J2000
    };
    imu_in(k, at_ms, ([0.0; 3], [0.0, 0.0, 1e-3]), 1.0, frame)
}

fn earth_fix(sequence: u32, at_ms: u64) -> Observation {
    let at_s = at_ms as f64 * 1e-3;
    Observation::in_frame(
        stamp(1, sequence, at_ms),
        ObservationValue::CartesianPositionFix {
            position: CartesianPosition::new(
                326_000_000.0 + 800.0 * at_s + 40.0,
                8_000_000.0 + 300.0 * at_s - 30.0,
                1_000_000.0 + 20.0,
            ),
            covariance: SymmetricCov3::from_diagonal(100.0, 100.0, 100.0),
        },
        gnss(),
        FrameId::EARTH_J2000,
    )
}

fn recenter(filter: &mut InertialNavigationFilter) {
    filter
        .recenter_space_frame(
            moon_env(),
            MOON_POSITION_M,
            MOON_VELOCITY_MPS,
            None,
            t(RECENTER_MS),
        )
        .expect("plausible recentering");
}

/// IMU samples to 20 s with the recentering at 10 s. `fix_now_ms` is the
/// arrival time of the fix acquired at 5 s.
fn run(fix_now_ms: u64) -> InertialNavigationFilter {
    let mut filter = space_filter();
    for k in 1..=20_u32 {
        let at_ms = u64::from(k) * 1_000;
        assert_accepted(filter.ingest_imu(&coast_imu(k), t(at_ms)));
        if at_ms == RECENTER_MS {
            recenter(&mut filter);
        }
        if at_ms == fix_now_ms {
            assert_accepted(filter.ingest(&earth_fix(1, 5_000), t(at_ms)));
        }
    }
    filter
}

fn run_core(fix_now_ms: u64) -> InertialCore {
    *core(&run(fix_now_ms))
}

#[test]
fn a_delayed_fix_replays_through_the_change_of_frame() {
    let in_order = run_core(5_000);
    let no_fix = run_core(u64::MAX);
    let mut delayed_filter = run(12_000);
    let delayed = *core(&delayed_filter);
    assert_eq!(delayed.regime.frame(), FrameId::MOON_J2000);
    let scale = in_order.nominal.p.norm();
    assert!((in_order.nominal.p - delayed.nominal.p).norm() < 1e-9 * scale);
    assert!((in_order.nominal.v - delayed.nominal.v).norm() < 1e-9);
    assert!(in_order.nominal.q.angle_to(&delayed.nominal.q) < 1e-12);
    assert!((in_order.p - delayed.p).abs().max() < 1e-9 * in_order.p.abs().max());
    // The fix moved the state: without it the final state differs.
    assert!((no_fix.nominal.p - in_order.nominal.p).norm() > 1.0);
    // A fix in the frame of the Earth after the change is refused.
    let late = earth_fix(2, 20_000);
    assert_eq!(
        delayed_filter.ingest(&late, t(20_000)),
        IngestOutcome::Rejected(RejectionReason::FrameMismatch {
            expected: FrameId::MOON_J2000,
            got: FrameId::EARTH_J2000,
        })
    );
    let solution = delayed_filter.tick_space(t(20_000)).expect("initialized");
    assert_eq!(solution.frame, FrameId::MOON_J2000);
    assert!(delayed_filter.tick(t(20_000)).is_none());
}

#[test]
fn the_change_of_frame_shifts_the_state_and_keeps_the_attitude_errors() {
    let mut filter = space_filter();
    for k in 1..=10_u32 {
        assert_accepted(filter.ingest_imu(&coast_imu(k), t(u64::from(k) * 1_000)));
    }
    let before = *core(&filter);
    recenter(&mut filter);
    let after = core(&filter);
    let shift = Vector3::from(MOON_POSITION_M);
    assert!((after.nominal.p - (before.nominal.p - shift)).norm() < 1e-6);
    assert!(
        (after.nominal.v - (before.nominal.v - Vector3::from(MOON_VELOCITY_MPS))).norm() < 1e-9
    );
    assert_eq!(after.p, before.p);
    assert!(
        after.nominal.p.norm() < 66_000_000.0,
        "inside the lunar sphere of influence"
    );
    assert_eq!(
        filter.recenter_space_frame(moon_env(), [0.0; 3], [0.0; 3], None, t(9_000)),
        Err(RejectionReason::AcquisitionTimeRegression {
            last: t(RECENTER_MS),
            got: t(9_000)
        })
    );
}

#[test]
fn an_environment_update_keeps_the_frame() {
    let mut filter = space_filter();
    assert_eq!(
        filter.update_space_environment(moon_env()),
        Err(RejectionReason::FrameMismatch {
            expected: FrameId::EARTH_J2000,
            got: FrameId::MOON_J2000
        })
    );
    let mut moved = earth_env();
    if let SpaceGravityModel::MultiBody { third_bodies, .. } = &mut moved.gravity {
        *third_bodies = ThirdBodySet::from_slice(&[PointMassBody {
            naif_id: 301,
            gm_m3_per_s2: 4.904_869_5e12,
            position_in_frame_m: [380_000_000.0, 40_000_000.0, 0.0],
        }])
        .expect("one body");
    }
    assert!(filter.update_space_environment(moved).is_ok());
    assert_eq!(core(&filter).regime.env, moved);
    assert_eq!(filter.environment(), moved);
    assert_eq!(filter.rejections().frame_mismatch, 1);
    // A surface filter keeps its ellipsoid through an environment update
    // in its own frame.
    let mut surface = surface_filter(
        InertialConfig::default(),
        PlanetaryBody::EARTH_WGS84,
        earth_origin(),
    );
    assert_eq!(
        surface.update_space_environment(moon_env()),
        Err(RejectionReason::FrameMismatch {
            expected: FrameId::EARTH_ITRF93,
            got: FrameId::MOON_J2000
        })
    );
    let rotating = SpaceEnvironment::from_body(PlanetaryBody::EARTH_WGS84);
    assert!(surface.update_space_environment(rotating).is_ok());
    assert_eq!(
        surface.surface_ellipsoid(),
        Some(navigate_geodesy::BodyEllipsoid::WGS84)
    );
    assert!(surface.tick(t(0)).is_some());
}
