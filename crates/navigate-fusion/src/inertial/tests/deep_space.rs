//! Deep-space navigation near a superior conjunction: the solar Shapiro
//! delay of a two-way range from a station on the Earth, and optical
//! navigation bearings to Mars during the approach. The line from the
//! Earth to the spacecraft passes 0.08 AU from the Sun, so the angle
//! between the Sun and the spacecraft as seen from the Earth is about
//! 4.6°.

use nalgebra::{SMatrix, Vector3};
use navigate_contract::{
    CartesianPosition, CartesianVelocity, DurationNanos, SensorClass, SymmetricCov3,
};

use super::*;
use crate::inertial::estimate::InertialCore;
use crate::inertial::models::line_of_sight::tangent_basis;
use crate::inertial::models::radiometric::linearize;
use crate::inertial::nominal::POS;
use crate::inertial::{ForceModelCoast, PropagationMode};
use crate::observation::{LineOfSightReference, Observation, ObservationValue};
use crate::rejection::RejectionReason;
use navigate_astrodynamics::{
    ASTRONOMICAL_UNIT_M, CentralBodyGravity, SpaceEnvironment, SpaceGravityModel,
};

const GM_SUN: f64 = 1.327_124_400_18e20;
const SPEED_OF_LIGHT_MPS: f64 = 299_792_458.0;
const OPNAV_SIGMA_RAD: f64 = 1e-5;
const END_MS: u64 = 600_000;

fn sun_env() -> SpaceEnvironment {
    SpaceEnvironment::inertial(
        FrameId::SUN_J2000,
        SpaceGravityModel::CentralBody(CentralBodyGravity {
            gm_m3_per_s2: GM_SUN,
            j2: 0.0,
            equatorial_radius_m: 6.957e8,
            pole_unit_in_frame: [0.0, 0.0, 1.0],
        }),
    )
}

/// A station on the Earth at one astronomical unit.
fn station() -> Vector3<f64> {
    Vector3::new(ASTRONOMICAL_UNIT_M, 6.4e6, 0.0)
}

/// The spacecraft behind the Sun as seen from the Earth: 0.2 AU from the
/// Sun–Earth line, 1.5 AU from the Sun.
fn spacecraft() -> (Vector3<f64>, Vector3<f64>) {
    (
        Vector3::new(-1.5 * ASTRONOMICAL_UNIT_M, 3.0e10, 0.0),
        Vector3::new(1_500.0, -24_000.0, 800.0),
    )
}

fn mars() -> Vector3<f64> {
    spacecraft().0 + Vector3::new(2.0e8, 1.0e8, 5.0e7)
}

/// The one-way Shapiro delay of the leg from `a` to `b` past the Sun at
/// the origin.
fn expected_delay(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    let (r1, r2, r12) = (a.norm(), b.norm(), (a - b).norm());
    2.0 * GM_SUN / (SPEED_OF_LIGHT_MPS * SPEED_OF_LIGHT_MPS)
        * ((r1 + r2 + r12) / (r1 + r2 - r12)).ln()
}

fn two_way(measured_m: f64, sun: bool) -> ObservationValue {
    ObservationValue::TwoWayRange {
        tx_station_pos_m: station().into(),
        rx_station_pos_m: station().into(),
        gravitation_gm_m3_s2: sun.then_some((GM_SUN, [0.0; 3])),
        round_trip_range_m: measured_m,
        turnaround_delay_m: 0.0,
        variance_m2: 25.0,
    }
}

fn config() -> InertialConfig {
    InertialConfig {
        fusion: crate::config::FusionConfig {
            staleness_bound: DurationNanos::from_millis(60_000),
            ..InertialConfig::default().fusion
        },
        ..InertialConfig::default()
    }
}

fn filter_at(position: Vector3<f64>, sigma_m: f64) -> InertialNavigationFilter {
    let (_, v) = spacecraft();
    let mut filter = InertialNavigationFilter::new_space(config(), CLOCK, sun_env());
    let variance = sigma_m * sigma_m;
    filter
        .initialize_space(
            CartesianPosition::new(position.x, position.y, position.z),
            CartesianVelocity::new(v.x, v.y, v.z),
            identity_attitude(),
            SymmetricCov3::from_diagonal(variance, variance, variance),
            SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            SymmetricCov3::from_diagonal(1e-12, 1e-12, 1e-12),
            t(0),
            gnss(),
        )
        .expect("plausible");
    filter
        .set_propagation_mode(
            PropagationMode::ForceModel(ForceModelCoast::new(60.0)),
            t(0),
        )
        .expect("plausible");
    filter
}

fn dsn(sequence: u32, value: ObservationValue) -> Observation {
    Observation::in_frame(
        stamp(1, sequence, 0),
        value,
        SourceComposition::of(SensorClass::RadioNavigation),
        FrameId::SUN_J2000,
    )
}

#[test]
fn the_solar_shapiro_delay_adds_tens_of_kilometers_near_conjunction() {
    let (r, _) = spacecraft();
    let delay = 2.0 * expected_delay(&station(), &r);
    assert!((10e3..60e3).contains(&delay), "{delay}");
    let geometric = 2.0 * (r - station()).norm();
    let measured = geometric + delay;
    let truth_core = InertialCore {
        nominal: crate::inertial::nominal::InertialNominal::at(
            r,
            spacecraft().1,
            nalgebra::UnitQuaternion::identity(),
        ),
        ..*core(&filter_at(r, 1.0))
    };
    let with_sun = linearize(&truth_core, &two_way(measured, true), &config()).expect("model");
    let without = linearize(&truth_core, &two_way(measured, false), &config()).expect("model");
    assert!(
        with_sun.residual[0].abs() < 1e-3,
        "{}",
        with_sun.residual[0]
    );
    assert!((without.residual[0] / delay - 1.0).abs() < 1e-6);
    // The model with the delay accepts the range. The model without it
    // sees an error of tens of kilometers and the gate refuses it.
    let offset = r + Vector3::new(300.0, -400.0, 200.0);
    let mut relativistic = filter_at(offset, 1_000.0);
    assert_accepted(relativistic.ingest(&dsn(1, two_way(measured, true)), t(0)));
    let mut newtonian = filter_at(offset, 1_000.0);
    let outcome = newtonian.ingest(&dsn(1, two_way(measured, false)), t(0));
    assert!(
        matches!(
            outcome,
            IngestOutcome::Rejected(RejectionReason::InnovationGate { .. })
        ),
        "{outcome:?}"
    );
}

/// The position covariance in the axes of the line of sight to Mars:
/// the 1-sigma along the line and the largest 1-sigma across it.
fn line_of_sight_sigmas(state: &InertialCore) -> (f64, f64) {
    let p = state.p.fixed_view::<3, 3>(POS, POS).into_owned();
    let along = (mars() - state.nominal.p).normalize();
    let across: SMatrix<f64, 3, 2> = tangent_basis(&along);
    let across_cov = across.transpose() * p * across;
    let largest = across_cov.symmetric_eigenvalues().max();
    ((along.transpose() * p * along)[0].sqrt(), largest.sqrt())
}

#[test]
fn optical_bearings_to_mars_constrain_the_position_across_the_line_of_sight() {
    let env = sun_env();
    let model = env.force_model();
    let (mut r, mut v) = spacecraft();
    let mut truth = std::vec![(r, v)];
    for _ in 0..(END_MS / 1_000) {
        (r, v) = model.rk4_step(&r, &v, 1.0);
        truth.push((r, v));
    }
    let initial_error = Vector3::new(20e3, -30e3, 15e3);
    let mut filter = filter_at(spacecraft().0 + initial_error, 100e3);
    let (along_0, across_0) = line_of_sight_sigmas(core(&filter));
    assert!(along_0 > 99e3 && across_0 > 99e3);
    for k in 1..=10_u32 {
        let at_ms = u64::from(k) * 60_000;
        let (r, _) = truth[(at_ms / 1_000) as usize];
        let direction = (mars() - r).normalize();
        let tilt = nalgebra::UnitQuaternion::from_scaled_axis(
            Vector3::new(noise(7 * k), noise(7 * k + 1), noise(7 * k + 2)) * OPNAV_SIGMA_RAD,
        );
        let obs = Observation::in_frame(
            stamp(2, k, at_ms),
            ObservationValue::LineOfSight {
                measured_body_unit: (tilt * direction).into(),
                reference: LineOfSightReference::TargetPosition {
                    position_in_frame_m: mars().into(),
                },
                angular_variance_rad2: OPNAV_SIGMA_RAD * OPNAV_SIGMA_RAD,
            },
            SourceComposition::of(SensorClass::VisualLandmark),
            FrameId::SUN_J2000,
        );
        assert_accepted(filter.ingest(&obs, t(at_ms)));
    }
    let state = *core(&filter);
    // One bearing has no information along the line. The line turns by
    // a few degrees over the approach, so the bearings give a little.
    let (along, across) = line_of_sight_sigmas(&state);
    assert!(across < 3e3, "{across}");
    assert!(along > 10.0 * across, "{along} {across}");
    assert!(along < along_0, "{along}");
    let (r_end, _) = truth[(END_MS / 1_000) as usize];
    let error = state.nominal.p - r_end;
    let axis = (mars() - r_end).normalize();
    let cross_error = (error - axis * axis.dot(&error)).norm();
    assert!(cross_error < 3.0 * across.max(1e3), "{cross_error}");
    assert!(cross_error < 0.1 * (initial_error - axis * axis.dot(&initial_error)).norm());
}
