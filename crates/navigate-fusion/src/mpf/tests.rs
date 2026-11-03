use core::f64::consts::TAU;

use navigate_contract::{
    ClockDomainId, GeodeticPosition, MonotonicNanos, NedVelocity, ObservationStamp, SensorClass,
    SolutionQuality, SourceComposition, SourceEpoch, SourceId, SymmetricCov3, WrappingSequence,
};
use navigate_geodesy::{BodyEllipsoid, LocalTangentPlane, NedOffset};

use super::rng::SplitMix64;
use super::*;
use crate::config::FusionConfig;
use crate::navigation_filter::NavigationFilter;
use crate::observation::{Observation, ObservationValue};
use crate::rejection::{IngestOutcome, RejectionReason};

fn origin() -> GeodeticPosition {
    GeodeticPosition::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 500.0)
}

fn plane() -> LocalTangentPlane {
    LocalTangentPlane::new(origin()).expect("plausible origin")
}

/// An analytic anomaly field over the plane of [`origin`].
struct WaveMap {
    plane: LocalTangentPlane,
    amplitude_t: f64,
    north_wavenumber: f64,
    east_wavenumber: f64,
}

impl WaveMap {
    fn at(&self, north_m: f64, east_m: f64) -> f64 {
        self.amplitude_t
            * ((self.north_wavenumber * north_m).cos() + (self.east_wavenumber * east_m).cos())
    }
}

impl ScalarFieldMap for WaveMap {
    fn sample(&self, position: &GeodeticPosition) -> Option<f64> {
        let offset = self.plane.to_ned(position);
        Some(self.at(offset.north_m, offset.east_m))
    }
}

#[test]
fn the_random_source_is_deterministic_and_standard_normal() {
    let mut a = SplitMix64::new(42);
    let mut b = SplitMix64::new(42);
    let draws: Vec<f64> = (0..20_000).map(|_| a.gaussian()).collect();
    assert!(draws.iter().all(|x| *x == b.gaussian()));
    let mean = draws.iter().sum::<f64>() / draws.len() as f64;
    let variance = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / draws.len() as f64;
    assert!(
        mean.abs() < 0.03 && (variance - 1.0).abs() < 0.03,
        "{mean} {variance}"
    );
    let uniform = SplitMix64::new(7).uniform();
    assert!((0.0..1.0).contains(&uniform));
}

#[test]
fn the_grid_map_interpolates_inside_and_refuses_outside() {
    let step = 1e-4;
    let map = BilinearGridMap::new(
        0.8,
        step,
        0.1,
        step,
        2,
        3,
        vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
    )
    .expect("valid grid");
    let at = |lat: f64, lon: f64| map.sample(&GeodeticPosition::new(lat, lon, 0.0));
    let near =
        |value: Option<f64>, expected: f64| value.is_some_and(|v| (v - expected).abs() < 1e-9);
    assert!(near(at(0.8, 0.1), 0.0));
    assert!(near(at(0.8 + step, 0.1 + 2.0 * step), 12.0));
    let middle = at(0.8 + 0.5 * step, 0.1 + 1.5 * step).expect("inside");
    assert!((middle - 6.5).abs() < 1e-9, "{middle}");
    assert_eq!(at(0.8 - step, 0.1), None);
    assert_eq!(at(0.8, 0.1 + 3.0 * step), None);
    assert!(matches!(
        BilinearGridMap::new(0.8, step, 0.1, step, 2, 3, vec![0.0; 5]),
        Err(MpfError::InvalidMap { rows: 2, cols: 3 })
    ));
    assert!(BilinearGridMap::new(0.8, -step, 0.1, step, 2, 2, vec![0.0; 4]).is_err());
}

fn filter(
    config: MpfConfig,
    sigma_m: f64,
    velocity: NedVelocity,
    velocity_sigma: f64,
) -> MarginalizedParticleFilter {
    MarginalizedParticleFilter::new(
        config,
        origin(),
        BodyEllipsoid::WGS84,
        SymmetricCov3::from_diagonal(sigma_m * sigma_m, sigma_m * sigma_m, 1.0),
        velocity,
        SymmetricCov3::from_diagonal(velocity_sigma.powi(2), velocity_sigma.powi(2), 0.01),
    )
    .expect("valid filter")
}

/// The share of the weight with a north position above zero.
fn north_mass(f: &MarginalizedParticleFilter) -> f64 {
    f.x_n
        .iter()
        .zip(f.weights())
        .filter(|(x, _)| x.x > 0.0)
        .map(|(_, w)| w)
        .sum()
}

fn north_of(estimate: &MpfEstimate) -> NedOffset {
    plane().to_ned(&estimate.position)
}

/// A map that is symmetric in north about the prior mean. The prior
/// mean is on a peak with a zero gradient.
fn bimodal_map() -> WaveMap {
    WaveMap {
        plane: plane(),
        amplitude_t: 1e-7,
        north_wavenumber: TAU / 4_000.0,
        east_wavenumber: 0.0,
    }
}

/// Noise variance of the bimodal map measurements in tesla².
const BIMODAL_VARIANCE_T2: f64 = 1e-18;

#[test]
fn a_turn_resolves_a_symmetric_bimodal_map() {
    let map = bimodal_map();
    let variance = BIMODAL_VARIANCE_T2;
    let mut f = filter(
        MpfConfig::default(),
        600.0,
        NedVelocity::new(0.0, 50.0, 0.0),
        0.1,
    );
    let mut truth = (500.0, 0.0);
    let fly = |f: &mut MarginalizedParticleFilter, truth: &mut (f64, f64), v: (f64, f64)| {
        truth.0 += v.0;
        truth.1 += v.1;
        f.predict(1.0).expect("valid step");
        f.ingest_velocity(
            NedVelocity::new(v.0, v.1, 0.0),
            SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
        )
        .expect("valid fix");
        let measured = map.amplitude_t * (map.north_wavenumber * truth.0).cos() + map.amplitude_t;
        f.ingest_scalar_map(&map, measured, variance)
            .expect("on the map");
    };
    for _ in 0..60 {
        fly(&mut f, &mut truth, (0.0, 50.0));
    }
    let mass = north_mass(&f);
    assert!((0.2..0.8).contains(&mass), "both modes keep weight: {mass}");
    let before_turn = f.estimate();
    assert!(!before_turn.unimodal);
    let solution = f.publish(
        MonotonicNanos::from_nanos(0),
        ClockDomainId::new(1),
        SourceComposition::of(SensorClass::MagneticAnomaly),
    );
    assert_eq!(solution.integrity.quality, SolutionQuality::Unusable);
    for _ in 0..60 {
        fly(&mut f, &mut truth, (20.0, 0.0));
    }
    let after_turn = f.estimate();
    let error = north_of(&after_turn).north_m - truth.0;
    assert!(after_turn.unimodal, "{}", after_turn.horizontal_1sigma_m);
    // The two modes are 1000 m apart. The rest of the error trades
    // against the 3 nT map bias prior.
    assert!(error.abs() < 100.0, "{error}");
    assert!(north_mass(&f) > 0.95);
    let solution = f.publish(
        MonotonicNanos::from_nanos(1),
        ClockDomainId::new(1),
        SourceComposition::of(SensorClass::MagneticAnomaly),
    );
    assert_eq!(solution.integrity.quality, SolutionQuality::Degraded);
    assert_eq!(solution.ellipsoid, navigate_contract::Ellipsoid::WGS84);
}

#[test]
fn the_linearized_filter_stalls_on_the_zero_gradient_peak() {
    // The six-state filter linearizes at its mean on the peak. The
    // gradient there is zero, so it refuses the map measurement and
    // keeps the 500 m error.
    let map = bimodal_map();
    let variance = BIMODAL_VARIANCE_T2;
    let mut ekf = NavigationFilter::new(FusionConfig::default(), ClockDomainId::new(1));
    let stamp = |source, sequence, ms: u64| {
        ObservationStamp::new(
            SourceId::new(source),
            SourceEpoch::new(1),
            WrappingSequence::new(sequence),
            MonotonicNanos::from_nanos(ms * 1_000_000),
            ClockDomainId::new(1),
        )
    };
    let fix = Observation::new(
        stamp(1, 1, 0),
        ObservationValue::PositionFix {
            position: origin(),
            covariance: SymmetricCov3::from_diagonal(360_000.0, 360_000.0, 1.0),
        },
        SourceComposition::of(SensorClass::Gnss),
    );
    assert!(
        ekf.ingest(&fix, MonotonicNanos::from_nanos(0))
            .is_accepted()
    );
    let anomaly = Observation::new(
        stamp(2, 1, 0),
        ObservationValue::MagneticAnomaly {
            measured_t: map.amplitude_t * (map.north_wavenumber * 500.0).cos() + map.amplitude_t,
            map_t_at_lin: 2.0 * map.amplitude_t,
            gradient_ned_t_per_m: [0.0, 0.0, 0.0],
            linearization_point: origin(),
            valid_radius_m: 5_000.0,
            variance_t2: variance,
        },
        SourceComposition::of(SensorClass::MagneticAnomaly),
    );
    assert!(matches!(
        ekf.ingest(&anomaly, MonotonicNanos::from_nanos(0)),
        IngestOutcome::Rejected(RejectionReason::UninformativeGradient { .. })
    ));
    let stuck = ekf
        .tick(MonotonicNanos::from_nanos(0))
        .expect("initialized");
    assert!(plane().to_ned(&stuck.position).north_m.abs() < 1e-6);
}

/// Velocity error at the end of a map-only flight with an initial
/// velocity error of 3 m/s north.
fn velocity_error_after_map_flight(coupled: bool) -> f64 {
    let map = WaveMap {
        plane: plane(),
        amplitude_t: 2e-7,
        north_wavenumber: TAU / 2_000.0,
        east_wavenumber: TAU / 1_700.0,
    };
    let config = MpfConfig {
        particle_count: 1_000,
        ..MpfConfig::default()
    };
    let truth_v = (30.0, 20.0);
    let mut f = filter(
        config,
        20.0,
        NedVelocity::new(truth_v.0 + 3.0, truth_v.1, 0.0),
        3.0,
    );
    let mut truth = (0.0, 0.0);
    let mut squared = 0.0;
    for step in 1..=120 {
        truth.0 += truth_v.0;
        truth.1 += truth_v.1;
        f.predict_with_coupling(1.0, coupled).expect("valid step");
        f.ingest_scalar_map(&map, map.at(truth.0, truth.1), 4e-18)
            .expect("on the map");
        if step > 90 {
            let v = f.estimate().velocity;
            squared += (v.north_mps - truth_v.0).powi(2) + (v.east_mps - truth_v.1).powi(2);
        }
    }
    (squared / 30.0).sqrt()
}

#[test]
fn the_cross_coupling_lets_map_updates_correct_the_velocity() {
    let coupled = velocity_error_after_map_flight(true);
    let uncoupled = velocity_error_after_map_flight(false);
    assert!(uncoupled > 2.5, "{uncoupled}");
    assert!(coupled < 0.5 * uncoupled, "{coupled} vs {uncoupled}");
    assert!(coupled < 1.0, "{coupled}");
}

#[test]
fn identical_seeds_give_identical_estimates_and_refusals_change_nothing() {
    let run = || {
        let mut f = filter(
            MpfConfig::default(),
            50.0,
            NedVelocity::new(10.0, 0.0, 0.0),
            1.0,
        );
        f.predict(1.0).expect("valid");
        f.ingest_altitude(505.0, 4.0).expect("valid");
        f
    };
    let (a, b) = (run(), run());
    assert_eq!(a.estimate(), b.estimate());
    let mut c = run();
    let before = c.estimate();
    assert_eq!(c.predict(0.0), Err(MpfError::NonFiniteValue));
    assert_eq!(
        c.ingest_altitude(500.0, 0.0),
        Err(MpfError::ImplausibleCovariance)
    );
    let off_map = BilinearGridMap::new(-1.0, 1e-4, -1.0, 1e-4, 2, 2, vec![0.0; 4]).expect("valid");
    assert_eq!(
        c.ingest_scalar_map(&off_map, 0.0, 1e-18),
        Err(MpfError::NoParticleOnMap {
            particle_count: 2_000
        })
    );
    assert_eq!(c.estimate(), before);
    let invalid = MpfConfig {
        particle_count: 0,
        ..MpfConfig::default()
    };
    assert!(matches!(
        MarginalizedParticleFilter::new(
            invalid,
            origin(),
            BodyEllipsoid::WGS84,
            SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
            NedVelocity::new(0.0, 0.0, 0.0),
            SymmetricCov3::from_diagonal(1.0, 1.0, 1.0)
        ),
        Err(MpfError::InvalidConfig)
    ));
}
