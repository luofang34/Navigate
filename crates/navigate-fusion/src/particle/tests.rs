use core::f64::consts::TAU;
use std::vec;
use std::vec::Vec;

use navigate_contract::{
    ClockDomainId, GeodeticPosition, MonotonicNanos, NedVelocity, ObservationStamp, SensorClass,
    SolutionQuality, SourceComposition, SourceEpoch, SourceId, SymmetricCov3, WrappingSequence,
};
use navigate_geodesy::{BodyEllipsoid, LocalTangentPlane, NedOffset};

use super::rng::SplitMix64;
use super::*;
use crate::config::FusionConfig;
use crate::kinematic::NavigationFilter;
use crate::observation::{Observation, ObservationValue};
use crate::rejection::{IngestOutcome, RejectionReason};

fn origin() -> GeodeticPosition {
    GeodeticPosition::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 500.0)
}

fn plane() -> LocalTangentPlane {
    LocalTangentPlane::new(origin(), navigate_geodesy::BodyEllipsoid::WGS84)
        .expect("plausible origin")
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

/// Number of particles in the tests.
const PARTICLES: usize = 2_000;

/// Caller-owned storage of one filter: the active particles and the
/// scratch area.
struct Storage {
    active: Vec<MpfParticle>,
    scratch: Vec<MpfParticle>,
}

impl Storage {
    fn new(count: usize) -> Self {
        Self {
            active: vec![MpfParticle::ZERO; count],
            scratch: vec![MpfParticle::ZERO; count],
        }
    }

    fn pool(&mut self) -> ParticlePool<'_> {
        ParticlePool::new(&mut self.active, &mut self.scratch).expect("equal slices")
    }
}

/// A map with no coverage anywhere.
struct NoCoverage;

impl ScalarFieldMap for NoCoverage {
    fn sample(&self, _position: &GeodeticPosition) -> Option<f64> {
        None
    }
}

fn filter(
    storage: &mut Storage,
    config: MpfConfig,
    sigma_m: f64,
    velocity: NedVelocity,
    velocity_sigma: f64,
) -> MarginalizedParticleFilter<'_> {
    MarginalizedParticleFilter::new(
        config,
        storage.pool(),
        origin(),
        BodyEllipsoid::WGS84,
        SymmetricCov3::from_diagonal(sigma_m * sigma_m, sigma_m * sigma_m, 1.0),
        velocity,
        SymmetricCov3::from_diagonal(velocity_sigma.powi(2), velocity_sigma.powi(2), 0.01),
    )
    .expect("valid filter")
}

/// The share of the weight with a north position above zero.
fn north_mass(f: &MarginalizedParticleFilter<'_>) -> f64 {
    f.particles()
        .iter()
        .filter(|p| p.position_plane_m()[0] > 0.0)
        .map(|p| p.log_weight().exp())
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
    let mut storage = Storage::new(PARTICLES);
    let mut f = filter(
        &mut storage,
        MpfConfig::default(),
        600.0,
        NedVelocity::new(0.0, 50.0, 0.0),
        0.1,
    );
    let mut truth = (500.0, 0.0);
    let fly = |f: &mut MarginalizedParticleFilter<'_>, truth: &mut (f64, f64), v: (f64, f64)| {
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
    let config = MpfConfig::default();
    let truth_v = (30.0, 20.0);
    let mut storage = Storage::new(1_000);
    let mut f = filter(
        &mut storage,
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
    let run = |storage: &mut Storage| {
        let mut f = filter(
            storage,
            MpfConfig::default(),
            50.0,
            NedVelocity::new(10.0, 0.0, 0.0),
            1.0,
        );
        f.predict(1.0).expect("valid");
        f.ingest_altitude(505.0, 4.0).expect("valid");
        f.estimate()
    };
    let (mut first, mut second) = (Storage::new(PARTICLES), Storage::new(PARTICLES));
    assert_eq!(run(&mut first), run(&mut second));
    let mut storage = Storage::new(PARTICLES);
    let mut c = filter(
        &mut storage,
        MpfConfig::default(),
        50.0,
        NedVelocity::new(10.0, 0.0, 0.0),
        1.0,
    );
    c.predict(1.0).expect("valid");
    let before = c.estimate();
    let particles: Vec<MpfParticle> = c.particles().to_vec();
    assert_eq!(c.predict(0.0), Err(MpfError::NonFiniteValue));
    assert_eq!(
        c.ingest_altitude(500.0, 0.0),
        Err(MpfError::ImplausibleCovariance)
    );
    assert_eq!(
        c.ingest_scalar_map(&NoCoverage, 0.0, 1e-18),
        Err(MpfError::NoParticleOnMap {
            particle_count: PARTICLES
        })
    );
    assert_eq!(c.estimate(), before);
    assert_eq!(c.particles(), particles.as_slice());
}

#[test]
fn the_pool_needs_two_slices_of_one_nonzero_length() {
    let mut short = vec![MpfParticle::ZERO; 3];
    let mut long = vec![MpfParticle::ZERO; 4];
    assert_eq!(
        ParticlePool::new(&mut short, &mut long).map(|pool| pool.len()),
        Err(MpfError::InvalidConfig)
    );
    let (mut empty_a, mut empty_b) = (Vec::new(), Vec::new());
    assert!(ParticlePool::new(&mut empty_a, &mut empty_b).is_err());
    let invalid = MpfConfig {
        resample_threshold_ratio: 1.5,
        ..MpfConfig::default()
    };
    let mut storage = Storage::new(8);
    assert!(matches!(
        MarginalizedParticleFilter::new(
            invalid,
            storage.pool(),
            origin(),
            BodyEllipsoid::WGS84,
            SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
            NedVelocity::new(0.0, 0.0, 0.0),
            SymmetricCov3::from_diagonal(1.0, 1.0, 1.0)
        ),
        Err(MpfError::InvalidConfig)
    ));
}

#[test]
fn resampling_swaps_the_slices_and_keeps_the_particle_count() {
    let mut storage = Storage::new(64);
    let mut f = filter(
        &mut storage,
        MpfConfig {
            resample_threshold_ratio: 1.0,
            ..MpfConfig::default()
        },
        50.0,
        NedVelocity::new(0.0, 0.0, 0.0),
        1.0,
    );
    f.ingest_altitude(560.0, 1.0).expect("valid");
    let weights: Vec<f64> = f.particles().iter().map(MpfParticle::log_weight).collect();
    let uniform = -(64.0_f64).ln();
    assert!(
        weights.iter().all(|w| (w - uniform).abs() < 1e-12),
        "resampled"
    );
    assert_eq!(f.particles().len(), 64);
    assert!((f.effective_sample_size() - 64.0).abs() < 1e-9);
}
