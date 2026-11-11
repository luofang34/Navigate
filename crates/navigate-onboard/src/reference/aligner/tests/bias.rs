//! The IMU bias that starts the filter, from a rest window or from the
//! prior of a transfer alignment, with each variant of the filter.

use nalgebra::Vector3;
use navigate_contract::SymmetricCov3;

use super::rig::{CLOCK, Rig, at};
use super::{ATTITUDE_DEG, feed, normalized_errors};
use crate::{
    Alignment, AlignmentStatus, ErrorBudget, GnssCorrelatedError, ImuBiasPrior, Navigator,
    NavigatorFilter, ReferenceNavigator, SensorSuite,
};

/// The one-sigma of the gyroscope prior of a ground calibration, rad/s.
const PRIOR_SIGMA: f64 = 1e-4;

/// The correlated-error blocks of a navigator: one for each means of its
/// suite with a correlated error.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Blocks {
    /// No means with a correlated error.
    None,
    /// A GNSS receiver that declares a slow error.
    Gnss,
    /// That receiver and a terrain camera, whose map cells share an error.
    #[cfg(feature = "terrain-camera")]
    GnssAndMap,
}

impl Blocks {
    const ALL: &[Self] = &[
        Self::None,
        Self::Gnss,
        #[cfg(feature = "terrain-camera")]
        Self::GnssAndMap,
    ];

    /// Whether `filter` is the variant of these blocks.
    fn built(self, filter: &NavigatorFilter) -> bool {
        match self {
            Self::None => matches!(filter, NavigatorFilter::Full(_)),
            Self::Gnss => matches!(filter, NavigatorFilter::OneBlock(_)),
            #[cfg(feature = "terrain-camera")]
            Self::GnssAndMap => matches!(filter, NavigatorFilter::TwoBlocks(_)),
        }
    }
}

/// The navigator of `rig` with the blocks `blocks`, told the field of the
/// place.
fn navigator(rig: &Rig, blocks: Blocks) -> ReferenceNavigator {
    let b = rig.budgets;
    let slow = (blocks != Blocks::None).then_some(GnssCorrelatedError {
        tau_s: 60.0,
        horizontal_sigma_m: 2.0,
        vertical_sigma_m: 3.0,
    });
    let gnss = ErrorBudget::Gnss {
        horizontal_sigma_m: b.gnss_horizontal_m,
        vertical_sigma_m: b.gnss_vertical_m,
        velocity_sigma_m_s: b.gnss_velocity_m_s,
        correlated: slow,
    };
    #[cfg_attr(not(feature = "terrain-camera"), expect(unused_mut))]
    let mut sensors = rig
        .suite_with(Some(b.mag_disturbance_rad), gnss)
        .sensors()
        .to_vec();
    #[cfg(feature = "terrain-camera")]
    if blocks == Blocks::GnssAndMap {
        use crate::SensorKind;
        use crate::reference::terrain::tests::fixes::belly;
        use crate::reference::terrain::tests::flight::{imager, suite};
        let terrain = suite(imager(), belly());
        sensors.extend(terrain.of_kind(SensorKind::TerrainCamera).cloned());
    }
    let suite = SensorSuite::new(sensors).unwrap();
    let navigator = rig.told(ReferenceNavigator::new(suite, CLOCK, None).unwrap());
    assert!(blocks.built(navigator.filter()), "{blocks:?}");
    navigator
}

/// The heading drift over 3 s after a state at 2 s, for a vehicle at rest
/// with 2 mrad/s of gyroscope bias about the vertical. `transfer` is
/// `None` for a self-alignment, else a transfer alignment of the same
/// state with or without a bias prior, with the filter of `blocks`.
fn heading_drift(transfer: Option<bool>, blocks: Blocks) -> f64 {
    let mut rig = Rig::new(ATTITUDE_DEG, 71, 1.0);
    rig.gyro_bias = Vector3::new(2e-4, -3e-4, 2e-3);
    let b = rig.budgets;
    let mut navigator = navigator(&rig, blocks);
    let mut sink = Vec::new();
    feed(&mut navigator, &mut rig, 1..=200, &mut sink);
    let aligned = navigator.solution(at(200)).unwrap();
    if let Some(with_prior) = transfer {
        let accel = b.accel_bias.powi(2);
        let gyro = PRIOR_SIGMA * PRIOR_SIGMA;
        // The prior of a ground calibration: the true bias, to its
        // one-sigma.
        let prior = ImuBiasPrior::new(
            [0.0; 3],
            SymmetricCov3::from_diagonal(accel, accel, accel),
            (rig.gyro_bias + Vector3::new(0.0, 0.0, 0.5 * PRIOR_SIGMA)).into(),
            SymmetricCov3::from_diagonal(gyro, gyro, gyro),
        );
        navigator
            .align(&Alignment {
                position: aligned.position,
                velocity: aligned.velocity,
                attitude: aligned.attitude.unwrap(),
                position_cov: aligned.position_cov,
                velocity_cov: aligned.velocity_cov,
                attitude_cov: aligned.attitude_cov.unwrap(),
                at: at(200),
                imu_bias_prior: with_prior.then_some(prior),
            })
            .unwrap();
        let Some(AlignmentStatus::Aligned { record, .. }) = navigator.alignment_status() else {
            panic!("aligned by transfer");
        };
        assert_eq!(record.gyro_bias_rad_s.is_some(), with_prior);
    }
    let start = normalized_errors(&rig, &aligned).3;
    feed(&mut navigator, &mut rig, 201..=500, &mut sink);
    let end = normalized_errors(&rig, &navigator.solution(at(500)).unwrap()).3;
    (end - start).abs()
}

/// The gyroscope bias of the rest window starts the filter's bias: the
/// vehicle keeps its heading after the state. A transfer alignment that
/// carries the bias in its prior, as a ground calibration before an
/// in-air start gives it, keeps the heading too, to the prior's one-sigma
/// times the time; one without a prior turns the heading by the bias
/// times the time. Each variant of the filter takes the bias as the
/// 22-state filter does.
#[test]
fn the_gyro_bias_of_a_window_or_a_prior_starts_the_filter_of_each_layout() {
    for blocks in Blocks::ALL.iter().copied() {
        let own = heading_drift(None, blocks);
        let with_prior = heading_drift(Some(true), blocks);
        let without = heading_drift(Some(false), blocks);
        // 2 mrad/s over 3 s is 6 mrad. The bias of the window is known to
        // the gyroscope noise over 2 s, 0.07 mrad/s; that of the prior to
        // 0.1 mrad/s, 0.3 mrad over 3 s.
        assert!(own < 1e-3, "{blocks:?} {own}");
        assert!(
            with_prior < 3.0 * PRIOR_SIGMA * 3.0,
            "{blocks:?} {with_prior}"
        );
        assert!(without > 4e-3, "{blocks:?} {without}");
    }
}
