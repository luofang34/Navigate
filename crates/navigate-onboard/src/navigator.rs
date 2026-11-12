//! The navigator trait.

use navigate_contract::{
    AttitudeQuaternion, GeodeticPosition, MonotonicNanos, NavigationSolution, NedVelocity,
    SymmetricCov3,
};

use core::any::Any;

use navigate_data::product::{KindName, ProductKind, ProductUse};
use navigate_data::{LayerKind, Sha256Digest};
use navigate_fusion::ImuBiasPrior;

use crate::{
    AcceptanceSink, AlignmentConfig, AlignmentStatus, GnssIntegrityConfig, GnssIntegrityStatus,
    MagneticReference, OnboardError, SensorInput, SensorSuite,
};

/// How much a navigator needs one kind of data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Need {
    /// The navigator refuses to load without it.
    Required,
    /// The navigator uses it when it is present.
    Optional,
}

/// One kind of data that a navigator needs or can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataRequirement {
    /// A core layer kind, or `None` for a product requirement.
    pub layer: Option<LayerKind>,
    /// A product kind, or `None` for a layer requirement.
    pub product: Option<ProductKind>,
    /// Altitude bands that the navigator can use, inclusive, or `None` for
    /// any band.
    pub altitude_bands: Option<(u32, u32)>,
    /// How much the navigator needs it.
    pub need: Need,
}

/// What a navigator did with one product chunk at load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductReport {
    /// The kind of the chunk.
    pub kind: KindName,
    /// The chunk digest.
    pub chunk: Sha256Digest,
    /// Use, fallback, or refusal, with the reason.
    pub decision: ProductUse,
}

/// The state that a transfer alignment gives a navigator, with its
/// covariances: the state of another estimator or of a survey.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Alignment {
    /// Position.
    pub position: GeodeticPosition,
    /// Velocity, NED.
    pub velocity: NedVelocity,
    /// Body-to-NED attitude.
    pub attitude: AttitudeQuaternion,
    /// Position covariance, NED, m².
    pub position_cov: SymmetricCov3,
    /// Velocity covariance, NED, (m/s)².
    pub velocity_cov: SymmetricCov3,
    /// Attitude covariance, rad².
    pub attitude_cov: SymmetricCov3,
    /// When the state is valid, on the navigator clock.
    pub at: MonotonicNanos,
    /// The IMU biases at the state, with their covariances, in the body
    /// axes: for example the gyroscope turn-on bias that a ground
    /// calibration measured in a rest window before an in-air start. The
    /// filter starts its biases from it; `None` starts them at zero with
    /// their stationary variances and clears a prior that an earlier
    /// alignment gave.
    pub imu_bias_prior: Option<ImuBiasPrior>,
}

/// A navigator: sensor inputs in, a navigation solution out.
pub trait Navigator {
    /// The declared sensors.
    fn suite(&self) -> &SensorSuite;

    /// Sets the state from a transfer alignment: the state of another
    /// estimator or of a survey, with its covariances. The source is never
    /// the true state of a simulation. A navigator that aligns itself
    /// needs no call; a call replaces an alignment in progress and an
    /// earlier state.
    ///
    /// # Errors
    ///
    /// Refuses an alignment that the filter does not accept.
    fn align(&mut self, alignment: &Alignment) -> Result<(), OnboardError>;

    /// Where the navigator is on the way to its first state, or `None`
    /// for a navigator that does not report it.
    fn alignment_status(&self) -> Option<AlignmentStatus> {
        None
    }

    /// Forgets the state and aligns again from the sensors. A navigator
    /// does not do this by itself in flight.
    ///
    /// # Errors
    ///
    /// Refuses when the navigator cannot align itself.
    fn realign(&mut self) -> Result<(), OnboardError> {
        Err(OnboardError::Config {
            field: "realign",
            detail: "this navigator does not align itself".into(),
        })
    }

    /// States the Earth's field at the place, for the heading of a
    /// magnetometer.
    ///
    /// # Errors
    ///
    /// Refuses a reference that is not usable, and a navigator that does
    /// not take one.
    fn set_magnetic_reference(&mut self, reference: MagneticReference) -> Result<(), OnboardError> {
        Err(OnboardError::Config {
            field: "magnetic reference",
            detail: format!("this navigator does not take one: {reference:?}"),
        })
    }

    /// Sets how the navigator aligns itself. A host that holds the
    /// navigator as a trait object reaches it here.
    ///
    /// # Errors
    ///
    /// Refuses a configuration that is not usable, and a navigator that
    /// does not align itself.
    fn set_alignment_config(&mut self, config: AlignmentConfig) -> Result<(), OnboardError> {
        Err(OnboardError::Config {
            field: "alignment configuration",
            detail: format!("this navigator does not align itself: {config:?}"),
        })
    }

    /// Starts the check of the GNSS positions against the other sensors
    /// with `config`, or stops it with `None`. Without a call there is no
    /// check. A check that starts after the first state takes the
    /// solution of that time as honest.
    ///
    /// # Errors
    ///
    /// Refuses a configuration that is not usable, and a navigator that
    /// has no such check.
    fn set_gnss_integrity(
        &mut self,
        config: Option<GnssIntegrityConfig>,
    ) -> Result<(), OnboardError> {
        Err(OnboardError::Config {
            field: "GNSS integrity configuration",
            detail: format!("this navigator has no GNSS check: {config:?}"),
        })
    }

    /// The state of the GNSS check, or `None` without one.
    fn gnss_integrity(&self) -> Option<GnssIntegrityStatus> {
        None
    }

    /// Takes one input and writes what it did with each measurement into
    /// `sink`. A refused aiding measurement is a record, not an error.
    ///
    /// # Errors
    ///
    /// Refuses an input from an undeclared sensor, data of the wrong kind,
    /// and an IMU increment that the filter cannot take.
    fn ingest(
        &mut self,
        input: &SensorInput,
        sink: &mut dyn AcceptanceSink,
    ) -> Result<(), OnboardError>;

    /// The solution at `now`, or `None` before the alignment.
    fn solution(&mut self, now: MonotonicNanos) -> Option<NavigationSolution>;

    /// The data that the navigator needs or can use, before load. A
    /// navigator that uses no map data gives none.
    fn requirements(&self) -> Vec<DataRequirement> {
        Vec::new()
    }

    /// What the navigator did with each product chunk at load.
    fn load_report(&self) -> &[ProductReport] {
        &[]
    }

    /// A complete copy of the state, for a fork or a later restore.
    fn snapshot(&self) -> Box<dyn Navigator>;

    /// Restores the state of `snapshot`.
    ///
    /// # Errors
    ///
    /// Refuses a snapshot of another navigator type.
    fn restore(&mut self, snapshot: &dyn Navigator) -> Result<(), OnboardError>;

    /// The navigator as [`Any`], so that [`Navigator::restore`] can check
    /// the type of a snapshot.
    fn as_any(&self) -> &dyn Any;
}
