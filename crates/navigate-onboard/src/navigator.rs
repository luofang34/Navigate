//! The navigator trait.

use navigate_contract::{
    AttitudeQuaternion, GeodeticPosition, MonotonicNanos, NavigationSolution, NedVelocity,
    SymmetricCov3,
};

use core::any::Any;

use navigate_data::product::{KindName, ProductKind, ProductUse};
use navigate_data::{LayerKind, Sha256Digest};

use crate::{AcceptanceSink, OnboardError, SensorInput, SensorSuite};

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

/// The initial state of a navigator, with its covariances.
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
}

/// A navigator: sensor inputs in, a navigation solution out.
pub trait Navigator {
    /// The declared sensors.
    fn suite(&self) -> &SensorSuite;

    /// Starts the state.
    ///
    /// # Errors
    ///
    /// Refuses an alignment that the filter does not accept.
    fn align(&mut self, alignment: &Alignment) -> Result<(), OnboardError>;

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
