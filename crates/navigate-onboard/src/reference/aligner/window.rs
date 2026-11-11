//! One rest window of the alignment: what the sensors gave in it.

use nalgebra::Vector3;
use navigate_contract::{GeodeticPosition, SymmetricCov3};

use crate::PositionSource;

/// The mean and the scatter of the samples of one vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Scatter {
    count: u32,
    mean: Vector3<f64>,
    /// Sum of the squared deviations from the mean, each axis.
    squares: Vector3<f64>,
}

impl Default for Scatter {
    fn default() -> Self {
        Self {
            count: 0,
            mean: Vector3::zeros(),
            squares: Vector3::zeros(),
        }
    }
}

impl Scatter {
    /// Adds one sample.
    pub(crate) fn add(&mut self, sample: Vector3<f64>) {
        self.count = self.count.saturating_add(1);
        let before = sample - self.mean;
        self.mean += before / f64::from(self.count);
        self.squares += before.component_mul(&(sample - self.mean));
    }

    pub(crate) const fn count(&self) -> u32 {
        self.count
    }

    pub(crate) const fn mean(&self) -> Vector3<f64> {
        self.mean
    }

    /// The largest one-sigma of the three axes.
    pub(crate) fn largest_sigma(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        (self.squares.max() / f64::from(self.count - 1)).sqrt()
    }
}

/// What the sensors gave in one rest window.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct Window {
    /// Time that the inertial increments of the window cover, seconds.
    pub seconds: f64,
    /// Specific force of each increment, m/s², body axes.
    pub force: Scatter,
    /// Body rate of each increment, rad/s.
    pub rate: Scatter,
    /// Magnetic field samples, body axes.
    pub field: Scatter,
    /// A sensor said that the vehicle is at rest.
    pub rest_evidence: bool,
    /// A GNSS velocity said that the vehicle moves.
    pub moved: bool,
    /// The last position with its covariance in north, east, down axes.
    pub position: Option<(GeodeticPosition, SymmetricCov3, PositionSource)>,
}
