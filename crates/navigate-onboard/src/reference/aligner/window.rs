//! One rest window of the alignment: the inertial increments of one
//! unbroken stretch of time, and what the other sensors gave in it, each
//! with its stamp.

use nalgebra::Vector3;
use navigate_contract::{GeodeticPosition, MonotonicNanos, SymmetricCov3};

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

    /// The largest sample variance of the three axes; `None` below two
    /// samples, which have no scatter.
    pub(crate) fn largest_variance(&self) -> Option<f64> {
        (self.count >= 2).then(|| self.squares.max() / f64::from(self.count - 1))
    }
}

/// A GNSS velocity: the velocity, north, east, down, and its one-sigma on
/// each axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Velocity {
    pub ned_m_s: Vector3<f64>,
    pub sigma_m_s: f64,
}

/// A position with its covariance in north, east, down axes, and what gave
/// it.
pub(crate) type Position = (GeodeticPosition, SymmetricCov3, PositionSource);

/// What the sensors gave in one rest window.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct Window {
    /// The start of the first increment and the end of the last.
    pub span: Option<(MonotonicNanos, MonotonicNanos)>,
    /// Specific force of each increment, m/s², body axes.
    pub force: Scatter,
    /// Body rate of each increment, rad/s.
    pub rate: Scatter,
    /// Magnetic field samples, body axes, tesla.
    pub field: Vec<(MonotonicNanos, Vector3<f64>)>,
    /// GNSS velocities.
    pub velocities: Vec<(MonotonicNanos, Velocity)>,
    /// Statements that the vehicle is at rest on a surveyed point.
    pub surveys: Vec<MonotonicNanos>,
    /// Positions.
    pub positions: Vec<(MonotonicNanos, Position)>,
}

impl Window {
    /// Time that the increments of the window cover, seconds.
    pub(crate) fn seconds(&self) -> f64 {
        self.span
            .map_or(0.0, |(start, end)| seconds_between(start, end))
    }

    /// A window that holds nothing of this one but what the other sensors
    /// gave at or after `from`.
    pub(crate) fn from(&self, from: MonotonicNanos) -> Self {
        let after = |at: &MonotonicNanos| *at >= from;
        Self {
            field: self
                .field
                .iter()
                .filter(|(at, _)| after(at))
                .copied()
                .collect(),
            velocities: self
                .velocities
                .iter()
                .filter(|(at, _)| after(at))
                .copied()
                .collect(),
            surveys: self
                .surveys
                .iter()
                .filter(|at| after(at))
                .copied()
                .collect(),
            positions: self
                .positions
                .iter()
                .filter(|(at, _)| after(at))
                .copied()
                .collect(),
            ..Self::default()
        }
    }

    /// Whether `at` lies in the span of the increments.
    pub(crate) fn holds(&self, at: MonotonicNanos) -> bool {
        self.span
            .is_some_and(|(start, end)| start <= at && at <= end)
    }
}

/// Seconds from `from` to `to`; negative when `to` is earlier.
pub(crate) fn seconds_between(from: MonotonicNanos, to: MonotonicNanos) -> f64 {
    let nanos = i128::from(to.as_nanos()) - i128::from(from.as_nanos());
    nanos as f64 * 1e-9
}
