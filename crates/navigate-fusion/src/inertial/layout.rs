//! The layout of the error state: a core block that every filter has,
//! and optional blocks that only some means of navigation need
//! (ADR-0023).
//!
//! The core block is the position, the velocity, the attitude, and the
//! accelerometer and gyroscope biases: 15 states at fixed indices. Each
//! optional block has an offset or none: the barometric altitude bias,
//! the magnetic map bias, the empirical acceleration of the dynamic
//! model compensation, and the receiver clock. A filter stores its
//! covariance with the dimension of its layout, so a layout without a
//! block does not pay for it.

/// The number of states of the core block.
pub const CORE_STATES: usize = 15;

/// The offsets of the error-state blocks of one filter. The core block
/// is at fixed indices; each optional block has an offset or none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct StateLayout {
    dimension: usize,
    alt_bias: Option<usize>,
    mag_bias: Option<usize>,
    dmc: Option<usize>,
    clock: Option<usize>,
}

impl StateLayout {
    /// Every block, 22 states: the core block (0–14), the altitude bias
    /// (15), the magnetic bias (16), the empirical acceleration (17–19),
    /// and the clock bias and drift (20–21).
    pub const FULL: Self = Self {
        dimension: 22,
        alt_bias: Some(15),
        mag_bias: Some(16),
        dmc: Some(17),
        clock: Some(20),
    };

    /// The number of error states.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// First index of the position error in meters, in the frame axes.
    #[must_use]
    pub const fn position(&self) -> usize {
        0
    }

    /// First index of the velocity error in m/s, in the frame axes.
    #[must_use]
    pub const fn velocity(&self) -> usize {
        3
    }

    /// First index of the attitude error in radians, in the body axes.
    #[must_use]
    pub const fn attitude(&self) -> usize {
        6
    }

    /// First index of the accelerometer bias error in m/s², in the body
    /// axes.
    #[must_use]
    pub const fn accel_bias(&self) -> usize {
        9
    }

    /// First index of the gyroscope bias error in rad/s, in the body
    /// axes.
    #[must_use]
    pub const fn gyro_bias(&self) -> usize {
        12
    }

    /// Index of the barometric altitude bias error in meters, or none.
    #[must_use]
    pub const fn alt_bias(&self) -> Option<usize> {
        self.alt_bias
    }

    /// Index of the scalar magnetic map bias error in tesla, or none.
    #[must_use]
    pub const fn mag_bias(&self) -> Option<usize> {
        self.mag_bias
    }

    /// First index of the three empirical acceleration errors of the
    /// dynamic model compensation in m/s², in the frame axes, or none.
    #[must_use]
    pub const fn dmc(&self) -> Option<usize> {
        self.dmc
    }

    /// Index of the receiver clock range bias error `c·δt` in meters, or
    /// none. The range-rate drift error `c·δṫ` in m/s follows it.
    #[must_use]
    pub const fn clock(&self) -> Option<usize> {
        self.clock
    }

    /// Whether the layout has dimension `n` and its blocks fit in it
    /// after the core block without an overlap.
    #[must_use]
    pub fn fits(&self, n: usize) -> bool {
        if self.dimension != n || n < CORE_STATES {
            return false;
        }
        let blocks = [
            (self.alt_bias, 1),
            (self.mag_bias, 1),
            (self.dmc, 3),
            (self.clock, 2),
        ];
        let mut used = [false; 64];
        if n > used.len() {
            return false;
        }
        for (offset, width) in blocks {
            let Some(start) = offset else {
                continue;
            };
            if start < CORE_STATES || start.saturating_add(width) > n {
                return false;
            }
            for index in start..start + width {
                let Some(slot) = used.get_mut(index) else {
                    return false;
                };
                if *slot {
                    return false;
                }
                *slot = true;
            }
        }
        true
    }
}
