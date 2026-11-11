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
//!
//! What a block does to the core block:
//!
//! - The altitude bias, the magnetic bias, and the clock have no term in
//!   the dynamics of the core block. Only their own measurements read
//!   them. A layout without such a block gives the same core estimate
//!   as one with it, when no measurement of that block arrives.
//! - The empirical acceleration enters the velocity in the force-model
//!   propagation when `InertialConfig::dmc` is set. A layout without the
//!   block refuses that configuration; without `dmc` the block has zero
//!   rows and is equivalent to its absence.

use nalgebra::{Const, DimMin};
use navigate_astrodynamics::{PlanetaryBody, SpaceEnvironment};
use navigate_contract::ClockDomainId;
use thiserror::Error;

use super::InertialNavigationFilter;
use super::config::InertialConfig;
use super::regime::Regime;

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

    /// The core block alone, 15 states: an IMU with any means that
    /// needs no bias state of its own, such as position fixes, velocity
    /// fixes, ranges, terrain clearance, lines of sight, and visual
    /// poses.
    pub const CORE: Self = Self {
        dimension: CORE_STATES,
        alt_bias: None,
        mag_bias: None,
        dmc: None,
        clock: None,
    };

    /// The core block and the barometric altitude bias (15), 16 states:
    /// an aircraft with an altimeter.
    pub const AVIATION: Self = Self {
        dimension: CORE_STATES + 1,
        alt_bias: Some(CORE_STATES),
        mag_bias: None,
        dmc: None,
        clock: None,
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

/// Why a filter refuses its layout at construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum LayoutError {
    /// The layout does not fit the dimension of the filter, or its
    /// blocks overlap.
    #[error("the layout of dimension {layout} does not fit a filter of dimension {filter}")]
    Dimension {
        /// The dimension of the layout.
        layout: usize,
        /// The dimension of the filter.
        filter: usize,
    },
    /// The configuration has a model whose block the layout does not
    /// have: the empirical acceleration (`dmc`) or the clock.
    #[error("the configuration needs the {block} block, which the layout does not have")]
    MissingBlock {
        /// The name of the block.
        block: &'static str,
    },
}

impl StateLayout {
    /// Refuses a layout that does not fit `n`, and a configuration with a
    /// model that has no block in the layout.
    fn check(&self, n: usize, config: &InertialConfig) -> Result<(), LayoutError> {
        if !self.fits(n) {
            return Err(LayoutError::Dimension {
                layout: self.dimension,
                filter: n,
            });
        }
        if config.dmc.is_some() && self.dmc.is_none() {
            return Err(LayoutError::MissingBlock { block: "dmc" });
        }
        if config.clock_model.is_some() && self.clock.is_none() {
            return Err(LayoutError::MissingBlock { block: "clock" });
        }
        Ok(())
    }
}

impl<const N: usize> InertialNavigationFilter<N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    /// Builds an uninitialized filter of the layout `config.layout`, of
    /// dimension `N`, on the surface of `body`, as
    /// [`InertialNavigationFilter::new_surface`] does for
    /// [`StateLayout::FULL`].
    ///
    /// # Errors
    ///
    /// [`LayoutError`] for a layout that does not fit `N` or that lacks a
    /// block the configuration needs.
    pub fn surface_with_layout(
        config: InertialConfig,
        clock: ClockDomainId,
        body: PlanetaryBody,
    ) -> Result<Self, LayoutError> {
        config.layout.check(N, &config)?;
        let mut config = config;
        if body.ellipsoid != navigate_geodesy::BodyEllipsoid::WGS84 {
            config.fusion.require_explicit_frame = true;
        }
        Ok(Self::build(config, clock, Regime::surface_of(&body)))
    }

    /// Builds an uninitialized filter of the layout `config.layout`, of
    /// dimension `N`, in the frame of `env`, as
    /// [`InertialNavigationFilter::new_space`] does for
    /// [`StateLayout::FULL`].
    ///
    /// # Errors
    ///
    /// [`LayoutError`] for a layout that does not fit `N` or that lacks a
    /// block the configuration needs.
    pub fn space_with_layout(
        config: InertialConfig,
        clock: ClockDomainId,
        env: SpaceEnvironment,
    ) -> Result<Self, LayoutError> {
        config.layout.check(N, &config)?;
        let mut config = config;
        config.fusion.require_explicit_frame = true;
        Ok(Self::build(config, clock, Regime::space(env)))
    }

    /// The layout of the error state.
    #[must_use]
    pub const fn layout(&self) -> StateLayout {
        self.config.layout
    }
}
