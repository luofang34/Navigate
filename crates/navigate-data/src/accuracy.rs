//! Accuracy and age of a layer as data, in integers.

use navigate_geodesy::float_math::hypot;
use serde::{Deserialize, Serialize};

use crate::DayNumber;

/// Which surface the elevation or the imagery shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceModel {
    /// The bare ground (DTM).
    BareEarth,
    /// The top of vegetation and structures (DSM).
    FirstReturn,
    /// No one states it.
    Unknown,
}

/// The declared accuracy of one layer. `None` means unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Accuracy {
    /// Horizontal one-sigma error of a feature position, millimetres.
    pub horizontal_mm: Option<u32>,
    /// Vertical one-sigma error of a height, millimetres.
    pub vertical_mm: Option<u32>,
    /// The day of collection.
    pub acquired: Option<DayNumber>,
    /// Growth of the horizontal one-sigma per year after collection, for
    /// example from construction and vegetation, millimetres per year.
    pub age_growth_mm_per_year: Option<u32>,
    /// The surface that the layer shows.
    pub surface: SurfaceModel,
}

impl Accuracy {
    /// An accuracy with each value unknown.
    #[must_use]
    pub const fn unknown() -> Self {
        Self {
            horizontal_mm: None,
            vertical_mm: None,
            acquired: None,
            age_growth_mm_per_year: None,
            surface: SurfaceModel::Unknown,
        }
    }

    /// The horizontal one-sigma in metres on day `today`: the root sum
    /// square of the stated value and the growth since collection. `None`
    /// when a needed value is unknown.
    #[must_use]
    pub fn horizontal_sigma_m(&self, today: DayNumber) -> Option<f64> {
        let base = f64::from(self.horizontal_mm?) / 1_000.0;
        let growth = f64::from(self.age_growth_mm_per_year?) / 1_000.0;
        let years = f64::from(today.0.saturating_sub(self.acquired?.0).max(0)) / 365.25;
        Some(hypot(base, growth * years))
    }
}

#[cfg(test)]
mod tests;
