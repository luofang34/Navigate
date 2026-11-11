//! Accuracy and age of a layer as data.

use alloc::string::String;

use navigate_geodesy::float_math::hypot;
use serde::{Deserialize, Serialize};

/// A one-sigma value in metres, or an explicit unknown.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sigma {
    /// The provider states the value.
    Known(f64),
    /// No one states the value.
    Unknown,
}

/// When the source data was collected.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Acquisition {
    /// An ISO 8601 date or year, as the provider states it.
    Date(String),
    /// No one states the date.
    Unknown,
}

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

/// The declared accuracy of one layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Accuracy {
    /// Horizontal one-sigma error of a feature position.
    pub horizontal: Sigma,
    /// Vertical one-sigma error of a height.
    pub vertical: Sigma,
    /// When the data was collected.
    pub acquired: Acquisition,
    /// Growth of the horizontal error per year after collection, for
    /// example from construction and vegetation.
    pub age_growth_per_year: Sigma,
    /// The surface that the layer shows.
    pub surface: SurfaceModel,
}

impl Accuracy {
    /// An accuracy with each value unknown.
    #[must_use]
    pub const fn unknown() -> Self {
        Self {
            horizontal: Sigma::Unknown,
            vertical: Sigma::Unknown,
            acquired: Acquisition::Unknown,
            age_growth_per_year: Sigma::Unknown,
            surface: SurfaceModel::Unknown,
        }
    }

    /// The horizontal sigma after `age_years`: the root sum square of the
    /// stated sigma and the age growth. Unknown when either is unknown.
    #[must_use]
    pub fn horizontal_after(&self, age_years: f64) -> Sigma {
        match (self.horizontal, self.age_growth_per_year) {
            (Sigma::Known(h), Sigma::Known(g)) => Sigma::Known(hypot(h, g * age_years.max(0.0))),
            _ => Sigma::Unknown,
        }
    }
}

#[cfg(test)]
mod tests;
