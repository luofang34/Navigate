//! Vertical datum of an elevation layer, and the conversion in the reader.
//!
//! A package keeps heights as the provider supplies them. The reader
//! converts them to heights above the WGS84 ellipsoid with a
//! [`GeoidUndulation`] model that the host supplies. Without a model, the
//! reader reports [`EllipsoidalHeight::UnknownUndulation`] with a bound;
//! it never takes an orthometric height as an ellipsoidal height.

use navigate_contract::Ellipsoid;
use navigate_geodesy::{GeoidUndulation, orthometric_to_ellipsoidal_m};
use serde::{Deserialize, Serialize};

/// Bound in metres on the size of the EGM96 undulation of the Earth: the
/// geoid is between approximately −107 m and +86 m.
pub const UNKNOWN_UNDULATION_BOUND_M: f64 = 107.0;

/// A geoid model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeoidModel {
    /// EGM96.
    Egm96,
    /// EGM2008.
    Egm2008,
    /// A model that this crate does not know.
    #[serde(other)]
    Unsupported,
}

/// The vertical reference of the heights in an elevation layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VerticalDatum {
    /// Heights above the ellipsoid of the body.
    Ellipsoidal,
    /// Heights above a geoid, for example mean sea level of EGM96.
    Orthometric {
        /// The geoid model.
        geoid: GeoidModel,
    },
    /// No one states the datum.
    Unknown,
    /// A datum that this crate does not know.
    #[serde(other)]
    Unsupported,
}

/// A height above the WGS84 ellipsoid, or why the reader cannot give one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EllipsoidalHeight {
    /// The height above the ellipsoid in metres.
    Known(f64),
    /// The undulation is unknown: the true ellipsoidal height is within
    /// `bound_m` of `height_m`.
    UnknownUndulation {
        /// The stored height in metres.
        height_m: f64,
        /// Bound on the error in metres.
        bound_m: f64,
    },
}

impl VerticalDatum {
    /// The ellipsoidal height of a stored height at a position in radians.
    /// `geoid` is the host model and the model it implements. Another model
    /// than the datum names, or no model, gives
    /// [`EllipsoidalHeight::UnknownUndulation`].
    #[must_use]
    pub fn to_ellipsoidal<M: GeoidUndulation + ?Sized>(
        &self,
        height_m: f64,
        latitude_rad: f64,
        longitude_rad: f64,
        geoid: Option<(GeoidModel, &M)>,
    ) -> EllipsoidalHeight {
        let unknown = EllipsoidalHeight::UnknownUndulation {
            height_m,
            bound_m: UNKNOWN_UNDULATION_BOUND_M,
        };
        match self {
            Self::Ellipsoidal => EllipsoidalHeight::Known(height_m),
            Self::Orthometric { geoid: wanted } => geoid
                .filter(|(model, _)| *model == *wanted && *wanted != GeoidModel::Unsupported)
                .and_then(|(_, model)| {
                    orthometric_to_ellipsoidal_m(
                        model,
                        Ellipsoid::WGS84,
                        latitude_rad,
                        longitude_rad,
                        height_m,
                    )
                })
                .map_or(unknown, EllipsoidalHeight::Known),
            Self::Unknown | Self::Unsupported => unknown,
        }
    }
}

#[cfg(test)]
mod tests;
