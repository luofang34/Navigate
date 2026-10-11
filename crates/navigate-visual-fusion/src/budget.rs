//! Error terms that image geometry cannot measure.

use crate::VisualFusionError;

/// Vertical reference of the altitudes in the map package.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VerticalDatum {
    /// Heights above the WGS84 ellipsoid.
    Wgs84Ellipsoid,
    /// Heights above a geoid. The host supplies the geoid separation `N`
    /// at the map anchor, so that ellipsoid height is `H + N`.
    Orthometric {
        /// Geoid height above the ellipsoid in metres.
        geoid_separation_m: f64,
    },
}

/// Root-mean-square error allowances declared by the host for a map, camera, and datum.
///
/// Each term must include bias as well as random scatter. These values are not
/// measured or validated by the adapter. Every term must be positive.
/// Unknown error is not zero error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisualErrorBudget {
    /// Horizontal registration error of the map imagery in metres.
    pub map_horizontal_m: f64,
    /// Vertical error of the map elevation model in metres.
    pub map_vertical_m: f64,
    /// Position error from camera calibration and mounting in metres.
    pub calibration_m: f64,
    /// Position error from the geometry model used to solve the pose, in metres.
    /// The host must assess the effect of the full map point set and viewing geometry.
    /// A coordinate difference at the camera or a map point is not a pose error bound.
    /// Coherent model errors must not shrink with the number of matched points.
    pub geometry_model_m: f64,
    /// Vertical reference of the map altitudes.
    pub vertical_datum: VerticalDatum,
    /// Error of the vertical datum conversion in metres.
    pub vertical_datum_m: f64,
}

impl VisualErrorBudget {
    pub(crate) fn validate(&self) -> Result<(), VisualFusionError> {
        let terms = [
            ("map_horizontal_m", self.map_horizontal_m),
            ("map_vertical_m", self.map_vertical_m),
            ("calibration_m", self.calibration_m),
            ("geometry_model_m", self.geometry_model_m),
            ("vertical_datum_m", self.vertical_datum_m),
        ];
        for (field, value) in terms {
            if !(value.is_finite() && value > 0.0) {
                return Err(VisualFusionError::InvalidBudget { field });
            }
        }
        if let VerticalDatum::Orthometric { geoid_separation_m } = self.vertical_datum
            && !geoid_separation_m.is_finite()
        {
            return Err(VisualFusionError::InvalidBudget {
                field: "geoid_separation_m",
            });
        }
        Ok(())
    }

    /// Horizontal and vertical variances in square metres, with the point-coordinate displacement.
    pub(crate) fn variances_m2(&self, model_error_m: f64) -> (f64, f64) {
        let horizontal =
            (self.map_horizontal_m + self.calibration_m + self.geometry_model_m + model_error_m)
                .powi(2);
        let vertical = (self.map_vertical_m
            + self.calibration_m
            + self.geometry_model_m
            + self.vertical_datum_m
            + model_error_m)
            .powi(2);
        (horizontal, vertical)
    }

    pub(crate) fn ellipsoid_height_m(&self, map_altitude_m: f64) -> f64 {
        match self.vertical_datum {
            VerticalDatum::Wgs84Ellipsoid => map_altitude_m,
            VerticalDatum::Orthometric { geoid_separation_m } => {
                map_altitude_m + geoid_separation_m
            }
        }
    }
}
