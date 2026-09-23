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

/// One-sigma error terms that the host declares for a map, camera, and datum.
///
/// Every term must be positive. Unknown error is not zero error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisualErrorBudget {
    /// Horizontal registration error of the map imagery in metres.
    pub map_horizontal_m: f64,
    /// Vertical error of the map elevation model in metres.
    pub map_vertical_m: f64,
    /// Position error from camera calibration and mounting in metres.
    pub calibration_m: f64,
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

    /// Horizontal and vertical variances in square metres, with the frame-model error.
    pub(crate) fn variances_m2(&self, model_error_m: f64) -> (f64, f64) {
        let model = model_error_m * model_error_m;
        let calibration = self.calibration_m * self.calibration_m;
        let horizontal = self.map_horizontal_m * self.map_horizontal_m + calibration + model;
        let vertical = self.map_vertical_m * self.map_vertical_m
            + calibration
            + self.vertical_datum_m * self.vertical_datum_m
            + model;
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
