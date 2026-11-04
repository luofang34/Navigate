//! Error terms that image geometry cannot measure.

use nalgebra::UnitQuaternion;

use crate::fusion_bridge::VisualFusionError;

/// Vertical reference of the altitudes in the map package.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VerticalDatum {
    /// Heights above the reference ellipsoid of the visual frame
    /// ([`crate::LocalFrame::reference_ellipsoid`]). For an Earth frame, this
    /// is the WGS84 ellipsoid. For a [`crate::LocalFrame::BodyTangentEnu`]
    /// frame, this is the body ellipsoid of the frame.
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

/// Camera mounting and attitude error terms that image geometry cannot measure.
///
/// Each term is a root-mean-square angular allowance in radians. It includes
/// bias as well as random scatter. The adapter does not measure or validate
/// the values. Every term must be positive. Unknown error is not zero error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisualAttitudeBudget {
    /// Rotation from camera right, up, back to body forward, right, down.
    /// A gimbal host supplies this rotation at capture time.
    pub eye_to_body: UnitQuaternion<f64>,
    /// Angular error of the camera mounting, or of the gimbal angles, in radians.
    pub mounting_rad: f64,
    /// Angular error of the map orientation and of the reference render, in radians.
    pub map_rad: f64,
}

impl VisualAttitudeBudget {
    /// Validate that `eye_to_body` is finite and that all angular allowances are positive.
    ///
    /// # Errors
    /// Refuses a non-finite mounting rotation or a non-positive angular error term.
    pub fn validate(&self) -> Result<(), VisualFusionError> {
        for (field, value) in [
            ("mounting_rad", self.mounting_rad),
            ("map_rad", self.map_rad),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(VisualFusionError::InvalidBudget { field });
            }
        }
        if !self.eye_to_body.coords.iter().all(|v| v.is_finite()) {
            return Err(VisualFusionError::InvalidBudget {
                field: "eye_to_body",
            });
        }
        Ok(())
    }

    /// Angular variance in square radians about each axis. Unknown correlation
    /// between the two terms adds their RMS values.
    pub(crate) fn variance_rad2(&self) -> f64 {
        (self.mounting_rad + self.map_rad).powi(2)
    }
}
