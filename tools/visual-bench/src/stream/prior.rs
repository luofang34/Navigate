//! Camera priors. Exactly one position and one orientation representation is required.

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use navigate_visual::LocalFrame;
use navigate_visual::{CameraPose, PosePrior, VisualError};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PriorRecord {
    /// Meters east, north, and up from the package anchor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position_enu_m: Option<[f64; 3]>,
    /// Latitude, longitude in degrees, and altitude in the package's datum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geodetic_lat_lon_alt_m: Option<[f64; 3]>,
    /// Unit quaternion in XYZW order. It rotates camera axes into local ENU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eye_to_enu_xyzw: Option<[f64; 4]>,
    /// Heading clockwise from north, tilt from down, and clockwise camera roll.
    /// Zero tilt looks down. A tilt of 90 degrees looks horizontally forward.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heading_tilt_roll_deg: Option<[f64; 3]>,
    pub position_radius_m: f64,
    pub attitude_radius_rad: f64,
}

impl PriorRecord {
    pub fn prior(&self, frame: LocalFrame) -> Result<PosePrior, VisualError> {
        let position = match (self.position_enu_m, self.geodetic_lat_lon_alt_m) {
            (Some(position), None) => Vector3::from_row_slice(&position),
            (None, Some(position)) => frame.local(position)?,
            _ => {
                return Err(VisualError::Invalid {
                    field: "exactly one prior position representation",
                });
            }
        };
        let prior = PosePrior {
            pose: CameraPose {
                position,
                orientation: self.orientation()?,
            },
            position_radius_m: self.position_radius_m,
            attitude_radius_rad: self.attitude_radius_rad,
        };
        prior.validate()?;
        Ok(prior)
    }

    fn orientation(&self) -> Result<UnitQuaternion<f64>, VisualError> {
        match (self.eye_to_enu_xyzw, self.heading_tilt_roll_deg) {
            (Some([x, y, z, w]), None) => {
                let q = Quaternion::new(w, x, y, z);
                if !q.coords.iter().all(|v| v.is_finite()) || (q.norm_squared() - 1.0).abs() > 1e-6
                {
                    return Err(VisualError::Invalid {
                        field: "unit camera quaternion",
                    });
                }
                Ok(UnitQuaternion::new_normalize(q))
            }
            (None, Some([heading, tilt, roll]))
                if [heading, tilt, roll].iter().all(|v| v.is_finite())
                    && (0.0..=180.0).contains(&tilt) =>
            {
                Ok(
                    UnitQuaternion::from_axis_angle(&Vector3::z_axis(), -heading.to_radians())
                        * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), tilt.to_radians())
                        * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), -roll.to_radians()),
                )
            }
            _ => Err(VisualError::Invalid {
                field: "exactly one valid prior orientation representation",
            }),
        }
    }
}

impl From<PosePrior> for PriorRecord {
    fn from(prior: PosePrior) -> Self {
        Self {
            position_enu_m: Some(prior.pose.position.into()),
            geodetic_lat_lon_alt_m: None,
            eye_to_enu_xyzw: Some(prior.pose.orientation.coords.into()),
            heading_tilt_roll_deg: None,
            position_radius_m: prior.position_radius_m,
            attitude_radius_rad: prior.attitude_radius_rad,
        }
    }
}

#[cfg(test)]
mod tests;
