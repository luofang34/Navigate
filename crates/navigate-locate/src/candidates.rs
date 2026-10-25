//! Candidate camera poses that look at one ground point.
//!
//! A user marks the ground point at the centre of the photo and the direction in
//! which the camera looks. The camera position is then unknown along that line:
//! a high camera that looks down steeply and a low camera that looks forward can
//! show the same centre. The sampler places cameras along these lines for a few
//! headings, downward angles and heights, all aimed at the same point.
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use navigate_visual::CameraPose;

/// Look-at candidate grid.
#[derive(Clone, Debug)]
pub struct LookAt {
    /// Compass heading of the view, degrees clockwise from north.
    pub heading_deg: f64,
    /// Offsets from `heading_deg`, degrees.
    pub heading_offsets_deg: Vec<f64>,
    /// Angles of the optical axis below the horizon, degrees.
    pub pitches_deg: Vec<f64>,
    /// Camera heights above the target ground, metres.
    pub heights_m: Vec<f64>,
}

impl LookAt {
    /// The grid that located the Oryol photo: ±12° heading, 35° to 55° down,
    /// 100 m to 220 m above ground.
    pub fn standard(heading_deg: f64) -> Self {
        Self {
            heading_deg,
            heading_offsets_deg: vec![-12.0, 0.0, 12.0],
            pitches_deg: vec![35.0, 45.0, 55.0],
            heights_m: vec![100.0, 150.0, 220.0],
        }
    }

    /// Poses aimed at `target`, a local position on the ground.
    pub fn poses(&self, target: Vector3<f64>) -> Vec<Candidate> {
        let mut out = Vec::new();
        for offset in &self.heading_offsets_deg {
            let heading = self.heading_deg + offset;
            for &pitch in &self.pitches_deg {
                for &height in &self.heights_m {
                    out.push(Candidate {
                        heading_deg: heading,
                        pitch_deg: pitch,
                        height_m: height,
                        pose: look_at(target, heading, pitch, height),
                    });
                }
            }
        }
        out
    }
}

/// One sampled pose and the values that made it.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    /// Compass heading, degrees.
    pub heading_deg: f64,
    /// Angle below the horizon, degrees.
    pub pitch_deg: f64,
    /// Height above the target ground, metres.
    pub height_m: f64,
    /// Camera pose in the local frame.
    pub pose: CameraPose,
}

/// The camera `height_m` above `target` that looks at it along `heading` and `pitch`.
pub fn look_at(
    target: Vector3<f64>,
    heading_deg: f64,
    pitch_deg: f64,
    height_m: f64,
) -> CameraPose {
    let orientation = orientation(heading_deg, pitch_deg, 0.0);
    let (sh, ch) = heading_deg.to_radians().sin_cos();
    let back = height_m / pitch_deg.to_radians().tan().max(1e-6);
    CameraPose {
        position: Vector3::new(
            target.x - sh * back,
            target.y - ch * back,
            target.z + height_m,
        ),
        orientation,
    }
}

/// Eye-to-world rotation for a heading, a downward pitch and a roll, all in degrees.
///
/// World axes are east, north, up. Eye axes are right, up and back.
pub fn orientation(heading_deg: f64, pitch_deg: f64, roll_deg: f64) -> UnitQuaternion<f64> {
    let (sh, ch) = heading_deg.to_radians().sin_cos();
    let (sp, cp) = pitch_deg.to_radians().sin_cos();
    let forward = Vector3::new(sh * cp, ch * cp, -sp);
    let right = Vector3::new(ch, -sh, 0.0);
    let back = -forward;
    let up = back.cross(&right);
    let level = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(
        Matrix3::from_columns(&[right, up, back]),
    ));
    level * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), roll_deg.to_radians())
}

/// Heading, downward pitch and roll, in degrees, of an eye-to-world rotation.
pub fn angles(orientation: &UnitQuaternion<f64>) -> [f64; 3] {
    let forward = orientation * -Vector3::z();
    let right = orientation * Vector3::x();
    let heading = forward.x.atan2(forward.y).to_degrees().rem_euclid(360.0);
    let pitch = (-forward.z).clamp(-1.0, 1.0).asin().to_degrees();
    let level_right = Vector3::new(forward.y, -forward.x, 0.0).normalize();
    let roll = -level_right
        .cross(&right)
        .dot(&forward)
        .atan2(level_right.dot(&right))
        .to_degrees();
    [heading, pitch, roll]
}

#[cfg(test)]
mod tests;
