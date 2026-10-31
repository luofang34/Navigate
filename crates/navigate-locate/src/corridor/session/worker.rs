//! Tracking state for an asynchronous map-anchor worker.

use nalgebra::Matrix3;

/// Tracking state for a map-anchor worker.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnchorWorkerState {
    /// Previous accepted camera position in `ENU` meters.
    pub prev_eye: Option<[f64; 3]>,
    /// Previous accepted camera rotation from world `ENU` to camera coordinates.
    pub prev_rot: Option<Matrix3<f64>>,
    /// Current translational velocity in meters per second.
    pub vel_enu_mps: [f64; 3],
}

impl AnchorWorkerState {
    /// Clear the stored pose and velocity after track loss.
    pub fn on_track_loss(&mut self) {
        self.prev_eye = None;
        self.prev_rot = None;
        self.vel_enu_mps = [0.0, 0.0, 0.0];
    }

    /// Record an accepted anchor pose and update velocity across `dt_s`.
    pub fn on_anchor_accepted(&mut self, new_eye: [f64; 3], new_rot: Matrix3<f64>, dt_s: f64) {
        if let Some(prev) = self.prev_eye {
            let dt = dt_s.clamp(1e-4, 60.0);
            self.vel_enu_mps = [
                (new_eye[0] - prev[0]) / dt,
                (new_eye[1] - prev[1]) / dt,
                (new_eye[2] - prev[2]) / dt,
            ];
        } else {
            self.vel_enu_mps = [0.0, 0.0, 0.0];
        }
        self.prev_eye = Some(new_eye);
        self.prev_rot = Some(new_rot);
    }

    /// Record a relocated position and update velocity across `dt_s`.
    pub fn on_relocation(&mut self, relocated_eye: [f64; 3], dt_s: f64) {
        if let Some(prev) = self.prev_eye {
            let dt = dt_s.clamp(1e-4, 60.0);
            self.vel_enu_mps = [
                (relocated_eye[0] - prev[0]) / dt,
                (relocated_eye[1] - prev[1]) / dt,
                (relocated_eye[2] - prev[2]) / dt,
            ];
        }
        self.prev_eye = Some(relocated_eye);
    }
}
