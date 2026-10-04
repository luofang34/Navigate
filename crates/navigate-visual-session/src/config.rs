//! Session limits and declared error models.

use crate::SessionError;

/// The declared growth of relative-motion error.
///
/// These values are host declarations, not measurements. Tracking results
/// have no covariance. The session uses this model to weight motion and to
/// bound the position uncertainty between map anchors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriftModel {
    /// Position error per metre of camera travel.
    pub translation_fraction: f64,
    /// Position error added by each motion step, in metres.
    pub step_floor_m: f64,
    /// Rotation error per radian of camera rotation.
    pub rotation_fraction: f64,
    /// Rotation error per metre of camera travel, in radians.
    pub rotation_per_m_rad: f64,
    /// Rotation error added by each motion step, in radians.
    pub step_floor_rad: f64,
}

impl Default for DriftModel {
    fn default() -> Self {
        Self {
            translation_fraction: 0.05,
            step_floor_m: 0.05,
            rotation_fraction: 0.05,
            rotation_per_m_rad: 0.000_5,
            step_floor_rad: 0.001,
        }
    }
}

/// When the session adds a keyframe to the revisit database and the pose graph.
///
/// Only camera motion adds a keyframe. A hovering camera keeps one keyframe,
/// so repeated anchors from one viewpoint replace each other and do not add
/// evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyframePolicy {
    /// Camera travel since the last keyframe, in metres.
    pub translation_m: f64,
    /// Camera rotation since the last keyframe, in radians.
    pub rotation_rad: f64,
}

impl Default for KeyframePolicy {
    fn default() -> Self {
        Self {
            translation_m: 15.0,
            rotation_rad: 0.26,
        }
    }
}

/// Map-anchor admission policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnchorPolicy {
    /// Gate width in standard deviations of the combined uncertainty.
    pub gate_sigma: f64,
    /// Largest position bound at which one consistent anchor is accepted.
    /// Above this bound, accepted anchors need agreement from other frames.
    pub direct_bound_m: f64,
    /// Number of mutually consistent anchors that relocate a track.
    pub consensus_anchors: usize,
    /// Smallest camera travel between anchors that count as separate evidence.
    pub consensus_separation_m: f64,
    /// Largest number of anchors that wait for agreement in one track.
    pub max_pending: usize,
    /// Side length of the map cells that share one map-error bias, in metres.
    pub shared_error_cell_m: f64,
}

impl Default for AnchorPolicy {
    fn default() -> Self {
        Self {
            gate_sigma: 3.0,
            direct_bound_m: 60.0,
            consensus_anchors: 2,
            consensus_separation_m: 5.0,
            max_pending: 16,
            shared_error_cell_m: 500.0,
        }
    }
}

/// Revisit search and closure policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RevisitPolicy {
    /// Keyframes adjacent in time that are never revisit candidates.
    pub min_keyframe_separation: usize,
    /// Capture time that separates a revisit from continuous tracking.
    pub min_time_separation_ns: u64,
    /// Search radius added to the uncertainty bounds, in metres.
    pub search_margin_m: f64,
    /// Largest angle between the optical axes of a revisit pair, in radians.
    pub max_view_angle_rad: f64,
    /// Largest number of candidates for one frame.
    pub max_candidates: usize,
    /// Closure position error that does not depend on distance, in metres.
    pub closure_floor_m: f64,
    /// Closure position error per metre between the two cameras.
    pub closure_fraction: f64,
    /// Closure rotation error, in radians.
    pub closure_rotation_rad: f64,
    /// Post-optimization residual, in standard deviations, that retracts a closure.
    pub retract_sigma: f64,
}

impl Default for RevisitPolicy {
    fn default() -> Self {
        Self {
            min_keyframe_separation: 6,
            min_time_separation_ns: 20_000_000_000,
            search_margin_m: 30.0,
            max_view_angle_rad: 0.8,
            max_candidates: 4,
            closure_floor_m: 0.5,
            closure_fraction: 0.02,
            closure_rotation_rad: 0.01,
            retract_sigma: 5.0,
        }
    }
}

/// Error model of ground-plane attitude observations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundPolicy {
    /// Normal error per unit of relief ratio (object height divided by the
    /// plane distance). Objects cover part of the image, so the normal moves
    /// by a fraction of the relief angle.
    pub relief_fraction: f64,
}

impl Default for GroundPolicy {
    fn default() -> Self {
        Self {
            relief_fraction: 0.25,
        }
    }
}

/// When a located frame is usable for each purpose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputPolicy {
    /// Largest capture-time distance to an anchor for recent map support, in nanoseconds.
    pub recent_anchor_ns: u64,
    /// Largest position bound for a navigation position, in metres.
    pub navigation_bound_m: f64,
    /// Largest position bound for ground projection, in metres.
    pub projection_bound_m: f64,
    /// Largest tilt bound for ground projection, in radians.
    pub projection_tilt_bound_rad: f64,
    /// Largest view angle from straight down for ground projection, in radians.
    pub projection_off_nadir_rad: f64,
    /// Largest attitude bound from map evidence for ground projection, in
    /// radians. Ground planes do not observe heading, and a heading error
    /// moves the image corners on the ground.
    pub projection_attitude_bound_rad: f64,
}

impl Default for OutputPolicy {
    fn default() -> Self {
        Self {
            recent_anchor_ns: 30_000_000_000,
            navigation_bound_m: 50.0,
            projection_bound_m: 30.0,
            projection_tilt_bound_rad: 0.087,
            projection_off_nadir_rad: 1.22,
            projection_attitude_bound_rad: 0.087,
        }
    }
}

/// Map relocation proposals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelocationPolicy {
    /// Largest number of proposals for one frame.
    pub max_candidates: usize,
    /// Distance between proposal positions, in metres.
    pub spacing_m: f64,
    /// Largest proposal distance from the predicted position, in metres.
    pub max_radius_m: f64,
}

impl Default for RelocationPolicy {
    fn default() -> Self {
        Self {
            max_candidates: 9,
            spacing_m: 40.0,
            max_radius_m: 160.0,
        }
    }
}

/// Memory bounds of one session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SessionLimits {
    /// Frame records kept for pose queries. The oldest record is removed first.
    pub max_frames: usize,
    /// Keyframes kept in the pose graph and the revisit database.
    pub max_keyframes: usize,
    /// Revisit closures kept in the pose graph.
    pub max_closures: usize,
    /// Map-error bias cells kept in the pose graph.
    pub max_bias_cells: usize,
    /// Events kept until the host drains them.
    pub max_events: usize,
    /// Descriptor length accepted for revisit ranking.
    pub max_descriptor_len: usize,
    /// Map cells remembered for fusion independence labels.
    pub max_ledger_cells: usize,
}

impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            max_frames: 65_536,
            max_keyframes: 512,
            max_closures: 512,
            max_bias_cells: 64,
            max_events: 256,
            max_descriptor_len: 4_096,
            max_ledger_cells: 16_384,
        }
    }
}

/// Session configuration.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SessionConfig {
    /// Relative-motion error model.
    pub drift: DriftModel,
    /// Keyframe selection.
    pub keyframes: KeyframePolicy,
    /// Map-anchor admission.
    pub anchors: AnchorPolicy,
    /// Revisit search and closure.
    pub revisits: RevisitPolicy,
    /// Ground-plane attitude error model.
    pub ground: GroundPolicy,
    /// Usability of located frames.
    pub output: OutputPolicy,
    /// Map relocation proposals.
    pub relocation: RelocationPolicy,
    /// Memory bounds.
    pub limits: SessionLimits,
    /// Position bound above which a frame is reported as not located, in metres.
    pub max_located_bound_m: f64,
}

impl SessionConfig {
    /// A configuration with all default values and a 500 m located bound.
    pub fn standard() -> Self {
        Self {
            max_located_bound_m: 500.0,
            ..Self::default()
        }
    }

    /// Check that every value is finite and in range.
    ///
    /// # Errors
    /// Returns the first field that fails validation.
    pub fn validate(&self) -> Result<(), SessionError> {
        let d = &self.drift;
        let k = &self.keyframes;
        let a = &self.anchors;
        let r = &self.revisits;
        let l = &self.limits;
        let o = &self.output;
        let checks: [(&'static str, bool); 11] = [
            ("ground policy", positive(&[self.ground.relief_fraction])),
            (
                "output policy",
                positive(&[
                    o.navigation_bound_m,
                    o.projection_bound_m,
                    o.projection_tilt_bound_rad,
                    o.projection_off_nadir_rad,
                    o.projection_attitude_bound_rad,
                ]) && o.recent_anchor_ns > 0,
            ),
            (
                "relocation policy",
                positive(&[self.relocation.spacing_m, self.relocation.max_radius_m])
                    && (1..=64).contains(&self.relocation.max_candidates),
            ),
            (
                "drift model",
                positive(&[
                    d.translation_fraction,
                    d.step_floor_m,
                    d.rotation_fraction,
                    d.rotation_per_m_rad,
                    d.step_floor_rad,
                ]),
            ),
            (
                "keyframe policy",
                positive(&[k.translation_m, k.rotation_rad]),
            ),
            (
                "anchor policy",
                positive(&[
                    a.gate_sigma,
                    a.direct_bound_m,
                    a.consensus_separation_m,
                    a.shared_error_cell_m,
                ]) && a.consensus_anchors >= 2
                    && a.max_pending >= a.consensus_anchors,
            ),
            (
                "revisit policy",
                positive(&[
                    r.search_margin_m,
                    r.max_view_angle_rad,
                    r.closure_floor_m,
                    r.closure_fraction,
                    r.closure_rotation_rad,
                    r.retract_sigma,
                ]) && r.max_candidates > 0,
            ),
            (
                "session limits",
                l.max_frames >= 2
                    && l.max_keyframes >= 4
                    && l.max_closures > 0
                    && l.max_bias_cells > 0
                    && l.max_events >= 4
                    && l.max_descriptor_len > 0
                    && l.max_ledger_cells > 0,
            ),
            ("located bound", positive(&[self.max_located_bound_m])),
            ("revisit separation", r.min_keyframe_separation >= 1),
            ("keyframe limit", l.max_keyframes <= 4_096),
        ];
        match checks.iter().find(|(_, ok)| !ok) {
            Some((field, _)) => Err(SessionError::Invalid { field }),
            None => Ok(()),
        }
    }
}

fn positive(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite() && *v > 0.0)
}
