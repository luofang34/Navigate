//! The confirmation of an acquisition by the next one.
//!
//! The belief of a later frame is the belief of the earlier one carried
//! forward by the inertial solution, so two acquisitions of one place
//! correct their beliefs alike: the corrections differ by the errors of
//! the two fixes and by what the inertial solution added to the error of
//! the belief between them. Each fix comes from its own frame and its own
//! reference view.

use nalgebra::Vector2;

use super::super::config::TerrainConfig;

/// An acquisition fix that waits for the next one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in super::super) struct Candidate {
    pub at_ns: u64,
    /// East and north from the belief to the fix, metres.
    pub correction_m: Vector2<f64>,
    /// Horizontal variance of the fix, the sum of its east and north
    /// variances, square metres.
    pub fix_variance_m2: f64,
    /// Horizontal one-sigma of the belief, the root of the sum of its east
    /// and north variances, metres.
    pub belief_sigma_m: f64,
}

impl Candidate {
    /// Whether `next`, later and at most the confirmation time after this
    /// one, corrects its belief as this one did: within the search
    /// multiple of the one-sigma of the difference.
    pub(in super::super) fn confirmed_by(&self, next: &Self, config: &TerrainConfig) -> bool {
        let age_s = next.at_ns.saturating_sub(self.at_ns) as f64 * 1e-9;
        // What the inertial solution added to the horizontal variance of
        // the belief over the interval.
        let carried_m2 = (next.belief_sigma_m.powi(2) - self.belief_sigma_m.powi(2)).max(0.0);
        let sigma_m = (self.fix_variance_m2 + next.fix_variance_m2 + carried_m2).sqrt();
        next.at_ns > self.at_ns
            && age_s <= config.acquisition_confirm_s
            && (next.correction_m - self.correction_m).norm()
                <= config.search_sigma_multiple * sigma_m
    }
}
