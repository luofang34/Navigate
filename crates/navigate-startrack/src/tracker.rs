//! The choice of the identification mode from frame to frame.
//!
//! An attitude prior makes identification fast and robust, but only while
//! the prior is near the truth. [`StarTracker`] uses the prior when the
//! last frames solved and the prior is narrow enough. It goes back to
//! lost-in-space after [`TrackerConfig::max_failed_frames`] frames in a
//! row without a solution, and for any frame whose prior is wider than
//! [`TrackerConfig::max_prior_sigma_rad`]. A prior that an inertial
//! navigator propagates through a long gap grows, and a prior that has
//! gone wrong makes every frame fail; either way the tracker recovers
//! without the help of the host.

use crate::{AttitudePrior, IdentifyMode, PairCatalogue, StarTrackError};

/// The settings of [`StarTracker`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackerConfig {
    /// Frames in a row without a solution after which the tracker drops
    /// the prior and uses lost-in-space.
    pub max_failed_frames: u32,
    /// Widest prior, one standard deviation in radians, that the tracker
    /// uses. Identification with a prior gates each star by the prior
    /// deviation, so a wider prior admits more wrong candidates than
    /// lost-in-space does.
    pub max_prior_sigma_rad: f64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_failed_frames: 3,
            max_prior_sigma_rad: 2.0_f64.to_radians(),
        }
    }
}

/// Whether the tracker holds a lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackerLock {
    /// No frame has solved since the start or since the lock was lost.
    Lost,
    /// A frame solved, followed by `failed_frames` frames without a
    /// solution.
    Locked {
        /// Frames without a solution since the last solved frame.
        failed_frames: u32,
    },
}

/// The identification mode of each frame from the outcome of the frames
/// before it. The state is a function of the recorded outcomes only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StarTracker {
    config: TrackerConfig,
    lock: TrackerLock,
}

impl StarTracker {
    /// A tracker without a lock.
    ///
    /// # Errors
    /// Refuses a prior limit that is not finite and positive.
    pub fn new(config: TrackerConfig) -> Result<Self, StarTrackError> {
        if !(config.max_prior_sigma_rad.is_finite() && config.max_prior_sigma_rad > 0.0) {
            return Err(StarTrackError::Invalid {
                field: "tracker prior limit",
            });
        }
        Ok(Self {
            config,
            lock: TrackerLock::Lost,
        })
    }

    /// The lock state.
    #[must_use]
    pub const fn lock(&self) -> TrackerLock {
        self.lock
    }

    /// The identification mode of the next frame. `prior` is the attitude
    /// of the host for the acquisition time of that frame, if it has one.
    #[must_use]
    pub fn mode<'a>(
        &self,
        prior: Option<AttitudePrior>,
        table: &'a PairCatalogue,
    ) -> IdentifyMode<'a> {
        let usable = |p: &AttitudePrior| {
            p.sigma_rad.is_finite()
                && p.sigma_rad > 0.0
                && p.sigma_rad <= self.config.max_prior_sigma_rad
        };
        match (self.lock, prior.filter(usable)) {
            (TrackerLock::Locked { .. }, Some(prior)) => IdentifyMode::Prior(prior),
            _ => IdentifyMode::LostInSpace(table),
        }
    }

    /// Records whether the frame solved.
    pub fn record(&mut self, solved: bool) {
        self.lock = match (solved, self.lock) {
            (true, _) => TrackerLock::Locked { failed_frames: 0 },
            (false, TrackerLock::Lost) => TrackerLock::Lost,
            (false, TrackerLock::Locked { failed_frames }) => {
                let failed_frames = failed_frames.saturating_add(1);
                if failed_frames >= self.config.max_failed_frames {
                    TrackerLock::Lost
                } else {
                    TrackerLock::Locked { failed_frames }
                }
            }
        };
    }
}

#[cfg(test)]
mod tests;
