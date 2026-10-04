//! Session errors.

use crate::FrameKey;

/// A session input violates its contract. A refused input changes no state.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// A configuration or calibration value is not usable.
    #[error("invalid session input: {field}")]
    Invalid {
        /// The field that failed validation.
        field: &'static str,
    },
    /// The frame index or capture time did not increase inside one epoch.
    #[error("frame {received:?} does not follow frame {previous:?}")]
    FrameOrder {
        /// Last frame of the epoch.
        previous: FrameKey,
        /// Refused frame.
        received: FrameKey,
    },
    /// The frame is not in the session or was removed by the memory bound.
    #[error("frame {0:?} is not available in the session")]
    UnknownFrame(FrameKey),
    /// The frame uses a clock that differs from the session clock.
    #[error("frame {0:?} uses a clock domain that differs from the session clock")]
    ClockDomain(FrameKey),
    /// Motion must start at the newest frame of an epoch and end at the next frame.
    #[error("motion from {from:?} to {to:?} does not extend the epoch head")]
    MotionOrder {
        /// Start frame of the motion.
        from: FrameKey,
        /// End frame of the motion.
        to: FrameKey,
    },
    /// The motion has no metric scale. The session does not integrate it.
    #[error("motion from {from:?} to {to:?} has no metric scale")]
    ScaleUnobservable {
        /// Start frame of the motion.
        from: FrameKey,
        /// End frame of the motion.
        to: FrameKey,
    },
    /// The frame has no odometry, and later frames of its epoch already do.
    #[error("frame {0:?} has no odometry and is older than its segment head")]
    NoOdometry(FrameKey),
    /// The evidence was already used in this session.
    #[error("observation {0} was already used")]
    RepeatedEvidence(String),
    /// The evidence digest does not match the frame record.
    #[error("evidence digest does not match frame {0:?}")]
    EvidenceMismatch(FrameKey),
    /// A visual geometry check failed.
    #[error("visual geometry for frame {frame:?}: {source}")]
    Visual {
        /// Frame of the check.
        frame: FrameKey,
        /// Geometry error.
        #[source]
        source: navigate_visual::VisualError,
    },
}
