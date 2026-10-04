//! Frame identity, media time, and capture time.

use navigate_contract::ClockDomainId;

/// One camera or media source. The host assigns a different value to each source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId(pub u32);

/// A run of frames with continuous motion.
///
/// The host starts a new epoch after a seek, a restart, a decoder reset, or a
/// gap that tracking cannot bridge. Frames in different epochs have no known
/// relative motion until a map anchor or a revisit closure joins them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContinuityEpoch(pub u32);

/// The identity of one frame in a session.
///
/// `index` increases inside one epoch. Its value has no meaning across epochs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameKey {
    /// Source of the frame.
    pub stream: StreamId,
    /// Continuity epoch inside the source.
    pub continuity: ContinuityEpoch,
    /// Frame order inside the epoch.
    pub index: u64,
}

impl FrameKey {
    /// The source and epoch that contain this frame.
    pub fn track(&self) -> (StreamId, ContinuityEpoch) {
        (self.stream, self.continuity)
    }
}

/// A presentation timestamp in the time base of the media container.
///
/// The value is the media position of the frame. It is not an acquisition
/// time. Playback and the frame-to-pose binding use this value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediaTime {
    /// Presentation timestamp in time-base units.
    pub pts: i64,
    /// Time-base numerator in seconds.
    pub timebase_num: u32,
    /// Time-base denominator.
    pub timebase_den: u32,
}

impl MediaTime {
    /// The presentation time in seconds, or `None` for an invalid time base.
    pub fn seconds(&self) -> Option<f64> {
        (self.timebase_num > 0 && self.timebase_den > 0)
            .then(|| self.pts as f64 * f64::from(self.timebase_num) / f64::from(self.timebase_den))
    }
}

/// The acquisition time of a frame in one declared clock domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureTime {
    /// Clock that produced `at_ns`.
    pub clock: ClockDomainId,
    /// Monotonic acquisition time in nanoseconds.
    pub at_ns: u64,
    /// Bound on the difference between `at_ns` and the true exposure time.
    pub error_bound_ns: u64,
}

/// The identity and timing of one processed frame. Pixels stay with the host.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameRecord {
    /// Session identity of the frame.
    pub key: FrameKey,
    /// Container presentation time, when the source is a media file or stream.
    pub media: Option<MediaTime>,
    /// Acquisition time in the session clock domain.
    pub capture: CaptureTime,
    /// `navigate_visual::Frame::evidence_sha256` of the processed pixels.
    pub observation_sha256: String,
}

impl FrameRecord {
    /// Record a processed frame with its evidence digest.
    pub fn from_frame(
        key: FrameKey,
        media: Option<MediaTime>,
        capture: CaptureTime,
        frame: &navigate_visual::Frame,
    ) -> Self {
        Self {
            key,
            media,
            capture,
            observation_sha256: frame.evidence_sha256(),
        }
    }
}
