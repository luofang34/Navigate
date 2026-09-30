//! Host resource grants and bounded visual work.
use crate::FrameStamp;
use std::time::Duration;

/// A host-owned execution device. Several profiles can share one device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceId(pub u64);

/// A configured matcher, image shape, and execution backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProfileId(pub u64);

/// Distinct cost histories for distinct visual work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkKind {
    /// Check a local geographic candidate against the observation.
    MapCheck,
    /// Search alternatives after acquisition or tracking failure.
    Recovery,
    /// Optional relative visual motion estimation.
    Tracking,
}

/// A profile whose output quality the host has validated for this work.
#[derive(Clone, Copy, Debug)]
pub struct ExecutionProfile {
    /// Stable identity of this configured execution path.
    pub id: ProfileId,
    /// Device required by this path.
    pub device: DeviceId,
    /// Work for which this profile is valid.
    pub work: WorkKind,
    /// Conservative initial whole-work cost, including preparation and transfers.
    pub initial_cost: Duration,
    /// Conservative initial longest call that cannot be interrupted.
    pub initial_call: Duration,
    /// Peak memory needed by this profile, including required resident models.
    pub peak_bytes: u64,
}

/// A grant from the flight scheduler, not a claim that hardware is idle.
#[derive(Clone, Copy, Debug)]
pub struct ResourceGrant {
    /// Device which the host permits VPS to use.
    pub device: DeviceId,
    /// End of this grant in the controller's monotonic clock domain.
    pub until: Duration,
    /// Maximum allowed duration of one call that cannot be interrupted.
    pub maximum_call: Duration,
    /// Memory available to this profile.
    pub memory_bytes: u64,
}

/// A request from the navigation and geographic-search policy.
#[derive(Clone, Copy, Debug)]
pub struct WorkDemand {
    /// Work whose results are useful now.
    pub work: WorkKind,
    /// Required completion time from predicted uncertainty or verification age.
    /// None means the bound is unknown, so the check is due now.
    pub due_by: Option<Duration>,
}

/// Limits that apply independently of model scores and geometry thresholds.
#[derive(Clone, Copy, Debug)]
pub struct ControllerConfig {
    /// Maximum age of a frame when work starts.
    pub maximum_capture_age: Duration,
    /// Maximum useful age of a completed result.
    pub maximum_result_age: Duration,
    /// Time reserved beyond the measured cost estimate.
    pub reserve: Duration,
    /// Minimum delay between starts, including failed attempts.
    pub minimum_interval: Duration,
}

/// Identity of one admitted unit of work. Completion requires the same ticket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkTicket {
    pub(super) generation: u64,
    pub(super) profile: ProfileId,
    pub(super) observation: FrameStamp,
}
impl WorkTicket {
    /// The execution profile selected by the controller.
    pub fn profile(&self) -> ProfileId {
        self.profile
    }
    /// The capture identity retained through asynchronous execution.
    pub fn observation(&self) -> FrameStamp {
        self.observation
    }
}

/// The scheduler's decision. It never accepts a geographic pose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// Start bounded work using the named profile.
    Start {
        /// Identity used to account for completion.
        ticket: WorkTicket,
        /// Whether predicted completion exceeds a known due time. None means unknown.
        late: Option<bool>,
    },
    /// Another unit of visual work is active.
    Busy,
    /// A newer frame is needed before work can start.
    StaleObservation,
    /// Wait until this monotonic time unless the demand changes.
    NotDue(Duration),
    /// No validated profile fits the grants, memory, or useful result age.
    NoResources,
}

/// Measured cost of completed work, including rejected matches and failures.
#[derive(Clone, Copy, Debug)]
pub struct WorkCost {
    /// Elapsed time from admission through preprocessing, execution and geometry.
    pub total: Duration,
    /// Longest measured call that could not be interrupted.
    pub longest_call: Duration,
}
