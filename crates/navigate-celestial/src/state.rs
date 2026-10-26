use crate::Epoch;

/// Coordinate origin and orientation, in the NAIF identifier system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameId {
    /// Body or system at the coordinate origin. Zero is the solar system barycentre.
    pub center: i32,
    /// Axis orientation. One is J2000. Body-fixed axes require a rotation model.
    pub orientation: i32,
}

impl FrameId {
    /// Solar system barycentre, with J2000 axes.
    pub const SSB_J2000: Self = Self {
        center: 0,
        orientation: 1,
    };

    /// Earth centre, with Earth-fixed ITRF93 axes (NAIF frame 13000).
    pub const EARTH_ITRF93: Self = Self {
        center: 399,
        orientation: 13000,
    };
}

/// A geometric Cartesian state. The velocity is the derivative in `frame`.
///
/// A body-fixed state with zero velocity moves in inertial space. A frame
/// adapter must include the rotation derivative when it converts velocity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    /// Position in metres from the frame origin.
    pub position_m: [f64; 3],
    /// Coordinate velocity in metres per TDB second.
    pub velocity_m_s: [f64; 3],
    /// Event time, with an explicit time scale.
    pub epoch: Epoch,
    /// Origin and axes of both vectors.
    pub frame: FrameId,
}

/// Geometric states and frame conversion for an observation calculation.
///
/// Implementations must not apply light-time or aberration corrections.
/// Missing data must return an error. This interface does not fetch data.
pub trait Ephemeris {
    /// The source error from this ephemeris implementation.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Convert position and velocity to SSB/J2000 at the state's epoch.
    fn barycentric_state(&self, state: State) -> Result<State, Self::Error>;

    /// Return a body's geometric SSB/J2000 state at the requested epoch.
    fn body_state(&self, body: i32, epoch: Epoch) -> Result<State, Self::Error>;
}
