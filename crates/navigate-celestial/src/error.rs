/// A failed observation calculation. No fallback direction is returned.
#[derive(Debug, thiserror::Error)]
pub enum ObservationError<E: std::error::Error + 'static> {
    /// A state has invalid components or a physically invalid velocity.
    #[error("invalid {role} state: {reason}")]
    InvalidState {
        /// The state that failed validation.
        role: &'static str,
        /// The failed invariant.
        reason: &'static str,
    },
    /// A provider failed to evaluate or transform a state.
    #[error("cannot evaluate {role} for target {target}")]
    Ephemeris {
        /// The operation that failed.
        role: &'static str,
        /// The requested NAIF body identifier.
        target: i32,
        /// The original provider error.
        #[source]
        source: E,
    },
    /// The provider returned a state at another epoch or in another frame.
    #[error("{role} provider state does not match the requested epoch or SSB/J2000 frame")]
    ProviderContract {
        /// The provider result that failed validation.
        role: &'static str,
    },
    /// There is no finite, nonzero line of sight.
    #[error("target {target} has no finite nonzero range from the observer")]
    Range {
        /// The requested NAIF body identifier.
        target: i32,
    },
    /// The light-time iteration failed its convergence condition.
    #[error("reception light time did not converge for target {target}")]
    LightTime {
        /// The requested NAIF body identifier.
        target: i32,
    },
}
