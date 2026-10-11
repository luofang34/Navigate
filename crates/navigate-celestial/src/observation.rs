use crate::{Duration, Ephemeris, Epoch, FrameId, ObservationError, State, TimeScale};
use nalgebra::Vector3 as Vector;

type Vector3 = Vector<f64>;

const LIGHT_SPEED_M_S: f64 = 299_792_458.0;

/// The physical model used for the returned direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Correction {
    /// Simultaneous geometric states. No propagation or aberration correction.
    Geometric,
    /// Converged Euclidean reception light time and special-relativistic aberration.
    ///
    /// This model excludes gravitational light deflection, gravitational time
    /// delay, refraction, and instrument effects. It does not calibrate clocks.
    VacuumReception,
}

/// A finite-distance body as evaluated for one observer event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sight {
    /// NAIF target identifier.
    pub target: i32,
    /// Observer event time in TDB. This is not a remote packet arrival time.
    pub received_at: Epoch,
    /// Target evaluation time. Equals `received_at` for geometric states.
    pub target_epoch: Epoch,
    /// Direction in J2000-aligned axes. In reception mode the axes belong to
    /// the observer's instantaneous rest frame, reached by a pure boost.
    pub direction: [f64; 3],
    /// Euclidean separation of the evaluated target and observer events, in metres.
    /// This is not a radar range or a relativistic proper distance.
    pub coordinate_range_m: f64,
    /// Reception propagation time; absent for a simultaneous geometric query.
    pub light_time: Option<Duration>,
    /// Corrections applied exactly once by this calculation.
    pub correction: Correction,
}

/// Calculate a body direction from an observer state at reception.
///
/// The observer's velocity is used for aberration. The provider evaluates the
/// target at emission time. The observer is not moved back to emission time.
///
/// # Errors
/// Returns a typed error for invalid states, missing data, coincident bodies,
/// a provider frame or epoch mismatch, or failed light-time convergence.
pub fn observe_body<P: Ephemeris>(
    provider: &P,
    mut observer: State,
    target: i32,
    correction: Correction,
) -> Result<Sight, ObservationError<P::Error>> {
    validate(observer, "observer")?;
    // Propagation uses continuous ephemeris time, never UTC calendar arithmetic.
    observer.epoch = observer.epoch.to_time_scale(TimeScale::TDB);
    let received_at = observer.epoch;
    let observer =
        provider
            .barycentric_state(observer)
            .map_err(|source| ObservationError::Ephemeris {
                role: "observer",
                target,
                source,
            })?;
    validate_provider(observer, received_at, "observer")?;
    let position = Vector3::from(observer.position_m);
    let mut target_epoch = received_at;
    for _ in 0..16 {
        let state = provider
            .body_state(target, target_epoch)
            .map_err(|source| ObservationError::Ephemeris {
                role: "target",
                target,
                source,
            })?;
        validate_provider(state, target_epoch, "target")?;
        let ray = Vector3::from(state.position_m) - position;
        let range = ray.norm();
        if !range.is_finite() || range <= 0.0 {
            return Err(ObservationError::Range { target });
        }
        let light_time = Duration::from_seconds(range / LIGHT_SPEED_M_S);
        let emission = received_at - light_time;
        if correction == Correction::Geometric
            || (emission - target_epoch).abs() <= Duration::from_total_nanoseconds(1)
        {
            let direction = if correction == Correction::Geometric {
                ray / range
            } else {
                aberrate(
                    ray / range,
                    Vector3::from(observer.velocity_m_s) / LIGHT_SPEED_M_S,
                )
            };
            return Ok(Sight {
                target,
                received_at,
                target_epoch,
                direction: direction.into(),
                coordinate_range_m: range,
                correction,
                light_time: (correction == Correction::VacuumReception)
                    .then_some(received_at - target_epoch),
            });
        }
        target_epoch = emission;
    }
    Err(ObservationError::LightTime { target })
}

fn aberrate(direction: Vector3, beta: Vector3) -> Vector3 {
    // Lorentz-transform an incoming photon; direction points towards the source.
    let inverse_gamma = (1.0 - beta.norm_squared()).sqrt();
    let dot = direction.dot(&beta);
    ((direction * inverse_gamma + beta * (1.0 + dot / (1.0 + inverse_gamma))) / (1.0 + dot))
        .normalize()
}

fn validate<E: std::error::Error + 'static>(
    state: State,
    role: &'static str,
) -> Result<(), ObservationError<E>> {
    if !state
        .position_m
        .iter()
        .chain(&state.velocity_m_s)
        .all(|x| x.is_finite())
    {
        return Err(ObservationError::InvalidState {
            role,
            reason: "non-finite position or velocity",
        });
    }
    Ok(())
}

fn validate_provider<E: std::error::Error + 'static>(
    state: State,
    epoch: Epoch,
    role: &'static str,
) -> Result<(), ObservationError<E>> {
    validate(state, role)?;
    if state.frame != FrameId::SSB_J2000 || state.epoch != epoch {
        return Err(ObservationError::ProviderContract { role });
    }
    if Vector3::from(state.velocity_m_s).norm() >= LIGHT_SPEED_M_S {
        return Err(ObservationError::InvalidState {
            role,
            reason: "barycentric speed must be below light speed",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
