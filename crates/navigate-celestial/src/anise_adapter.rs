use anise::{
    almanac::Almanac,
    constants::frames::SSB_J2000,
    errors::AlmanacError,
    math::{Vector3, cartesian::CartesianState},
    prelude::Frame,
};

use crate::{Ephemeris, Epoch, FrameId, State};

/// A frame mapping or ANISE evaluation failure.
#[derive(Debug, thiserror::Error)]
pub enum AniseError {
    /// A public NAIF frame identifier has no ANISE orientation mapping.
    #[error("no ANISE orientation mapping for NAIF frame {0}")]
    UnknownOrientation(i32),
    /// ANISE could not evaluate the supplied data.
    #[error("ANISE state evaluation failed")]
    Evaluation(#[source] Box<AlmanacError>),
}

/// Borrowed ANISE data and explicit orientation identifier mappings.
///
/// NAIF frame IDs and ANISE orientation IDs are different namespaces.
/// For example, NAIF ITRF93 is 13000 and ANISE ITRF93 is 3000.
pub struct AniseEphemeris<'a> {
    almanac: &'a Almanac,
    orientations: &'a [(i32, i32)],
}

impl<'a> AniseEphemeris<'a> {
    /// Use host-loaded kernels and `(NAIF frame ID, ANISE orientation ID)` pairs.
    /// J2000 is built in. Each other orientation must have an explicit mapping.
    pub fn new(almanac: &'a Almanac, orientations: &'a [(i32, i32)]) -> Self {
        Self {
            almanac,
            orientations,
        }
    }

    fn orientation(&self, frame: i32) -> Result<i32, AniseError> {
        if frame == 1 {
            return Ok(1);
        }
        self.orientations
            .iter()
            .find(|(id, _)| *id == frame)
            .map(|(_, orientation)| *orientation)
            .ok_or(AniseError::UnknownOrientation(frame))
    }
}

impl Ephemeris for AniseEphemeris<'_> {
    type Error = AniseError;

    fn barycentric_state(&self, state: State) -> Result<State, Self::Error> {
        let input = CartesianState {
            radius_km: Vector3::from(state.position_m) / 1000.0,
            velocity_km_s: Vector3::from(state.velocity_m_s) / 1000.0,
            epoch: state.epoch,
            frame: Frame::new(
                state.frame.center,
                self.orientation(state.frame.orientation)?,
            ),
        };
        self.almanac
            .transform_to(input, SSB_J2000, None)
            .map(convert)
            .map_err(|source| AniseError::Evaluation(Box::new(source)))
    }

    fn body_state(&self, body: i32, epoch: Epoch) -> Result<State, Self::Error> {
        self.almanac
            .transform(Frame::from_ephem_j2000(body), SSB_J2000, epoch, None)
            .map(convert)
            .map_err(|source| AniseError::Evaluation(Box::new(source)))
    }
}

fn convert(state: CartesianState) -> State {
    State {
        position_m: (state.radius_km * 1000.0).into(),
        velocity_m_s: (state.velocity_km_s * 1000.0).into(),
        epoch: state.epoch,
        frame: FrameId::SSB_J2000,
    }
}

#[cfg(test)]
mod tests;
