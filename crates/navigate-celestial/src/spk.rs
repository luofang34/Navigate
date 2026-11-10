//! Binary NAIF `DAF/SPK` Chebyshev ephemeris provider.
//!
//! [`SpkFile`] reads Type 2 and Type 3 Chebyshev segments from a binary `.bsp`
//! slice in memory without file I/O. [`SpkEphemeris`] chains those segments to
//! the solar-system barycentre (`SSB`, NAIF ID `0`) and implements
//! [`Ephemeris`].

pub use navigate_astrodynamics::spk::{
    DafEndian, SpkDataType, SpkEphemeris, SpkError, SpkFile, SpkSegment,
};

use nalgebra::Vector3;

use crate::{Ephemeris, Epoch, FrameId, State, TimeScale, earth_rotation};

const SSB: i32 = 0;
const EARTH: i32 = 399;
const J2000_FRAME: i32 = 1;
const ITRF93_FRAME: i32 = FrameId::EARTH_ITRF93.orientation_naif_id;

impl Ephemeris for SpkEphemeris {
    type Error = SpkError;

    fn barycentric_state(&self, state: State) -> Result<State, Self::Error> {
        let pos = Vector3::from(state.position_m);
        let vel = Vector3::from(state.velocity_m_s);
        let (rel_p, rel_v) = match (state.frame.center_naif_id, state.frame.orientation_naif_id) {
            (_, J2000_FRAME) => (pos, vel),
            (EARTH, ITRF93_FRAME) => {
                let to_j2000 = earth_rotation::j2000_to_earth_fixed(state.epoch).transpose();
                let v_fixed = vel + earth_rotation::rotation_velocity(pos);
                (to_j2000 * pos, to_j2000 * v_fixed)
            }
            (center, orientation) => {
                return Err(SpkError::UnsupportedObserverFrame {
                    center,
                    orientation,
                });
            }
        };
        let tdb_s = state.epoch.to_time_scale(TimeScale::TDB).to_tdb_seconds();
        let (origin_p, origin_v) =
            self.relative_state_tdb_j2000(state.frame.center_naif_id, SSB, tdb_s)?;
        Ok(State {
            position_m: (origin_p + rel_p).into(),
            velocity_m_s: (origin_v + rel_v).into(),
            epoch: state.epoch,
            frame: FrameId::SSB_J2000,
        })
    }

    fn body_state(&self, body: i32, epoch: Epoch) -> Result<State, Self::Error> {
        let tdb_s = epoch.to_time_scale(TimeScale::TDB).to_tdb_seconds();
        let (pos, vel) = self.relative_state_tdb_j2000(body, SSB, tdb_s)?;
        Ok(State {
            position_m: pos.into(),
            velocity_m_s: vel.into(),
            epoch,
            frame: FrameId::SSB_J2000,
        })
    }
}

#[cfg(test)]
mod tests;
