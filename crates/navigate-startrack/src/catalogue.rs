//! The reference star directions.

use navigate_celestial::{Epoch, StarSight, stars_ecef_tt};

use crate::StarTrackError;

/// The catalogue stars at least as bright as `faintest_v_magnitude`, in
/// ECEF axes at the instant `epoch`, brightest first.
///
/// `epoch` is the acquisition instant in any time scale, as for
/// [`navigate_celestial::stars_ecef_tt`], which gives the model and its
/// accuracy. The directions include annual aberration.
///
/// # Errors
/// Refuses a magnitude limit that is not finite, and returns
/// [`StarTrackError::Catalogue`] for an epoch outside the ephemeris range.
pub fn catalogue_ecef(
    epoch: Epoch,
    faintest_v_magnitude: f64,
) -> Result<Vec<StarSight>, StarTrackError> {
    if !faintest_v_magnitude.is_finite() {
        return Err(StarTrackError::Invalid {
            field: "faintest visual magnitude",
        });
    }
    Ok(stars_ecef_tt(epoch, faintest_v_magnitude)?)
}
