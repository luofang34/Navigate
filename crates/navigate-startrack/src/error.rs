//! Errors of the star tracker.

use navigate_camera::CameraError;
use navigate_celestial::AnalyticError;

/// A star image, a configuration, or a star set that the tracker cannot use.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StarTrackError {
    /// The camera model or the image is not valid.
    #[error("camera model or image: {0}")]
    Camera(#[from] CameraError),
    /// The catalogue epoch is outside the range of the ephemeris.
    #[error("star catalogue: {0}")]
    Catalogue(#[from] AnalyticError),
    /// A configuration or an input value is outside its valid range.
    #[error("invalid star tracker input: {field}")]
    Invalid {
        /// The name of the value.
        field: &'static str,
    },
    /// Identification needs an attitude prior. Identification without a
    /// prior (lost-in-space) is not supported.
    #[error("lost-in-space identification without an attitude prior is not supported")]
    LostInSpaceUnsupported,
    /// The image has more pixel groups above the threshold than the limit.
    /// A bright object, such as the Sun, the Moon, or a lit sky, can cause
    /// this.
    #[error("the image has more than {limit} pixel groups above the threshold")]
    TooManyComponents {
        /// The configured limit.
        limit: usize,
    },
    /// The attitude solution needs more matched stars.
    #[error("{found} matched stars; the attitude solution needs {needed}")]
    TooFewStars {
        /// Matched stars.
        found: usize,
        /// Stars that the solution needs.
        needed: usize,
    },
    /// The matched directions do not define the attitude about all three
    /// axes, for example because all stars lie on one great circle.
    #[error("the matched star directions do not define the attitude")]
    Degenerate,
}
