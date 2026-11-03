//! Reference frames and reference ellipsoids (ADR-0013).
//!
//! A [`FrameId`] names a frame with two NAIF integer codes: the body at
//! the frame origin and the frame orientation. An [`Ellipsoid`] names
//! the body-fixed frame of its surface and gives the two defining
//! parameters. This module holds data only. The conversion math is in
//! `navigate-geodesy`.

/// NAIF integer codes that the preset frames use.
pub mod naif {
    /// Solar system barycenter.
    pub const SSB: i32 = 0;
    /// The Sun.
    pub const SUN: i32 = 10;
    /// The Moon.
    pub const MOON: i32 = 301;
    /// The Earth.
    pub const EARTH: i32 = 399;
    /// Mars.
    pub const MARS: i32 = 499;
    /// The inertial J2000 orientation.
    pub const J2000: i32 = 1;
    /// The ITRF93 Earth body-fixed orientation.
    pub const ITRF93: i32 = 13_000;
    /// The lunar principal axis orientation, a dynamical frame.
    pub const MOON_PA: i32 = 31_000;
    /// The lunar mean Earth / polar axis orientation, the cartographic
    /// frame.
    pub const MOON_ME: i32 = 31_001;
    /// The IAU Mars body-fixed orientation.
    pub const IAU_MARS: i32 = 10_014;
}

/// One reference frame: the NAIF code of the body at the origin and the
/// NAIF code of the orientation.
///
/// Two frames are equal only when both codes are equal. Thus `MOON_ME`
/// and `MOON_PA` are different frames, although both have the Moon at
/// the origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameId {
    /// NAIF code of the body or barycenter at the frame origin.
    pub center_naif_id: i32,
    /// NAIF code of the frame orientation.
    pub orientation_naif_id: i32,
}

impl FrameId {
    /// Solar system barycenter, J2000 axes.
    pub const SSB_J2000: Self = Self::new(naif::SSB, naif::J2000);
    /// Sun center, J2000 axes.
    pub const SUN_J2000: Self = Self::new(naif::SUN, naif::J2000);
    /// Earth center, J2000 axes.
    pub const EARTH_J2000: Self = Self::new(naif::EARTH, naif::J2000);
    /// Earth center, ITRF93 body-fixed axes.
    pub const EARTH_ITRF93: Self = Self::new(naif::EARTH, naif::ITRF93);
    /// Moon center, J2000 axes.
    pub const MOON_J2000: Self = Self::new(naif::MOON, naif::J2000);
    /// Moon center, lunar principal axis body-fixed axes.
    pub const MOON_PA: Self = Self::new(naif::MOON, naif::MOON_PA);
    /// Moon center, lunar mean Earth / polar axis body-fixed axes.
    pub const MOON_ME: Self = Self::new(naif::MOON, naif::MOON_ME);
    /// Mars center, J2000 axes.
    pub const MARS_J2000: Self = Self::new(naif::MARS, naif::J2000);
    /// Mars center, IAU Mars body-fixed axes.
    pub const MARS_IAU: Self = Self::new(naif::MARS, naif::IAU_MARS);

    /// Builds a frame from its two NAIF codes.
    #[must_use]
    pub const fn new(center_naif_id: i32, orientation_naif_id: i32) -> Self {
        Self {
            center_naif_id,
            orientation_naif_id,
        }
    }

    /// Whether the orientation is the inertial J2000 orientation.
    #[must_use]
    pub const fn is_inertial(&self) -> bool {
        self.orientation_naif_id == naif::J2000
    }
}

/// A reference ellipsoid of revolution, fixed to one body frame.
///
/// The ellipsoid defines the geodetic coordinates of a
/// [`crate::GeodeticPosition`]: the planetographic latitude along the
/// ellipsoidal normal in `[-π/2, π/2]`, the east-positive longitude in
/// `[-π, π]`, and the height above the ellipsoid in meters. A flattening
/// of 0 gives a sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Ellipsoid {
    /// The body-fixed frame of the ellipsoid surface.
    pub frame: FrameId,
    /// Equatorial radius in meters.
    pub semi_major_axis_m: f64,
    /// Flattening `(a − b) / a`. It has no unit.
    pub flattening: f64,
}

impl Ellipsoid {
    /// The WGS84 ellipsoid in the ITRF93 Earth frame.
    pub const WGS84: Self = Self::new(FrameId::EARTH_ITRF93, 6_378_137.0, 1.0 / 298.257_223_563);
    /// The GRS80 ellipsoid in the ITRF93 Earth frame.
    pub const GRS80: Self = Self::new(FrameId::EARTH_ITRF93, 6_378_137.0, 1.0 / 298.257_222_101);
    /// The IAU lunar sphere in the mean Earth / polar axis frame.
    pub const MOON_ME: Self = Self::sphere(FrameId::MOON_ME, 1_737_400.0);
    /// The IAU lunar sphere in the principal axis frame.
    pub const MOON_PA: Self = Self::sphere(FrameId::MOON_PA, 1_737_400.0);
    /// The IAU Mars ellipsoid in the IAU Mars frame.
    pub const MARS_IAU: Self =
        Self::new(FrameId::MARS_IAU, 3_396_190.0, 1.0 / 169.894_447_223_611_8);

    /// Builds an ellipsoid from its frame and its two defining
    /// parameters. [`Self::is_plausible`] screens the values.
    #[must_use]
    pub const fn new(frame: FrameId, semi_major_axis_m: f64, flattening: f64) -> Self {
        Self {
            frame,
            semi_major_axis_m,
            flattening,
        }
    }

    /// Builds a sphere of radius `radius_m` in `frame`.
    #[must_use]
    pub const fn sphere(frame: FrameId, radius_m: f64) -> Self {
        Self::new(frame, radius_m, 0.0)
    }

    /// Whether the ellipsoid can define geodetic coordinates. The
    /// semi-major axis must be finite and positive. The flattening must
    /// be finite, 0 or more, and less than 1. The frame must be fixed to
    /// the body: an inertial frame turns under the surface.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        self.semi_major_axis_m.is_finite()
            && self.semi_major_axis_m > 0.0
            && self.flattening.is_finite()
            && self.flattening >= 0.0
            && self.flattening < 1.0
            && !self.frame.is_inertial()
    }
}

#[cfg(test)]
mod tests;
