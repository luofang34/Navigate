//! A synthetic star camera image for the tests.
//!
//! The body axes are the local NED axes at [`position`], and the camera
//! looks at the zenith. Real catalogue stars at [`epoch`] go through the
//! true attitude into a 16-bit image with a Gaussian point spread function
//! and seeded Gaussian noise.

use nalgebra::{UnitQuaternion, Vector2, Vector3};
use navigate_camera::{CameraMount, ImageView, ImagerModel, Intrinsics, Pixels};
use navigate_celestial::{Epoch, StarSight};
use navigate_contract::GeodeticPosition;
use navigate_fusion::inertial::ned_to_body_fixed_dcm;

use crate::{
    AttitudePrior, AttitudeSolution, DetectConfig, IdentifyConfig, SolveConfig, StarCentroid,
    StarMatch, catalogue_ecef, detect_stars, identify_stars, solve_attitude,
};

/// Image side in pixels.
pub(crate) const SIZE: u32 = 1024;
/// Faintest star in the image.
pub(crate) const RENDER_MAGNITUDE: f64 = 6.5;
/// Faintest star of the identification catalogue.
pub(crate) const CATALOGUE_MAGNITUDE: f64 = 5.5;
/// One standard deviation of the point spread function in pixels.
const PSF_SIGMA_PX: f64 = 1.3;
/// Sky background in counts.
pub(crate) const BACKGROUND: f64 = 200.0;

pub(crate) fn epoch() -> Epoch {
    Epoch::from_gregorian_utc_at_midnight(2026, 3, 20)
}

pub(crate) fn position() -> GeodeticPosition {
    GeodeticPosition::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 500.0)
}

/// The true rotation from the body axes (local NED) into the ECEF axes.
pub(crate) fn truth() -> UnitQuaternion<f64> {
    let p = position();
    UnitQuaternion::from_matrix(&ned_to_body_fixed_dcm(p.latitude_rad, p.longitude_rad))
}

pub(crate) fn imager() -> ImagerModel {
    let intrinsics = Intrinsics::centred(SIZE, SIZE, 20.0_f64.to_radians()).expect("intrinsics");
    ImagerModel::pinhole(intrinsics)
}

pub(crate) fn mount() -> CameraMount {
    CameraMount::zenith()
}

pub(crate) fn catalogue(faintest: f64) -> Vec<StarSight> {
    catalogue_ecef(epoch(), faintest).expect("catalogue")
}

/// A point source in the image.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Source {
    pub(crate) hr: u16,
    pub(crate) pixel: Vector2<f64>,
    pub(crate) flux: f64,
}

/// Total counts of a star of visual magnitude `v`.
pub(crate) fn flux_of(v: f64) -> f64 {
    2.0e4 * 10f64.powf(-0.4 * (v - 5.0))
}

/// The catalogue stars that `attitude` puts at least 4 pixels inside the
/// image.
pub(crate) fn sources(catalogue: &[StarSight], attitude: &UnitQuaternion<f64>) -> Vec<Source> {
    let k = imager().intrinsics;
    let cam_from_ecef = mount().cam_from_body * attitude.inverse();
    catalogue
        .iter()
        .filter_map(|s| {
            let pixel = k.project(&(cam_from_ecef * Vector3::from(s.direction)))?;
            let inside = 4.0..f64::from(SIZE) - 5.0;
            (inside.contains(&pixel.x) && inside.contains(&pixel.y)).then_some(Source {
                hr: s.hr,
                pixel,
                flux: flux_of(s.v_magnitude),
            })
        })
        .collect()
}

/// The error function, Abramowitz and Stegun 7.1.26 (error below 1.5e-7).
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let y = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}

/// The fraction of a unit Gaussian centred at `centre` inside the pixel
/// `index` (edges at `index ± 0.5`).
fn pixel_fraction(index: i64, centre: f64) -> f64 {
    let scale = 1.0 / (PSF_SIGMA_PX * std::f64::consts::SQRT_2);
    let low = (index as f64 - 0.5 - centre) * scale;
    let high = (index as f64 + 0.5 - centre) * scale;
    0.5 * (erf(high) - erf(low))
}

/// A seeded generator of standard normal values.
struct Noise(u64);

impl Noise {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// A 16-bit image of `sources` with Gaussian noise of `noise_sigma` counts.
pub(crate) fn render(sources: &[Source], noise_sigma: f64, seed: u64) -> Vec<u16> {
    let n = SIZE as usize;
    let mut image = vec![BACKGROUND; n * n];
    let reach = (6.0 * PSF_SIGMA_PX).ceil() as i64;
    for s in sources {
        let (cx, cy) = (s.pixel.x.round() as i64, s.pixel.y.round() as i64);
        for y in (cy - reach).max(0)..=(cy + reach).min(n as i64 - 1) {
            let fy = pixel_fraction(y, s.pixel.y);
            for x in (cx - reach).max(0)..=(cx + reach).min(n as i64 - 1) {
                image[y as usize * n + x as usize] += s.flux * fy * pixel_fraction(x, s.pixel.x);
            }
        }
    }
    let mut noise = Noise(seed.max(1));
    image
        .iter()
        .map(|v| {
            let value = if noise_sigma > 0.0 {
                v + noise_sigma * noise.normal()
            } else {
                *v
            };
            value.round().clamp(0.0, f64::from(u16::MAX)) as u16
        })
        .collect()
}

pub(crate) fn view(data: &[u16]) -> ImageView<'_> {
    let pixels = Pixels::Gray16 {
        data,
        stride: SIZE as usize,
        significant_bits: 16,
    };
    ImageView::new(SIZE, SIZE, pixels).expect("view")
}

/// A prior `angle_rad` from the truth about a tilted axis.
pub(crate) fn prior(angle_rad: f64, sigma_rad: f64) -> AttitudePrior {
    let axis = Vector3::new(0.6, -0.48, 0.64).normalize();
    AttitudePrior {
        ecef_from_body: truth() * UnitQuaternion::from_scaled_axis(axis * angle_rad),
        sigma_rad,
    }
}

/// One image through the whole chain.
pub(crate) struct Pipeline {
    pub(crate) centroids: Vec<StarCentroid>,
    pub(crate) catalogue: Vec<StarSight>,
    pub(crate) matches: Vec<StarMatch>,
    pub(crate) solution: AttitudeSolution,
}

/// Detection, identification with a 1° prior error, and the solution, with
/// the extra point sources `extra`.
pub(crate) fn pipeline(noise_sigma: f64, seed: u64, extra: &[Source]) -> Pipeline {
    let mut sources = sources(&catalogue(RENDER_MAGNITUDE), &truth());
    sources.extend_from_slice(extra);
    let data = render(&sources, noise_sigma, seed);
    let centroids = detect_stars(&view(&data), &DetectConfig::default()).expect("detect");
    let catalogue = catalogue(CATALOGUE_MAGNITUDE);
    let one_degree = 1.0_f64.to_radians();
    let prior = prior(one_degree, one_degree);
    let (imager, mount) = (imager(), mount());
    let matches = identify_stars(
        &centroids,
        &imager,
        &mount,
        Some(&prior),
        &catalogue,
        &IdentifyConfig::default(),
    )
    .expect("identify");
    let solution = solve_attitude(
        &matches,
        &centroids,
        &imager,
        &mount,
        &catalogue,
        &SolveConfig::default(),
    )
    .expect("solve");
    Pipeline {
        centroids,
        catalogue,
        matches,
        solution,
    }
}
