//! A photo laid on flat ground, north up, for one height, camera angle and heading.
//!
//! The ground under an unknown camera is unknown, so the search uses a flat plane at
//! the camera height below it. The ranking and the map-window check tolerate the
//! error that hills give; the final locate step uses the terrain.
use crate::orientation;
use image::{GrayImage, Luma, RgbImage};
use nalgebra::{UnitQuaternion, Vector3};
use navigate_place::{CANVAS_PX, Canvas};

/// Farthest ground from the point below the camera that a projection keeps, metres,
/// and in camera heights.
const RANGE_M: f64 = 6_000.0;
const RANGE_HEIGHTS: f64 = 10.0;

/// One hypothesis of the camera: height above the ground, angle below the horizon
/// and heading, and the photo calibration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// Height above the ground, metres.
    pub height_m: f64,
    /// Angle of the optical axis below the horizon, degrees.
    pub pitch_deg: f64,
    /// Compass heading of the optical axis, degrees.
    pub heading_deg: f64,
    /// Focal length, photo pixels.
    pub focal_px: f64,
}

impl View {
    fn rotation(&self) -> UnitQuaternion<f64> {
        orientation(self.heading_deg, self.pitch_deg, 0.0)
    }

    fn range_m(&self) -> f64 {
        RANGE_M.min(RANGE_HEIGHTS * self.height_m)
    }

    /// Metres east and north of the point below the camera, of the ground that a
    /// photo pixel shows, with the ground area of the pixel.
    fn ground(
        &self,
        size: (u32, u32),
        rotation: &UnitQuaternion<f64>,
        x: f64,
        y: f64,
    ) -> Option<([f64; 2], f64)> {
        let (w, h) = (f64::from(size.0), f64::from(size.1));
        let ray = rotation
            * Vector3::new(
                (x - w / 2.0) / self.focal_px,
                -(y - h / 2.0) / self.focal_px,
                -1.0,
            )
            .normalize();
        if ray.z > -0.01 {
            return None;
        }
        let t = self.height_m / -ray.z;
        let point = [ray.x * t, ray.y * t];
        (point[0].hypot(point[1]) <= self.range_m()).then_some((point, t * t / -ray.z))
    }

    /// The centre of the ground in view by area, metres east and north of the point
    /// below the camera.
    pub fn footprint_centre(&self, size: (u32, u32)) -> Option<[f64; 2]> {
        let rotation = self.rotation();
        let (mut sum, mut weight) = ([0.0, 0.0], 0.0);
        for j in 0..48 {
            for i in 0..64 {
                let x = (f64::from(i) + 0.5) * f64::from(size.0) / 64.0;
                let y = (f64::from(j) + 0.5) * f64::from(size.1) / 48.0;
                if let Some((p, a)) = self.ground(size, &rotation, x, y) {
                    sum = [sum[0] + p[0] * a, sum[1] + p[1] * a];
                    weight += a;
                }
            }
        }
        (weight > 0.0).then(|| [sum[0] / weight, sum[1] / weight])
    }

    /// Photo pixel of a ground point, metres east and north of the point below the
    /// camera, if the photo shows it.
    fn pixel(
        &self,
        size: (u32, u32),
        inverse: &UnitQuaternion<f64>,
        [east, north]: [f64; 2],
    ) -> Option<[f64; 2]> {
        if east.hypot(north) > self.range_m() {
            return None;
        }
        let eye = inverse * Vector3::new(east, north, -self.height_m);
        if eye.z >= -1.0 {
            return None;
        }
        let x = f64::from(size.0) / 2.0 + self.focal_px * eye.x / -eye.z;
        let y = f64::from(size.1) / 2.0 - self.focal_px * eye.y / -eye.z;
        (x >= 0.0 && y >= 0.0 && x < f64::from(size.0) - 1.0 && y < f64::from(size.1) - 1.0)
            .then_some([x, y])
    }
}

fn bilinear<const N: usize>(at: impl Fn(u32, u32) -> [f32; N], [x, y]: [f64; 2]) -> [f32; N] {
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (fx, fy) = ((x - x.floor()) as f32, (y - y.floor()) as f32);
    let (a, b, c, d) = (
        at(x0, y0),
        at(x0 + 1, y0),
        at(x0, y0 + 1),
        at(x0 + 1, y0 + 1),
    );
    std::array::from_fn(|i| {
        (a[i] * (1.0 - fx) + b[i] * fx) * (1.0 - fy) + (c[i] * (1.0 - fx) + d[i] * fx) * fy
    })
}

/// The photo on the ground in a square of `side_m` centred at `centre`, metres east
/// and north of the point below the camera.
pub fn canvas(photo: &RgbImage, view: &View, centre: [f64; 2], side_m: f64) -> Canvas {
    let size = photo.dimensions();
    let inverse = view.rotation().inverse();
    let step = side_m / CANVAS_PX as f64;
    let mut out = Canvas::empty();
    for row in 0..CANVAS_PX {
        for column in 0..CANVAS_PX {
            let east = centre[0] + (column as f64 + 0.5) * step - side_m / 2.0;
            let north = centre[1] + side_m / 2.0 - (row as f64 + 0.5) * step;
            if let Some(p) = view.pixel(size, &inverse, [east, north]) {
                out.rgb[row * CANVAS_PX + column] =
                    bilinear(|x, y| photo.get_pixel(x, y).0.map(f32::from), p);
                out.valid[row * CANVAS_PX + column] = true;
            }
        }
    }
    out
}

/// The photo on the ground in grey, `size` pixels of `gsd_m` metres, centred at
/// `centre`, north up. Zero marks ground outside the photo.
pub fn grey(
    photo: &GrayImage,
    view: &View,
    centre: [f64; 2],
    gsd_m: f64,
    [w, h]: [u32; 2],
) -> GrayImage {
    let size = photo.dimensions();
    let inverse = view.rotation().inverse();
    GrayImage::from_fn(w, h, |u, v| {
        let east = centre[0] + (f64::from(u) + 0.5 - f64::from(w) / 2.0) * gsd_m;
        let north = centre[1] - (f64::from(v) + 0.5 - f64::from(h) / 2.0) * gsd_m;
        match view.pixel(size, &inverse, [east, north]) {
            Some(p) => Luma([
                bilinear(|x, y| [f32::from(photo.get_pixel(x, y).0[0])], p)[0].clamp(1.0, 255.0)
                    as u8,
            ]),
            None => Luma([0]),
        }
    })
}
