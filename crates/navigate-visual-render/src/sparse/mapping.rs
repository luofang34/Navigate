//! The map coordinates of a march point: a quadratic model about the eye in
//! place of an ellipsoid conversion at every step.

use nalgebra::{Matrix3, Vector3};
use navigate_visual::LocalFrame;

/// The largest disagreement with the exact conversion that a model may
/// have anywhere within its reach, metres on the ground or in height.
///
/// Five centimetres is a few hundredths of a texel of the imagery zoom
/// that a terrain view selects from a few hundred metres up, and far below
/// the relief of the elevation grid. The third-order term that the model
/// leaves out grows with the latitude: at a reach of 6 km it is about
/// 4 mm at 47°, 15 mm at 70° and 5 cm at 79°; farther north a march takes
/// the exact conversion.
const MODEL_TOLERANCE_M: f64 = 0.05;

/// Step of the central differences that take the model, metres: long
/// enough that rounding of the Mercator coordinates stays far below the
/// tolerance, short enough that the fourth-order term is negligible.
const DIFFERENCE_STEP_M: f64 = 100.0;

/// Equatorial radius of WGS84, metres: one unit of Web Mercator spans its
/// circumference at the equator.
const WGS84_A_M: f64 = 6_378_137.0;

/// Unit Web Mercator coordinates and the altitude of a frame position.
pub(crate) type MapPoint = ([f64; 2], f64);

/// How a march finds the map coordinates of its points.
#[derive(Clone, Debug)]
pub(crate) enum MapCoordinates {
    /// The conversion of the frame at every point.
    Exact(LocalFrame),
    /// A quadratic model about the eye that agrees with the conversion of
    /// the frame within [`MODEL_TOLERANCE_M`] over the reach of the march.
    Model(Box<Quadratic>),
}

impl MapCoordinates {
    /// The coordinates for a march from `eye` out to `reach_m`: the model
    /// when it holds over the whole reach, else the exact conversion, as at
    /// a pole, across the antimeridian, or on a frame whose conversion is
    /// already cheap.
    pub(crate) fn about(frame: LocalFrame, eye: Vector3<f64>, reach_m: f64) -> Self {
        if matches!(frame, LocalFrame::AnchorMercator { .. }) {
            return Self::Exact(frame);
        }
        match Quadratic::about(frame, eye) {
            Some(model) if model.holds(frame, reach_m) => Self::Model(Box::new(model)),
            _ => Self::Exact(frame),
        }
    }

    /// The map coordinates of `point`.
    pub(crate) fn at(&self, point: Vector3<f64>) -> MapPoint {
        match self {
            Self::Exact(frame) => exact(*frame, point),
            Self::Model(model) => model.at(point),
        }
    }
}

/// The conversion of the frame: geodetic position, then Web Mercator.
pub(super) fn exact(frame: LocalFrame, point: Vector3<f64>) -> MapPoint {
    if matches!(frame, LocalFrame::AnchorMercator { .. }) {
        (frame.mercator_xy(point), point.z)
    } else {
        let [lat_deg, lon_deg, alt_m] = frame.geodetic(point);
        let mx = (lon_deg + 180.0) / 360.0;
        let my = (1.0 - lat_deg.to_radians().tan().asinh() / std::f64::consts::PI) * 0.5;
        ([mx, my], alt_m)
    }
}

/// Second-order Taylor model of `(mx, my, alt)` about an eye.
#[derive(Clone, Debug)]
pub(crate) struct Quadratic {
    eye: Vector3<f64>,
    value: Vector3<f64>,
    /// Row `k` is the gradient of output `k`: Mercator x, Mercator y,
    /// altitude.
    gradient: Matrix3<f64>,
    /// The Hessian of each output.
    hessian: [Matrix3<f64>; 3],
    /// Metres on the ground per unit of Web Mercator at the eye.
    metres_per_unit: f64,
}

impl Quadratic {
    /// The model from central differences of the conversion about `eye`;
    /// none when a difference is not finite.
    fn about(frame: LocalFrame, eye: Vector3<f64>) -> Option<Self> {
        let h = DIFFERENCE_STEP_M;
        let f = |offset: Vector3<f64>| {
            let ([mx, my], alt) = exact(frame, eye + offset);
            Vector3::new(mx, my, alt)
        };
        let (x, y, z) = (Vector3::x(), Vector3::y(), Vector3::z());
        let value = f(Vector3::zeros());
        let slope = |a: Vector3<f64>| (f(a * h) - f(-a * h)) / (2.0 * h);
        let curve = |a: Vector3<f64>| (f(a * h) - 2.0 * value + f(-a * h)) / (h * h);
        let mixed = |a: Vector3<f64>, b: Vector3<f64>| {
            (f((a + b) * h) - f((a - b) * h) - f((b - a) * h) + f(-(a + b) * h)) / (4.0 * h * h)
        };
        let gradient = Matrix3::from_columns(&[slope(x), slope(y), slope(z)]);
        let (xx, yy, zz) = (curve(x), curve(y), curve(z));
        let (xy, xz, yz) = (mixed(x, y), mixed(x, z), mixed(y, z));
        let of = |k: fn(&Vector3<f64>) -> f64| {
            Matrix3::new(
                k(&xx),
                k(&xy),
                k(&xz),
                k(&xy),
                k(&yy),
                k(&yz),
                k(&xz),
                k(&yz),
                k(&zz),
            )
        };
        let hessian = [of(|v| v.x), of(|v| v.y), of(|v| v.z)];
        let ([_, my], _) = exact(frame, eye);
        // The latitude of the eye from its Mercator ordinate.
        let lat = (std::f64::consts::PI * (1.0 - 2.0 * my)).sinh().atan();
        let metres_per_unit = 2.0 * std::f64::consts::PI * WGS84_A_M * lat.cos();
        let finite = value.iter().all(|v| v.is_finite())
            && gradient.iter().all(|v| v.is_finite())
            && hessian.iter().flat_map(|m| m.iter()).all(|v| v.is_finite())
            && metres_per_unit.is_finite();
        finite.then_some(Self {
            eye,
            value,
            gradient,
            hessian,
            metres_per_unit,
        })
    }

    /// The modelled map coordinates of `point`.
    fn at(&self, point: Vector3<f64>) -> MapPoint {
        let d = point - self.eye;
        let linear = self.value + self.gradient * d;
        let [h0, h1, h2] = &self.hessian;
        let q = |h: &Matrix3<f64>| 0.5 * d.dot(&(h * d));
        ([linear.x + q(h0), linear.y + q(h1)], linear.z + q(h2))
    }

    /// Whether the model agrees with the conversion within the tolerance
    /// at 98 directions on each of three spheres about the eye out to
    /// `reach_m`. The left-out term grows with the cube of the distance
    /// and smoothly with direction, so the farthest sphere bounds it.
    fn holds(&self, frame: LocalFrame, reach_m: f64) -> bool {
        let mut directions = Vec::with_capacity(98);
        for x in -2_i32..=2 {
            for y in -2_i32..=2 {
                for z in -2_i32..=2 {
                    if x.abs().max(y.abs()).max(z.abs()) == 2 {
                        let v = Vector3::new(f64::from(x), f64::from(y), f64::from(z));
                        directions.push(v.normalize());
                    }
                }
            }
        }
        [reach_m / 3.0, 2.0 * reach_m / 3.0, reach_m]
            .iter()
            .flat_map(|r| directions.iter().map(move |u| self.eye + u * *r))
            .all(|point| self.error_m(frame, point) <= MODEL_TOLERANCE_M)
    }

    /// The disagreement of the model with the conversion at `point`,
    /// metres: the larger of the ground and the height errors.
    fn error_m(&self, frame: LocalFrame, point: Vector3<f64>) -> f64 {
        let ([mx, my], alt) = exact(frame, point);
        let ([px, py], palt) = self.at(point);
        let ground = (mx - px).hypot(my - py) * self.metres_per_unit;
        let error = ground.max((alt - palt).abs());
        if error.is_finite() {
            error
        } else {
            f64::INFINITY
        }
    }
}
