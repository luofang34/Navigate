//! Great-circle geometry on a sphere, for planning only.

use navigate_geodesy::float_math::{acos, atan2, cos, hypot, sin};

/// Mean radius of the Earth in metres.
const MEAN_RADIUS_M: f64 = 6_371_008.8;

type Unit = [f64; 3];

fn unit([lat, lon]: [f64; 2]) -> Unit {
    let (lat, lon) = (lat.to_radians(), lon.to_radians());
    [cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)]
}

fn angle(a: Unit, b: Unit) -> f64 {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    acos((ax * bx + ay * by + az * bz).clamp(-1.0, 1.0))
}

/// Great-circle distance in metres between two `[lat, lon]` points in
/// degrees.
#[must_use]
pub fn great_circle_m(a: [f64; 2], b: [f64; 2]) -> f64 {
    angle(unit(a), unit(b)) * MEAN_RADIUS_M
}

/// The point at fraction `t` of the great circle from `a` to `b`. A `t`
/// past 1 continues the circle beyond `b`.
#[must_use]
pub fn point_along(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    let (ua, ub) = (unit(a), unit(b));
    let theta = angle(ua, ub);
    let (wa, wb) = if theta < 1e-12 {
        (1.0 - t, t)
    } else {
        (
            sin((1.0 - t) * theta) / sin(theta),
            sin(t * theta) / sin(theta),
        )
    };
    let ([ax, ay, az], [bx, by, bz]) = (ua, ub);
    let (x, y, z) = (wa * ax + wb * bx, wa * ay + wb * by, wa * az + wb * bz);
    [atan2(z, hypot(x, y)).to_degrees(), atan2(y, x).to_degrees()]
}
