use nalgebra::{Matrix3, Vector3};
use navigate_contract::GeodeticPosition;
use navigate_geodesy::geodetic_to_body_fixed;

use super::PlanetaryBody;

/// Central differences of `f` with step `h` in each axis.
fn finite_difference(
    f: impl Fn(&Vector3<f64>) -> Vector3<f64>,
    r: &Vector3<f64>,
    h: f64,
) -> Matrix3<f64> {
    let mut jacobian = Matrix3::zeros();
    for axis in 0..3 {
        let mut step = Vector3::zeros();
        step[axis] = h;
        let column = (f(&(r + step)) - f(&(r - step))) / (2.0 * h);
        jacobian.set_column(axis, &column);
    }
    jacobian
}

fn body_fixed(body: &PlanetaryBody, lat_deg: f64, alt_m: f64) -> Vector3<f64> {
    let position = GeodeticPosition::new(lat_deg.to_radians(), 0.3, alt_m);
    let r = geodetic_to_body_fixed(&position, body.ellipsoid);
    Vector3::new(r.x_m, r.y_m, r.z_m)
}

#[test]
fn surface_gravity_has_the_known_magnitude_on_each_body() {
    let cases = [
        (PlanetaryBody::EARTH_WGS84, 0.0, 9.780),
        (PlanetaryBody::EARTH_WGS84, 90.0, 9.832),
        (PlanetaryBody::MOON_ME, 0.0, 1.62),
        (PlanetaryBody::MARS_IAU, 0.0, 3.71),
    ];
    for (body, lat_deg, expected) in cases {
        assert!(body.is_plausible());
        let (g, _) = body.apparent_gravity_body_fixed(&body_fixed(&body, lat_deg, 0.0));
        assert!(
            (g.norm() - expected).abs() < 0.01,
            "{lat_deg}: {}",
            g.norm()
        );
        let surface = body.surface_gravity_mps2(lat_deg.to_radians());
        assert!((surface - g.norm()).abs() < 1e-12, "{surface}");
    }
}

#[test]
fn the_somigliana_form_matches_the_wgs84_normal_gravity() {
    // WGS84 normal gravity (NIMA TR8350.2): 9.7803253359 m/s² at the
    // equator, 9.8321849378 m/s² at the poles, and 9.806199203 m/s² at
    // 45°. The J2 model of the body omits the higher zonal terms.
    let earth = PlanetaryBody::EARTH_WGS84;
    for (lat_deg, expected) in [
        (0.0, 9.780_325_335_9),
        (45.0, 9.806_199_203),
        (90.0, 9.832_184_937_8),
    ] {
        let gamma = earth.somigliana_gravity_mps2(f64::to_radians(lat_deg));
        assert!((gamma - expected).abs() < 2e-4, "{lat_deg}: {gamma}");
    }
    let equator = earth.somigliana_gravity_mps2(0.0);
    assert!((equator - earth.surface_gravity_mps2(0.0)).abs() < 1e-12);
}

#[test]
fn apparent_gravity_gradient_matches_finite_differences() {
    for body in [
        PlanetaryBody::EARTH_WGS84,
        PlanetaryBody::MOON_PA,
        PlanetaryBody::MARS_IAU,
    ] {
        for lat_deg in [-90.0, -35.0, 0.0, 61.0] {
            let r = body_fixed(&body, lat_deg, 2_000.0);
            let (_, analytic) = body.apparent_gravity_body_fixed(&r);
            let numeric = finite_difference(|p| body.apparent_gravity_body_fixed(p).0, &r, 10.0);
            let error = (analytic - numeric).abs().max();
            assert!(error < 1e-6 * analytic.abs().max(), "{lat_deg}: {error}");
        }
    }
}

#[test]
fn an_implausible_body_is_refused() {
    let body = PlanetaryBody {
        gm_m3_per_s2: -1.0,
        ..PlanetaryBody::EARTH_WGS84
    };
    assert!(!body.is_plausible());
    let spinning = PlanetaryBody {
        rotation_rate_rad_per_s: f64::NAN,
        ..PlanetaryBody::MARS_IAU
    };
    assert!(!spinning.is_plausible());
}

/// The lunar constants are those of the GRAIL GRGM1200A field: its `μ`, and
/// its `C̄₂₀` at the 1738 km reference radius, which fixes `J2·R²`.
#[test]
fn moon_gravity_is_grgm1200a() {
    const GRGM1200A_GM_M3_PER_S2: f64 = 4.902_800_122_445_3e12;
    const GRGM1200A_C20_NORMALIZED: f64 = -9.088_433_934_742_43e-5;
    const GRGM1200A_RADIUS_M: f64 = 1_738_000.0;
    let moon = PlanetaryBody::MOON_PA;
    let radius = moon.ellipsoid.semi_major_axis_m();
    let j2_r2 = -GRGM1200A_C20_NORMALIZED * 5.0_f64.sqrt() * GRGM1200A_RADIUS_M.powi(2);
    assert!(((moon.j2 * radius * radius) / j2_r2 - 1.0).abs() < 1e-9);
    assert!((moon.gm_m3_per_s2 / GRGM1200A_GM_M3_PER_S2 - 1.0).abs() < 1e-12);
    assert_eq!(PlanetaryBody::MOON_ME.j2, moon.j2);
}
