//! EXPLORATION: the tilt of the elevation error under a footprint as the
//! truth of a terrain fix.

use super::dem::{fields, fix, hover, mount, world};
use super::world::{CORRELATION_ZOOM, DemError};

const SPACING_M: f64 = 1_000.0;

fn half(agl_m: f64) -> [f64; 2] {
    [0.4 * agl_m, 160.0 / 150.0 / 2.0 * agl_m]
}

fn predicted(field: DemError, centre: [f64; 2], agl_m: f64) -> [f64; 2] {
    let [sn, se, _] = field.plane_fit(centre, half(agl_m));
    [-agl_m * sn, -agl_m * se]
}

#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn explore_validate() {
    super::flight::show_events();
    let truth = world(DemError::None, CORRELATION_ZOOM);
    for agl_m in [400.0, 800.0] {
        let (mut mp, mut pp, mut mm, mut n) = (0.0, 0.0, 0.0, 0.0);
        let mut resid = 0.0;
        let mut pairs = Vec::new();
        for field in fields() {
            let map = world(field, CORRELATION_ZOOM);
            for k in 0..9 {
                let centre = [
                    f64::from(k % 3) * SPACING_M - SPACING_M,
                    f64::from(k / 3) * SPACING_M - SPACING_M,
                ];
                let flight = hover(k, agl_m);
                let nadir = mount(0.0, false);
                let (Some(a), Some(b)) = (
                    fix((&map, &truth), &flight, nadir),
                    fix((&truth, &truth), &flight, nadir),
                ) else {
                    continue;
                };
                let m = a.0 - b.0;
                let p = predicted(field, centre, agl_m);
                for axis in 0..2 {
                    mp += m[axis] * p[axis];
                    pp += p[axis] * p[axis];
                    mm += m[axis] * m[axis];
                    n += 1.0;
                    pairs.push((m[axis], p[axis]));
                }
            }
        }
        let gain = mp / pp;
        for (m, p) in &pairs {
            resid += (m - gain * p).powi(2);
        }
        let r = mp / (mp.abs().max(1e-12)).signum() / (pp * mm).sqrt() * mp.signum();
        tracing::warn!(
            "VALIDATE agl={agl_m} n={n} gain={gain:.2} r={r:.2} rms_measured={:.2} rms_predicted={:.2} rms_resid={:.2}",
            (mm / n).sqrt(),
            (pp / n).sqrt(),
            (resid / n).sqrt()
        );
    }
}

#[test]
fn explore_structure() {
    super::flight::show_events();
    let seeds: Vec<DemError> = (0..60)
        .map(|seed| DemError::Field {
            sigma_m: 5.0,
            length_m: 300.0,
            seed: 1_000 + seed,
        })
        .collect();
    for agl_m in [400.0, 800.0, 1_760.0] {
        let base: Vec<[f64; 2]> = seeds
            .iter()
            .map(|f| predicted(*f, [0.0, 0.0], agl_m))
            .collect();
        let rms = (base.iter().map(|p| p[0] * p[0] + p[1] * p[1]).sum::<f64>()
            / (2.0 * base.len() as f64))
            .sqrt();
        let mut line = format!("SEPARATION agl={agl_m} rms={rms:.2}:");
        for d in [235.0, 470.0, 940.0, 1_400.0, 2_000.0] {
            let other: Vec<[f64; 2]> = seeds
                .iter()
                .map(|f| predicted(*f, [d, 0.0], agl_m))
                .collect();
            let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
            for (a, b) in base.iter().zip(&other) {
                for axis in 0..2 {
                    ab += a[axis] * b[axis];
                    aa += a[axis] * a[axis];
                    bb += b[axis] * b[axis];
                }
            }
            line.push_str(&format!(" d={d}:{:.2}", ab / (aa * bb).sqrt()));
        }
        tracing::warn!("{line}");
    }
    for (h1, h2) in [(800.0, 1_600.0), (1_600.0, 1_800.0), (400.0, 800.0)] {
        let a: Vec<[f64; 2]> = seeds
            .iter()
            .map(|f| predicted(*f, [0.0, 0.0], h1))
            .collect();
        let b: Vec<[f64; 2]> = seeds
            .iter()
            .map(|f| predicted(*f, [0.0, 0.0], h2))
            .collect();
        let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
        for (x, y) in a.iter().zip(&b) {
            for axis in 0..2 {
                ab += x[axis] * y[axis];
                aa += x[axis] * x[axis];
                bb += y[axis] * y[axis];
            }
        }
        tracing::warn!(
            "HEIGHTS {h1}/{h2}: corr={:.2} rms ratio={:.2}",
            ab / (aa * bb).sqrt(),
            (bb / aa).sqrt()
        );
    }
}
