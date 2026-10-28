//! Fit the long-range series to JPL DE431 and write `long_range_fit.txt`.
//!
//! ```text
//! cargo run --release -p navigate-celestial --features anise --example long_range_fit -- \
//!     de431_part-1.bsp,de431_part-2.bsp VSOP87B.ear elp82b_directory \
//!     crates/navigate-celestial/fixtures/long_range_fit.txt
//! ```
//!
//! The samples are 200000 geometric geocentric positions of the Moon and the
//! Sun from 1 January -1999 to 1 December 2999 TDB, spread over the orbital
//! phases by a golden-ratio sequence. The committed `de431-geocentric.txt`,
//! at other epochs, tests the result.
//!
//! The fit corrects the polynomials of W1, W2, W3, the mean longitude of the
//! Earth-Moon barycentre, and its perihelion to cubic order, by Gauss-Newton
//! iteration on the Moon longitude, latitude, and distance. A separate
//! least-squares fit gives a cubic correction to the VSOP87B longitude. The
//! tool writes the largest remaining errors in each millennium and the tidal
//! acceleration that the fitted T^2 term of W1 implies.

#[path = "long_range/evaluate.rs"]
mod evaluate;
#[path = "long_range/theory.rs"]
mod theory;

use std::{f64::consts::PI, io::Write as _, path::Path};

use anise::almanac::Almanac;
use nalgebra::{DMatrix, DVector};
use navigate_celestial::{AniseEphemeris, Ephemeris, Epoch, TimeScale};
use theory::{ARCSECONDS_PER_RADIAN, Elements, Elp, Error, Fit, MOON_NAMES, ORDER, SPAN_CENTURIES};

const SAMPLES: u32 = 200_000;
const GOLDEN_FRACTION: f64 = 0.618_033_988_749_894_9;
/// Every fourth sample enters the Jacobian.
const JACOBIAN_STRIDE: usize = 4;
const ITERATIONS: usize = 3;
const PARAMETERS: usize = MOON_NAMES.len() * (ORDER + 1);
/// The DE200 tidal acceleration of the theory, in arcseconds per century
/// squared. It is in the T^2 term of W1.
const THEORY_TIDAL_ACCELERATION: f64 = -23.8946;
/// The mean distance of the Moon, which turns a range error into an angle.
const MOON_DISTANCE_M: f64 = 384.4e6;

/// One JPL sample: TDB centuries, then the Moon and the Sun as longitude,
/// latitude, and distance in the J2000 ecliptic.
struct Sample {
    t: f64,
    moon: [f64; 3],
    sun: [f64; 3],
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [kernels, vsop_path, elp_dir, out_path] = args.as_slice() else {
        return Err(
            "usage: long_range_fit <kernel.bsp>[,<kernel.bsp>...] <VSOP87B.ear> <elp82b directory> <fit>"
                .into(),
        );
    };
    let samples = samples(kernels)?;
    let elp = Elp::read(Path::new(elp_dir))?;
    let mut fit = Fit::default();
    let subset: Vec<&Sample> = samples.iter().step_by(JACOBIAN_STRIDE).collect();
    for _ in 0..ITERATIONS {
        fit = gauss_newton_step(&elp, &fit, &subset)?;
    }
    let all: Vec<&Sample> = samples.iter().collect();
    let moon_errors = moon_residuals(&elp, &fit, &all)?;

    let vsop = theory::vsop(Path::new(vsop_path))?;
    let sun_errors: Vec<f64> = all
        .iter()
        .map(|sample| {
            wrap(evaluate::sun_longitude(sample.t, &vsop) - sample.sun[0]) * ARCSECONDS_PER_RADIAN
        })
        .collect();
    let basis = DMatrix::from_fn(all.len(), ORDER + 1, |row, k| scaled_power(all[row].t, k));
    let solution = basis
        .clone()
        .svd(true, true)
        .solve(&-DVector::from_vec(sun_errors.clone()), 1e-12)?;
    for (k, value) in fit.sun.iter_mut().enumerate() {
        *value = solution[k] / SPAN_CENTURIES.powi(power(k));
    }
    let sun_errors: Vec<f64> = (basis * solution + DVector::from_vec(sun_errors))
        .iter()
        .copied()
        .collect();

    std::fs::write(out_path, text(&fit))?;
    report(&all, &moon_errors, &sun_errors, &fit)
}

fn samples(kernels: &str) -> Result<Vec<Sample>, Error> {
    let mut almanac = Almanac::default();
    for path in kernels.split(',') {
        almanac = almanac
            .load(path)
            .map_err(|error| format!("{path}: {error}"))?;
    }
    let provider = AniseEphemeris::new(&almanac, &[]);
    let start = Epoch::from_gregorian(-1999, 1, 1, 0, 0, 0, 0, TimeScale::TDB).to_tdb_seconds();
    let end = Epoch::from_gregorian(2999, 12, 1, 0, 0, 0, 0, TimeScale::TDB).to_tdb_seconds();
    (0..SAMPLES)
        .map(|row| {
            let fraction = (f64::from(row) * GOLDEN_FRACTION).fract();
            let epoch = Epoch::from_tdb_seconds(start + fraction * (end - start));
            let position = |body: i32| {
                provider
                    .body_state(body, epoch)
                    .map(|state| state.position_m)
            };
            let earth = position(399)?;
            let geocentric = |body: [f64; 3]| {
                evaluate::spherical(evaluate::ecliptic(std::array::from_fn(|i| {
                    body[i] - earth[i]
                })))
            };
            Ok(Sample {
                t: epoch.to_tdb_centuries_since_j2000(),
                moon: geocentric(position(301)?),
                sun: geocentric(position(10)?),
            })
        })
        .collect()
}

fn wrap(angle: f64) -> f64 {
    (angle + PI).rem_euclid(2.0 * PI) - PI
}

/// The power of t of parameter column `k`.
fn power(k: usize) -> i32 {
    i32::try_from(k).unwrap_or(i32::MAX)
}

/// `(t / 40)^k`, the basis that keeps the columns of the fit comparable.
fn scaled_power(t: f64, k: usize) -> f64 {
    (t / SPAN_CENTURIES).powi(power(k))
}

fn with_parameters(fit: &Fit, parameters: &[f64]) -> Fit {
    let mut fit = *fit;
    for (index, value) in parameters.iter().enumerate() {
        let (name, k) = (index / (ORDER + 1), index % (ORDER + 1));
        fit.moon[name][k] = value / SPAN_CENTURIES.powi(power(k));
    }
    fit
}

fn parameters(fit: &Fit) -> Vec<f64> {
    (0..PARAMETERS)
        .map(|index| {
            let (name, k) = (index / (ORDER + 1), index % (ORDER + 1));
            fit.moon[name][k] * SPAN_CENTURIES.powi(power(k))
        })
        .collect()
}

/// Moon longitude, latitude, and distance errors in arcseconds, in blocks.
fn moon_residuals(elp: &Elp, fit: &Fit, samples: &[&Sample]) -> Result<Vec<f64>, Error> {
    let elements = Elements::new(fit);
    let terms = elp.terms(&elements);
    let errors: Vec<[f64; 3]> = std::thread::scope(|scope| {
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        let handles: Vec<_> = samples
            .chunks(samples.len().div_ceil(threads).max(1))
            .map(|chunk| {
                let (terms, w1) = (&terms, &elements.w[0]);
                scope.spawn(move || {
                    chunk
                        .iter()
                        .map(|sample| {
                            let [lon, lat, r] =
                                evaluate::spherical(evaluate::moon(sample.t, terms, w1));
                            [
                                wrap(lon - sample.moon[0]) * ARCSECONDS_PER_RADIAN,
                                (lat - sample.moon[1]) * ARCSECONDS_PER_RADIAN,
                                (r - sample.moon[2]) / MOON_DISTANCE_M * ARCSECONDS_PER_RADIAN,
                            ]
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().map_err(|_| "a residual thread failed"))
            .collect::<Result<Vec<_>, _>>()
            .map(|chunks| chunks.concat())
    })?;
    Ok((0..3)
        .flat_map(|i| errors.iter().map(move |e| e[i]))
        .collect())
}

fn gauss_newton_step(elp: &Elp, fit: &Fit, samples: &[&Sample]) -> Result<Fit, Error> {
    let x = parameters(fit);
    let r0 = moon_residuals(elp, fit, samples)?;
    // A step of 1 arcsecond at |t| = 40 in each parameter.
    let step = 1.0;
    let mut jacobian = DMatrix::zeros(r0.len(), PARAMETERS);
    for j in 0..PARAMETERS {
        let mut shifted = x.clone();
        shifted[j] += step;
        let r = moon_residuals(elp, &with_parameters(fit, &shifted), samples)?;
        for (row, (r, r0)) in r.iter().zip(&r0).enumerate() {
            jacobian[(row, j)] = (r - r0) / step;
        }
    }
    let dx = jacobian
        .svd(true, true)
        .solve(&-DVector::from_vec(r0), 1e-12)?;
    let x: Vec<f64> = x.iter().zip(dx.iter()).map(|(x, dx)| x + dx).collect();
    Ok(with_parameters(fit, &x))
}

fn text(fit: &Fit) -> String {
    let line = |name: &str, values: &[f64]| {
        let values: Vec<String> = values.iter().map(|v| format!("{v:?}")).collect();
        format!("{name} {}\n", values.join(" "))
    };
    let mut text = String::from(
        "# Generated by examples/long_range_fit.rs: corrections in arcseconds by power\n\
         # of TDB Julian centuries from J2000, fitted to DE431 from -1999 to 2999.\n",
    );
    for (name, values) in MOON_NAMES.iter().zip(&fit.moon) {
        text += &line(name, values);
    }
    text + &line("SUN", &fit.sun)
}

fn report(samples: &[&Sample], moon: &[f64], sun: &[f64], fit: &Fit) -> Result<(), Error> {
    let n = samples.len();
    let millennia = |errors: &[f64]| {
        (-4..1)
            .map(|millennium| {
                let range = f64::from(millennium) * 10.0..f64::from(millennium + 1) * 10.0;
                let largest = samples
                    .iter()
                    .zip(errors)
                    .filter(|(sample, _)| range.contains(&sample.t))
                    .map(|(_, error)| error.abs())
                    .fold(0.0, f64::max);
                format!("{largest:6.2}")
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    let elongation: Vec<f64> = moon.iter().zip(sun).map(|(moon, sun)| moon - sun).collect();
    let mut out = std::io::stdout().lock();
    writeln!(
        out,
        "largest errors in arcseconds by millennium, -2000 to 3000"
    )?;
    writeln!(out, "Moon longitude  {}", millennia(&moon[..n]))?;
    writeln!(out, "Moon latitude   {}", millennia(&moon[n..2 * n]))?;
    writeln!(out, "Moon distance   {}", millennia(&moon[2 * n..]))?;
    writeln!(out, "Sun longitude   {}", millennia(sun))?;
    writeln!(out, "elongation      {}", millennia(&elongation))?;
    writeln!(
        out,
        "implied tidal acceleration {:.4} arcsec/cy^2 (T^2 of W1), T^3 correction {:.6} arcsec/cy^3",
        THEORY_TIDAL_ACCELERATION + 2.0 * fit.moon[0][2],
        fit.moon[0][3]
    )?;
    Ok(())
}
