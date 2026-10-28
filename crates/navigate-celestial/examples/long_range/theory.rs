//! VSOP87B and ELP2000-82B terms for the long-range tables and their fit.
//!
//! ELP2000-82B follows the IMCCE Fortran ELP82B_2: the constants fitted to
//! DE200, the arguments of ELP 2000-85, and the Laskar precession to J2000.
//! Two changes extend it to -2000 to 3000:
//!
//! - The perturbation arguments keep the terms in t^2 to t^4 of the Delaunay
//!   arguments, as the main problem does. ELP82B_2 keeps only the linear
//!   part, which at -2000 moves the phase of the mean anomaly by 14 degrees.
//! - The polynomials of the mean longitudes W1, W2, W3, the mean longitude of
//!   the Earth-Moon barycentre, and its perihelion get the corrections of a
//!   [`Fit`], and so does the VSOP87B longitude. The fit replaces the DE200
//!   tidal acceleration of the theory: the T^2 term of W1 comes from DE431.

use std::{collections::BTreeMap, f64::consts::PI, path::Path};

/// Arcseconds in one radian.
pub const ARCSECONDS_PER_RADIAN: f64 = 648_000.0 / PI;
const DEGREE: f64 = PI / 180.0;
const AM: f64 = 0.074_801_329_518;
const ALPHA: f64 = 0.002_571_881_335;
/// A Moon term stays when its amplitude times 40^power (arcseconds, or km
/// for the distance) reaches this value.
pub const MOON_LIMIT: f64 = 0.005;
/// A VSOP87B term stays when its amplitude times 4^p (radians or AU)
/// reaches this value, where p is the power of the time in millennia.
pub const VSOP_LIMIT: f64 = 5e-8;
/// The span of the tables, in Julian centuries from J2000.
pub const SPAN_CENTURIES: f64 = 40.0;
/// The order of the fitted correction polynomials.
pub const ORDER: usize = 3;
/// The names of the Moon corrections in a fit file.
pub const MOON_NAMES: [&str; 5] = ["W1", "W2", "W3", "EART", "PERI"];

/// An error that stops a table or fit tool.
pub type Error = Box<dyn std::error::Error>;

/// Corrections in arcseconds by power of TDB Julian centuries from J2000.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Fit {
    /// W1, W2, W3, the mean longitude of the Earth-Moon barycentre, and its
    /// perihelion, in the order of ELP2000-82B.
    pub moon: [[f64; ORDER + 1]; 5],
    /// The VSOP87B longitude.
    pub sun: [f64; ORDER + 1],
}

/// Mean longitudes, Delaunay arguments, and planetary arguments, in
/// radians by power of t.
pub struct Elements {
    /// W1, W2, and W3.
    pub w: [[f64; 5]; 3],
    delaunay: [[f64; 5]; 4],
    zeta: [f64; 5],
    planets: [[f64; 5]; 8],
}

fn dms(degrees: f64, minutes: f64, seconds: f64) -> f64 {
    (degrees + minutes / 60.0 + seconds / 3600.0) * DEGREE
}

fn arcseconds(values: [f64; 5]) -> [f64; 5] {
    values.map(|value| value / ARCSECONDS_PER_RADIAN)
}

fn corrected(mut theory: [f64; 5], correction: &[f64; ORDER + 1]) -> [f64; 5] {
    for (value, correction) in theory.iter_mut().zip(correction) {
        *value += correction / ARCSECONDS_PER_RADIAN;
    }
    theory
}

impl Elements {
    /// The arguments with the corrections of `fit`.
    pub fn new(fit: &Fit) -> Self {
        let with_constant = |constant: f64, rest: [f64; 5]| {
            let mut values = rest;
            values[0] = constant;
            values
        };
        let theory_w = [
            with_constant(
                dms(218.0, 18.0, 59.955_71),
                arcseconds([0.0, 1_732_559_343.736_04, -5.8883, 0.6604e-2, -0.3169e-4]),
            ),
            with_constant(
                dms(83.0, 21.0, 11.674_75),
                arcseconds([0.0, 14_643_420.263_2, -38.2776, -0.45047e-1, 0.21301e-3]),
            ),
            with_constant(
                dms(125.0, 2.0, 40.398_16),
                arcseconds([0.0, -6_967_919.362_2, 6.3622, 0.7625e-2, -0.3586e-4]),
            ),
        ];
        let earth = with_constant(
            dms(100.0, 27.0, 59.220_59),
            arcseconds([0.0, 129_597_742.275_8, -0.0202, 0.9e-5, 0.15e-6]),
        );
        let perihelion = with_constant(
            dms(102.0, 56.0, 14.427_53),
            arcseconds([0.0, 1161.2283, 0.5327, -0.138e-3, 0.0]),
        );
        let w = [0, 1, 2].map(|i| corrected(theory_w[i], &fit.moon[i]));
        let earth = corrected(earth, &fit.moon[3]);
        let perihelion = corrected(perihelion, &fit.moon[4]);
        let mut delaunay = [[0.0; 5]; 4];
        for k in 0..5 {
            delaunay[0][k] = w[0][k] - earth[k];
            delaunay[3][k] = w[0][k] - w[2][k];
            delaunay[2][k] = w[0][k] - w[1][k];
            delaunay[1][k] = earth[k] - perihelion[k];
        }
        delaunay[0][0] += PI;
        let precession = 5029.0966 / ARCSECONDS_PER_RADIAN;
        let zeta = [w[0][0], w[0][1] + precession, w[0][2], w[0][3], w[0][4]];
        let planet = |degrees: [f64; 3], rate: f64| {
            [
                dms(degrees[0], degrees[1], degrees[2]),
                rate / ARCSECONDS_PER_RADIAN,
                0.0,
                0.0,
                0.0,
            ]
        };
        let planets = [
            planet([252.0, 15.0, 3.259_86], 538_101_628.688_98),
            planet([181.0, 58.0, 47.283_05], 210_664_136.433_55),
            earth,
            planet([355.0, 25.0, 59.788_66], 68_905_077.592_84),
            planet([34.0, 21.0, 5.342_12], 10_925_660.428_61),
            planet([50.0, 4.0, 38.896_94], 4_399_609.659_32),
            planet([314.0, 3.0, 18.018_41], 1_542_481.193_93),
            planet([304.0, 20.0, 55.195_75], 786_550.320_74),
        ];
        Self {
            w,
            delaunay,
            zeta,
            planets,
        }
    }
}

/// One line of an ELP2000-82B file, before the arguments enter.
enum Line {
    Main {
        ilu: [i32; 4],
        amplitude: f64,
    },
    Solar {
        ints: [i32; 5],
        phase: f64,
        amplitude: f64,
    },
    Planetary {
        ipla: [i32; 11],
        phase: f64,
        amplitude: f64,
    },
}

/// The 36 files of ELP2000-82B, parsed once.
pub struct Elp {
    files: Vec<(usize, Vec<Line>)>,
}

fn column(line: &str, start: usize, end: usize) -> &str {
    line.get(start..end.min(line.len())).unwrap_or("").trim()
}

fn integers<const N: usize>(line: &str) -> Result<[i32; N], Error> {
    let mut values = [0; N];
    for (i, value) in values.iter_mut().enumerate() {
        *value = column(line, 3 * i, 3 * i + 3).parse()?;
    }
    Ok(values)
}

impl Elp {
    /// Read ELP1 to ELP36 from `directory`.
    pub fn read(directory: &Path) -> Result<Self, Error> {
        let delnu = 0.55604 / 1_732_559_343.736_04;
        let (dele, delg) = (
            0.01789 / ARCSECONDS_PER_RADIAN,
            -0.08066 / ARCSECONDS_PER_RADIAN,
        );
        let delnp = -0.06424 / 1_732_559_343.736_04;
        let delep = -0.12879 / ARCSECONDS_PER_RADIAN;
        let dtasm = 2.0 * ALPHA / (3.0 * AM);
        let mut files = Vec::new();
        for ific in 1..=36 {
            let path = directory.join(format!("ELP{ific}"));
            let text = std::fs::read_to_string(&path)
                .map_err(|error| format!("{}: {error}", path.display()))?;
            let mut lines = Vec::new();
            for line in text.lines().skip(1).filter(|line| !line.trim().is_empty()) {
                lines.push(if ific <= 3 {
                    let ilu = integers::<4>(line)?;
                    let mut coefficient = [0.0; 7];
                    coefficient[0] = column(line, 14, 27).parse()?;
                    for k in 0..6 {
                        coefficient[k + 1] = column(line, 27 + 12 * k, 39 + 12 * k).parse()?;
                    }
                    let c0 = if ific == 3 {
                        coefficient[0] - 2.0 * coefficient[0] * delnu / 3.0
                    } else {
                        coefficient[0]
                    };
                    let tgv = coefficient[1] + dtasm * coefficient[5];
                    let amplitude = c0
                        + tgv * (delnp - AM * delnu)
                        + coefficient[2] * delg
                        + coefficient[3] * dele
                        + coefficient[4] * delep;
                    Line::Main { ilu, amplitude }
                } else if ific <= 9 || ific >= 22 {
                    Line::Solar {
                        ints: integers::<5>(line)?,
                        phase: column(line, 16, 25).parse::<f64>()? * DEGREE,
                        amplitude: column(line, 26, 35).parse()?,
                    }
                } else {
                    Line::Planetary {
                        ipla: integers::<11>(line)?,
                        phase: column(line, 34, 43).parse::<f64>()? * DEGREE,
                        amplitude: column(line, 44, 53).parse()?,
                    }
                });
            }
            files.push((ific, lines));
        }
        Ok(Self { files })
    }

    /// The terms with the arguments of `elements`.
    pub fn terms(&self, elements: &Elements) -> MoonTerms {
        let (delaunay, zeta, planets) = (&elements.delaunay, &elements.zeta, &elements.planets);
        let mut terms = MoonTerms::default();
        for (ific, lines) in &self.files {
            let variable = (ific - 1) % 3;
            let table = ific.div_ceil(3);
            let power = match table {
                3 | 5 | 7 | 9 => 1,
                12 => 2,
                _ => 0,
            };
            let reach = SPAN_CENTURIES.powi(power);
            for line in lines {
                match line {
                    Line::Main { ilu, amplitude } => {
                        let mut phase: [f64; 5] = std::array::from_fn(|k| {
                            (0..4).map(|i| f64::from(ilu[i]) * delaunay[i][k]).sum()
                        });
                        if variable == 2 {
                            phase[0] += PI / 2.0;
                        }
                        if amplitude.abs() >= MOON_LIMIT {
                            terms.main[variable].push((*amplitude, phase));
                        }
                    }
                    Line::Solar {
                        ints,
                        phase,
                        amplitude,
                    } => {
                        let mut polynomial: [f64; 5] = std::array::from_fn(|k| {
                            f64::from(ints[0]) * zeta[k]
                                + (0..4)
                                    .map(|i| f64::from(ints[1 + i]) * delaunay[i][k])
                                    .sum::<f64>()
                        });
                        polynomial[0] += phase;
                        if amplitude.abs() * reach >= MOON_LIMIT {
                            terms.perturbation[variable].push((*amplitude, power, polynomial));
                        }
                    }
                    Line::Planetary {
                        ipla,
                        phase,
                        amplitude,
                    } => {
                        let mut polynomial: [f64; 5] = std::array::from_fn(|k| {
                            let d = |i: usize| delaunay[i][k];
                            let p = |i: usize| f64::from(ipla[i]) * planets[i][k];
                            if *ific < 16 {
                                f64::from(ipla[8]) * d(0)
                                    + f64::from(ipla[9]) * d(2)
                                    + f64::from(ipla[10]) * d(3)
                                    + (0..8).map(p).sum::<f64>()
                            } else {
                                (0..4).map(|i| f64::from(ipla[i + 7]) * d(i)).sum::<f64>()
                                    + (0..7).map(p).sum::<f64>()
                            }
                        });
                        polynomial[0] += phase;
                        if amplitude.abs() * reach >= MOON_LIMIT {
                            terms.perturbation[variable].push((*amplitude, power, polynomial));
                        }
                    }
                }
            }
        }
        terms
    }
}

/// Main problem terms (amplitude, phase by power of t) and perturbations
/// (amplitude, power of t, phase by power of t), for longitude, latitude,
/// and distance. Amplitudes are in arcseconds, or km for the distance.
#[derive(Default)]
pub struct MoonTerms {
    /// The main problem.
    pub main: [Vec<(f64, [f64; 5])>; 3],
    /// The perturbations.
    pub perturbation: [Vec<(f64, i32, [f64; 5])>; 3],
}

/// VSOP87B terms (amplitude, phase, frequency) by variable (1 longitude, 2
/// latitude, 3 distance) and power of the time in millennia.
pub type Vsop = BTreeMap<(u32, u32), Vec<(f64, f64, f64)>>;

/// Read `VSOP87B.ear` and keep the terms that reach [`VSOP_LIMIT`].
pub fn vsop(path: &Path) -> Result<Vsop, Error> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut series = Vsop::new();
    let mut key = None;
    for line in text.lines() {
        if line.starts_with(" VSOP87") {
            let digit = |index: usize| {
                line.get(index..=index)
                    .and_then(|digit| digit.parse::<u32>().ok())
                    .ok_or_else(|| format!("{}: bad header {line}", path.display()))
            };
            key = Some((digit(41)?, digit(59)?));
            continue;
        }
        let Some(key) = key else {
            return Err(format!("{}: a term comes before its header", path.display()).into());
        };
        let fields: Vec<&str> = line.split_whitespace().collect();
        let [.., amplitude, phase, frequency] = fields.as_slice() else {
            return Err(format!("{}: short line {line}", path.display()).into());
        };
        let term: (f64, f64, f64) = (amplitude.parse()?, phase.parse()?, frequency.parse()?);
        if term.0 * 4.0_f64.powi(i32::try_from(key.1)?) >= VSOP_LIMIT || term.2 == 0.0 {
            series.entry(key).or_default().push(term);
        }
    }
    Ok(series)
}
