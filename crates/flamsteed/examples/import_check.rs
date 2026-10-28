//! Read a catalogue file with `Catalog::from_bytes` and compare it with the
//! embedded table.
//!
//! ```text
//! cargo run -p flamsteed --example import_check -- catalog.bin
//! ```
//!
//! The tool writes the format, the axes, the epoch, and the number of stars
//! of each provenance. For each star that is also in the embedded table it
//! measures the angle between the two directions at the file's epoch, which
//! checks the positions, and ten years later, which checks the motions. The
//! two tables can differ by the rotation between their axes, less than 0.1
//! arcsecond, and by the errors of the embedded table.

use std::{collections::BTreeMap, io::Write as _};

use flamsteed::{Astrometry, Catalog, Star};

type Error = Box<dyn std::error::Error>;

const ARCSECONDS_PER_RADIAN: f64 = 648_000.0 / std::f64::consts::PI;

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path] = args.as_slice() else {
        return Err("usage: import_check <catalogue file>".into());
    };
    let bytes = std::fs::read(path).map_err(|error| format!("{path}: {error}"))?;
    let imported = Catalog::from_bytes(&bytes)?;
    let embedded: BTreeMap<Option<u16>, Star> = Catalog::bright_star_catalogue()
        .stars()
        .map(|star| (star.hr(), star))
        .collect();
    let epoch = imported
        .stars()
        .next()
        .map_or(0.0, |s| s.reference_epoch_years);

    let mut out = std::io::stdout().lock();
    writeln!(
        out,
        "{path}: {} stars, axes {:?}, epoch {epoch} years from J2000",
        imported.len(),
        imported.axes()
    )?;
    let mut groups: BTreeMap<String, [Vec<f64>; 2]> = BTreeMap::new();
    for star in imported.stars() {
        let name = match star.astrometry {
            Astrometry::BrightStarCatalogue => "provenance 1 (Bright Star Catalogue)".to_owned(),
            Astrometry::Other { provenance, .. } => format!("provenance {provenance}"),
            _ => "other".to_owned(),
        };
        let group = groups.entry(name).or_default();
        let Some(reference) = embedded.get(&star.hr()) else {
            continue;
        };
        for (errors, years) in group.iter_mut().zip([epoch, epoch + 10.0]) {
            errors.push(separation(
                star.direction_at(years),
                reference.direction_at(years),
            ));
        }
    }
    for (name, [now, later]) in &mut groups {
        writeln!(
            out,
            "{name}: {} matched; difference from the embedded table at the epoch {}; ten years later {}",
            now.len(),
            summary(now),
            summary(later)
        )?;
    }
    Ok(())
}

fn separation(a: [f64; 3], b: [f64; 3]) -> f64 {
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let sine = cross.iter().map(|c| c * c).sum::<f64>().sqrt();
    let cosine: f64 = a.iter().zip(&b).map(|(a, b)| a * b).sum();
    sine.atan2(cosine) * ARCSECONDS_PER_RADIAN
}

/// Median, 99th percentile, and largest value, in arcseconds.
fn summary(values: &mut [f64]) -> String {
    values.sort_by(f64::total_cmp);
    let at = |fraction: f64| {
        let index = ((values.len() as f64 - 1.0) * fraction).round() as usize;
        values.get(index).copied().unwrap_or(f64::NAN)
    };
    format!(
        "median {:.3}\", 99% {:.3}\", largest {:.3}\"",
        at(0.5),
        at(0.99),
        at(1.0)
    )
}
