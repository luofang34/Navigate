//! Write flamsteed's direction of every embedded star at the epochs given, as
//! the black-box output for `verify/compare.py`.
//!
//! ```text
//! cargo run -p flamsteed --example directions -- -4000 -2330 -1000 -300 26 > directions.txt
//! ```
//!
//! The arguments are Julian years (TT) from J2000. Each line gives the HR
//! number, the years, and the unit direction.

use flamsteed::Catalog;
use std::io::{BufWriter, Write as _};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let epochs = std::env::args()
        .skip(1)
        .map(|text| text.parse::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    let mut out = BufWriter::new(std::io::stdout().lock());
    for star in Catalog::bright_star_catalogue().stars() {
        let Some(hr) = star.hr() else { continue };
        for &years in &epochs {
            let [x, y, z] = star.direction_at(years);
            writeln!(out, "{hr} {years:?} {x:?} {y:?} {z:?}")?;
        }
    }
    out.flush()?;
    Ok(())
}
