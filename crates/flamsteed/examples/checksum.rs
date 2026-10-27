//! Print the checksum that `qemu-riscv` prints, computed on the host.
//!
//! ```text
//! cargo run -p flamsteed --example checksum
//! ```

use flamsteed::Catalog;
use std::io::Write as _;

/// Julian years from J2000 for the checksum: 331 BC.
const YEARS: f64 = -2330.0;

fn main() -> std::io::Result<()> {
    let catalog = Catalog::bright_star_catalogue();
    let checksum: f64 = catalog
        .stars()
        .map(|star| star.direction_at(YEARS)[0])
        .sum();
    let checksum32: f32 = catalog
        .brighter_than(6.5)
        .motions::<f32>()
        .map(|motion| motion.direction_at(YEARS as f32)[0])
        .sum();
    writeln!(
        std::io::stdout(),
        "checksum f64 {checksum:.9} f32 {checksum32:.6}"
    )
}
