//! Compute every star direction on a bare-metal RV32IMAC core in QEMU, print
//! a checksum to compare with the host, and count instructions per star.
#![no_std]
#![no_main]

use flamsteed::{Catalog, Motion};
use panic_halt as _;
use semihosting::println;

/// Julian years from J2000 for the checksum: 331 BC.
const YEARS: f64 = -2330.0;

/// Compute the checksum and the instruction counts, then exit QEMU.
#[riscv_rt::entry]
fn main() -> ! {
    let catalog = Catalog::bright_star_catalogue();
    let instructions = riscv::register::minstret::read64;

    let start = instructions();
    let checksum: f64 = catalog
        .stars()
        .map(|star| star.direction_at(YEARS)[0])
        .sum();
    let full = (instructions() - start) / catalog.len() as u64;

    let start = instructions();
    let checksum32: f32 = catalog
        .brighter_than(6.5)
        .motions::<f32>()
        .map(|motion| motion.direction_at(YEARS as f32)[0])
        .sum();
    let prepare = (instructions() - start) / catalog.brighter_than(6.5).len() as u64;

    // A renderer prepares the motions once and evaluates them for each frame.
    let mut prepared = catalog.motions::<f32>();
    let frame: [Motion<f32>; 64] = core::array::from_fn(|_| {
        prepared.next().unwrap_or(Motion {
            position: [1.0, 0.0, 0.0],
            velocity: [0.0; 3],
            reference_epoch_years: 0.0,
        })
    });
    let start = instructions();
    let sum: f32 = frame
        .iter()
        .map(|motion| motion.direction_at(YEARS as f32)[0])
        .sum();
    let evaluate = (instructions() - start) / frame.len() as u64;

    println!("checksum f64 {checksum:.9} f32 {checksum32:.6}");
    println!(
        "instructions per star: f64 decode and evaluate {full}, f32 decode and evaluate {prepare}, prepared f32 evaluate {evaluate} ({sum:.3})"
    );
    semihosting::process::exit(0)
}
