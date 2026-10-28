//! Compute every star direction on a bare-metal Cortex-M33 in QEMU, print a
//! checksum to compare with the host, and count instructions per star.
#![no_std]
#![no_main]

use cortex_m::peripheral::{SYST, syst::SystClkSource};
use flamsteed::{Catalog, Motion};
use panic_halt as _;
use semihosting::println;

/// Julian years from J2000 for the checksum: 331 BC.
const YEARS: f64 = -2330.0;
/// With `-icount shift=0` one instruction takes 1 ns, and the MPS2 SysTick
/// reference clock runs at 25 MHz, so one tick is 40 instructions.
const INSTRUCTIONS_PER_TICK: u64 = 40;

fn instructions(syst: &mut SYST, run: impl FnOnce()) -> u64 {
    syst.set_reload(0x00FF_FFFF);
    syst.clear_current();
    syst.enable_counter();
    run();
    let ticks = 0x00FF_FFFF - SYST::get_current();
    syst.disable_counter();
    u64::from(ticks) * INSTRUCTIONS_PER_TICK
}

/// Compute the checksum and the instruction counts, then exit QEMU.
#[cortex_m_rt::entry]
fn main() -> ! {
    let Some(mut core) = cortex_m::Peripherals::take() else {
        semihosting::process::exit(1)
    };
    core.SYST.set_clock_source(SystClkSource::External);
    let catalog = Catalog::bright_star_catalogue();
    let stars = catalog.len() as u64;

    let mut checksum = 0.0;
    let full = instructions(&mut core.SYST, || {
        checksum = catalog
            .stars()
            .map(|star| star.direction_at(YEARS)[0])
            .sum();
    }) / stars;

    let bright = catalog.brighter_than(6.5);
    let mut checksum32 = 0.0f32;
    let decode = instructions(&mut core.SYST, || {
        checksum32 = bright
            .motions::<f32>()
            .map(|motion| motion.direction_at(YEARS as f32)[0])
            .sum();
    }) / bright.len() as u64;

    // A renderer prepares the motions once and evaluates them for each frame.
    let mut prepared = catalog.motions::<f32>();
    let frame: [Motion<f32>; 1024] = core::array::from_fn(|_| {
        prepared.next().unwrap_or(Motion {
            position: [1.0, 0.0, 0.0],
            velocity: [0.0; 3],
            reference_epoch_years: 0.0,
        })
    });
    let mut sum = 0.0f32;
    let evaluate = instructions(&mut core.SYST, || {
        sum = frame
            .iter()
            .map(|motion| motion.direction_at(YEARS as f32)[0])
            .sum();
    }) / frame.len() as u64;

    println!("checksum f64 {checksum:.9} f32 {checksum32:.6}");
    println!(
        "instructions per star: f64 decode and evaluate {full}, f32 decode and evaluate {decode}, prepared f32 evaluate {evaluate} ({sum:.3})"
    );
    semihosting::process::exit(0)
}
