# flamsteed

Bright stars and their motion in space, from public-domain data, as free
software.

flamsteed gives the direction of each naked-eye star at an epoch that you
supply. The tests cover -2000 to 3000. It embeds the Yale Bright Star
Catalogue: 9096 stars, about every star to visual magnitude 6.5. It reads no
file and no clock, and it does not allocate. It builds for
`wasm32-unknown-unknown` and, without the standard library, for
microcontrollers such as the RP2350.

The crate has its name from John Flamsteed (1646-1719), the first Astronomer
Royal. He measured the positions of about 3000 stars. In 1712 his data was
printed without his consent, and he burned the copies that he could get.
His own catalogue, the *Historia Coelestis Britannica*, came out after his
death. This crate keeps his data principle: the data and the method are open,
and each claim about accuracy has a measurement.

## Principles

- **Free software.** The code is under the GNU AGPL, version 3 or later.
- **Free data.** The catalogue is a United States government work in the
  public domain. The crate does not contain data with a non-commercial
  clause.
- **Measured claims.** Each accuracy statement comes from a test or a
  measurement that this repository can repeat.
- **No I/O.** The crate reads no file and no clock, and it does not
  allocate. You supply the epoch.
- **Small dependencies.** With the standard library the crate has no
  dependencies. Without it, `libm` (MIT) gives the float functions.

## Use

```rust
use flamsteed::Catalog;

// Julian years (TT) from J2000: -2330.0 is 331 BC.
let bright: Vec<[f64; 3]> = Catalog::bright_star_catalogue()
    .brighter_than(2.0)
    .stars()
    .map(|star| star.direction_at(-2330.0))
    .collect();
```

Each direction is a unit vector from the solar system barycentre. For the
embedded catalogue it is in J2000 equatorial axes as FK5 gives them, which
differ from ICRS by less than 0.1 arcsecond; `Catalog::axes` gives the axes
of a catalogue. For an observer on the Earth, apply aberration and Earth
axes. The `navigate-celestial` crate does these steps.

The catalogue keeps the stars in magnitude order, brightest first.
`brighter_than` gives a smaller catalogue, a prefix, and reads only the stars
that it keeps. Each `Star` gives its identifier, V magnitude, B-V colour,
position and velocity at the reference epoch, parallax, and radial velocity.
A value that the catalogue does not give is `None`. The right ascension,
declination, and proper motion are methods.

`Star::barycentric_position_pc_at` gives the position in parsecs from the
solar system barycentre, for an observer away from the Earth or for a
parallax calculation. It needs a trigonometric parallax, which the catalogue
gives for 2652 of the stars to magnitude 6.5. The positions agree with
Skyfield to a relative 1e-8 from -2000 to 3000.

## Speed and precision

The embedded table stores each star as a J2000 unit vector and a space
velocity in fixed point. Thus no trigonometry runs when the crate reads a star or moves
it. Only the binary orbits of `Star::component_direction_at` use a sine and a
cosine. `Motion` holds the vectors in `f64` or `f32`:

- `f64` keeps the precision of the table. Format version 1 stores a position
  to 0.1 milliarcsecond, a velocity to 0.0002 milliarcsecond per year, and a
  parallax to 1 milliarcsecond.
- `f32` keeps a direction within 0.03 arcsecond of `f64` for every star from
  -4000 to 1000 years from J2000 (measured maximum 0.021 arcsecond). A
  single-precision FPU, for example the Cortex-M33 FPU of the RP2350,
  calculates it natively. `catalog.motions::<f32>()` reads the embedded table
  to `f32` with no double-precision operation.

`Motion::direction_at` needs one square root. `Motion::position_at` gives the
direction before normalization, with only multiplications and additions, for
a GPU or a microcontroller that normalizes itself.

On stable Rust, `core` has no square root, so a build without the standard
library takes it from `libm`, which calculates it in software on Cortex-M.
For a microcontroller with an FPU, use a nightly compiler and the `nightly`
feature: the `f32` square root is then the hardware instruction, for example
`vsqrt.f32` on the Cortex-M33. On the QEMU Cortex-M33 this reduces the `f32`
read and `direction_at` from 160 to 100 instructions, and the prepared
`direction_at` from 97 to 35. CI checks the instruction in the
disassembly.

These are measured costs for each star:

| Path | Laptop CPU (Apple silicon) | Cortex-M33 with FPU | RV32IMAC, no FPU |
|---|---|---|---|
| read and `direction_at`, `f64` | 2.1 ns | 1,939 instructions | 3,011 instructions |
| `motions::<f32>()` and `direction_at` | 1.7 ns | 160 instructions | 1,973 instructions |
| prepared `Motion::direction_at` (`f64` on the laptop, `f32` on the microcontrollers) | 1.6 ns | 97 instructions | 1,202 instructions |

`scripts/check-flamsteed-qemu.sh` gives the microcontroller values. It runs
the crate in QEMU on a Cortex-M33 (`mps2-an505`, as the RP2350 Arm cores)
and on an RV32IMAC core (as the RP2350 RISC-V cores), and it compares each
checksum with the host. QEMU counts instructions, not clock cycles. CI builds
the crate for the RP2040 (Cortex-M0+, no FPU) but does not run it there; the
RV32IMAC column, which also has no FPU, is the nearest measurement.

A prepared `Motion<f32>` uses 28 bytes of RAM and a `Motion<f64>` 56 bytes.
All 9096 stars in `f32` use 255 KB.

## Targets and features

| Feature | Effect |
|---|---|
| `std` (default) | float functions from the standard library |
| `libm` | float functions from `libm`, for targets without the standard library |
| `nightly` | square roots from `core` on a nightly compiler, which use the hardware instruction |
| `all-stars` (default) | embed all 9096 stars: 309 KB |
| `magnitude-5` | embed the 1630 stars to magnitude 5.0: 55 KB |
| neither | embed the 518 stars to magnitude 4.0: 18 KB |

For a microcontroller, use `default-features = false, features = ["libm"]`
and add a magnitude feature if the flash has space. Cargo unifies features
across the dependency graph: if another crate depends on flamsteed with its
default features, the full table comes back. CI builds the crate for
`thumbv8m.main-none-eabihf` (RP2350 Arm cores),
`riscv32imac-unknown-none-elf` (RP2350 RISC-V cores), and
`thumbv6m-none-eabi` (RP2040). The code is about 32 KB on RV32IMAC, with the
QEMU runtime.

The build script checks the table and its CRC-32, and a constant check in
the crate rejects a bad header, so a damaged table stops the build.

## Model

Each star moves on a straight line in space from its J2000 position. The
motion has two parts: the proper motion, and the radial velocity where the
catalogue gives a trigonometric parallax. The light-time factor
1 / (1 - v_r / c) turns the observed rates into space motion, as in SOFA
`eraStarpv`. A star without a parallax has no radial term.

The tests compare 40 stars at five epochs from -2000 to 3000 with a
reference that agrees with Skyfield to 0.002 arcsecond. The difference from
the reference is less than 0.01 arcsecond.

## Accuracy

The catalogue proper motions limit the accuracy far from J2000: an error of
a few milliarcseconds per year becomes some arcseconds after 1000 years and
some tens of arcseconds after 4000. The orbits of binary stars also bend the
straight line; the next section treats them. For more precise positions and
motions, import a catalogue file in format version 2 (see Data).

## Binary stars

The catalogue proper motion of a star in a binary is the motion of the
system. `Star::direction_at` follows the system. `Star::component_direction_at`
adds the orbital motion of the star itself, for the 133 stars in binaries
with an ORB6 orbit of grade 1 to 3 that moves them by more than 0.1
arcsecond. The offset is zero at J2000. From -2000 to 3000, the largest
offsets are 11.0 arcseconds for Alpha Centauri A, 13.1 for Alpha Centauri B,
4.1 for Sirius A, and 2.3 for Procyon A.

The relative orbits agree with the USNO ORB6 ephemerides for 2025 to 2029
within two times their rounding. The share of each star in the relative
orbit needs the mass ratio, which ORB6 does not give:

- For Alpha Centauri, Sirius, and Procyon, the share comes from published
  dynamical masses.
- For the other systems, the share is a model, not a measurement: a
  main-sequence mass-luminosity relation applied to the magnitude difference.
  For Alpha Centauri this model gives a mass fraction of 0.42, not 0.46. For
  a giant or a white-dwarf companion it is wrong: it gives 0.09, not 0.33, for
  Sirius.

An orbit has an epoch range. Far from its observations, an error in the
period moves the star to a different phase on its orbit, but the offset
stays within the size of the star's orbit.

The FK5 axes of the embedded catalogue also turn slowly relative to ICRS, by
up to about 0.9 milliarcsecond per year (Mignard and Froeschle 2000). The
crate does not correct this rotation. At -2000 it is about 3.6 arcseconds.

The orbit offsets apply only to stars whose astrometry is from the Bright
Star Catalogue. The position of a star in another catalogue already includes
its orbital motion.

## Data

The data comes from CDS catalogue V/50, the Bright Star Catalogue, 5th
revised edition (Hoffleit and Warren 1991). The NASA Astronomical Data Center
prepared it. The `bsc5_table` example makes `data/bsc5.bin` from the CDS
file and records its SHA-256. The example also gives the 34-byte record
layout.

The crate embeds this catalogue by default. To use other catalogue data,
supply the bytes of a catalogue file to `Catalog::from_bytes`. The catalogue
borrows the bytes and does not copy them, so they can be in a file, in flash,
or in a memory map:

```rust
use flamsteed::{Catalog, FormatError};

/// The number of stars brighter than magnitude 3 in a catalogue file.
fn bright_stars(bytes: &[u8]) -> Result<usize, FormatError> {
    Ok(Catalog::from_bytes(bytes)?.brighter_than(3.0).len())
}
```

A file has a 32-byte header and records in magnitude order, brightest first.
The function checks the header, the length, the CRC-32 of the records, the
reference epoch, and the magnitude order, and it reads two formats:

| Format | Records | Axes | Position and velocity |
|---|---|---|---|
| 1 | 34 bytes, the embedded format | FK5 J2000 | fixed point |
| 2 | 72 bytes | ICRS | `f64`, with the provenance and the source identifier of each star |

Both formats identify each star by its HR number. In format 2, each record
gives the catalogue of its astrometry as `Star::astrometry`, and the reference
epoch is that of the file. `src/catalog/v2.rs` documents the layout. The
`import_check` example reads a file and compares each star with the embedded
table. A later format version can add other identifiers; this release rejects
such a file with `FormatError::Version` or `FormatError::Source`.

## Licence

The code is under the GNU Affero General Public License, version 3 or later.
Refer to `LICENSE`. The catalogue data is in the public domain.
