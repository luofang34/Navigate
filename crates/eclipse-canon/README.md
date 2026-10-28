# eclipse-canon

The solar and lunar eclipses of the Five Millennium Canon, from -1999 to
3000, as embedded data.

The crate embeds the two catalogues of the *Five Millennium Canon of Solar
Eclipses* and the *Five Millennium Canon of Lunar Eclipses* (Espenak and
Meeus): 11,898 solar eclipses and 12,064 lunar eclipses, with every column
of the catalogue pages. It reads no file and no clock. It does not allocate,
and it has no dependency. It builds without the standard library.

Eclipse Predictions by Fred Espenak (NASA's GSFC).

## Use

```rust
use eclipse_canon::{SolarKind, solar_eclipses};

for eclipse in solar_eclipses().filter(|e| e.kind == SolarKind::Total) {
    let jd_tt = eclipse.greatest_eclipse_jd_tt();
    // ...
}
```

`solar_eclipses()` and `lunar_eclipses()` give the rows in date order.
`SolarEclipse::by_number` and `LunarEclipse::by_number` give one row by its
catalogue number.

## Times and dates

- Times are TD, which is TT. Each row gives the Delta T that the catalogue
  used. To compare the geometry with another ephemeris, use the TD of
  greatest eclipse. A different model of Delta T changes only UT and the
  longitudes.
- Dates are in the Julian calendar before 1582 October 15 and in the
  Gregorian calendar from that day. Years are astronomical: year 0 is 1 BCE.
- The values have the precision of the catalogue pages: gamma and the
  magnitudes to 0.0001, positions and altitudes to 1 degree, and times to
  1 second.

## Data

`data/solar.bin` and `data/lunar.bin` come from the century pages of
<https://eclipse.gsfc.nasa.gov/SEcat5/> and
<https://eclipse.gsfc.nasa.gov/LEcat5/>. To make them again, download the
100 pages into one directory and run:

```text
cargo run -p eclipse-canon --example canon_tables -- PAGES crates/eclipse-canon/data
```

The record layouts are in `src/solar.rs` and `src/lunar.rs`.

## Licence

The code is under the GNU Affero General Public License, version 3 or later.
NASA gives permission to reproduce the eclipse data when it is accompanied
by the acknowledgment "Eclipse Predictions by Fred Espenak (NASA's GSFC)".
