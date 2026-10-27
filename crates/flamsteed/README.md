# flamsteed

Bright stars and their motion in space, from public-domain data, as free
software.

flamsteed gives the direction of each naked-eye star at an epoch that you
supply. The tests cover -2000 to 3000. It embeds the Yale Bright Star
Catalogue: 9096 stars, about every star to visual magnitude 6.5. It reads no
file and no clock, and it builds for `wasm32-unknown-unknown`.

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
- **No ESA dependency.** The crate does not contain ESA data. Its tests and
  the steps that make its tables do not use ESA data. ESA catalogues serve
  only for offline comparison, and this README gives those comparisons as
  measured differences.
- **Measured claims.** Each accuracy statement comes from a test or a
  measurement that this repository can repeat.
- **No I/O.** The crate has no dependencies, reads no file, and reads no
  clock. You supply the epoch.

## Use

```rust
use flamsteed::Catalog;

// Julian years (TT) from J2000: -2330.0 is 331 BC.
let bright: Vec<[f64; 3]> = Catalog::bright_star_catalogue()
    .brighter_than(2.0)
    .map(|star| star.direction_at(-2330.0))
    .collect();
```

Each direction is a unit vector in J2000 equatorial axes, from the solar
system barycentre. The axes are those of FK5, which differ from ICRS by less
than 0.1 arcsecond. For an observer on the
Earth, apply aberration and Earth axes. The `navigate-celestial` crate does
these steps.

The catalogue keeps the stars in magnitude order, brightest first. Thus
`brighter_than` reads only the stars that it returns. For many epochs, call
`star.motion()` once: `Motion::direction_at` then needs no trigonometry,
about 1.5 ns for each star on a laptop CPU. Each `Star` also gives
its identifier, V magnitude, B-V colour, reference position and epoch,
proper motion, parallax, and radial velocity. A value that the catalogue does
not give is `None`.

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

The catalogue proper motions limit the accuracy far from J2000. This table
gives the measured differences between flamsteed and the Hipparcos
catalogue, for the stars to magnitude 6.5, with the same model for both:

| Epoch | Median | 99 percent |
|---|---|---|
| 1700 | 1.4 arcsec | 9 arcsec |
| 1000 | 5 arcsec | 34 arcsec |
| -330 | 13 arcsec | 81 arcsec |
| -2000 | 23 arcsec | 141 arcsec |

Hipparcos is not the truth for long spans. Its proper motions come from three
years of observations, and the orbits of binary stars change them. The model
does not include the orbits of binary stars. Close binaries, for example
Alpha Centauri, can differ by some arcminutes in the past.

## Comparison with ESA catalogues

| Property | flamsteed | Hipparcos, Gaia DR3 |
|---|---|---|
| Licence of the data | public domain | CC BY-NC 3.0 IGO; commercial use needs ESA permission |
| Bright stars | all naked-eye stars | Gaia DR3 does not have about 20 percent of the stars brighter than G = 3, for example Polaris |
| Proper motions | differ from Hipparcos by a median of 2.6 mas/yr (FK5 stars) to 5.7 mas/yr (all stars), measured | formal errors of about 1 mas/yr (Hipparcos) or less (Gaia) |
| Package | embedded, no I/O, wasm32 | archive files or queries |

The ESA catalogues have more precise proper motions. flamsteed gives a free
licence, every bright star, and a small package with no I/O.

## Data

The data comes from CDS catalogue V/50, the Bright Star Catalogue, 5th
revised edition (Hoffleit and Warren 1991). The NASA Astronomical Data Center
prepared it. The `bsc5_table` example makes `data/bsc5.bin` from the CDS
file and records its SHA-256. The example also gives the 22-byte record
layout.

## Licence

The code is under the GNU Affero General Public License, version 3 or later.
Refer to `LICENSE`. The catalogue data is in the public domain.
