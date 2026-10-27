# flamsteed

Bright stars and their motion in space, from public-domain catalogue data.

The crate embeds the Yale Bright Star Catalogue, 5th revised edition: 9096
stars, which is about every star to visual magnitude 6.5. It moves each star
on a straight line in space to an epoch that you supply. It reads no file and
no clock, and it builds for `wasm32-unknown-unknown`.

## Use

```rust
use flamsteed::Catalog;

// Julian years (TT) from J2000: -2330.0 is 331 BC.
let bright: Vec<[f64; 3]> = Catalog::bright_star_catalogue()
    .stars()
    .filter(|star| star.v_magnitude <= 2.0)
    .map(|star| star.direction_at(-2330.0))
    .collect();
```

Each direction is a unit vector in J2000 equatorial axes, from the solar
system barycentre. For an observer, apply aberration and Earth axes.

## Model

Each star moves on a straight line in space from its J2000 position. The
motion has two parts: the proper motion, and the radial velocity where the
catalogue gives a trigonometric parallax. The light-time factor
1 / (1 - v_r / c) turns the observed rates into space motion, as in SOFA
`eraStarpv`. The tests compare the motion with Skyfield from -2000 to 3000.
The difference is less than 0.01 arcsecond.

## Accuracy

The catalogue proper motions limit the accuracy far from J2000. These are
measured differences between this crate and the Hipparcos catalogue, for the
stars to magnitude 6.5, with the same model for both:

| Epoch | Median | 99 percent |
|---|---|---|
| 1700 | 1.4 arcsec | 9 arcsec |
| 1000 | 5 arcsec | 34 arcsec |
| -330 | 13 arcsec | 81 arcsec |
| -2000 | 23 arcsec | 141 arcsec |

Hipparcos is not the truth for long spans. Its proper motions come from three
years of observations. The orbits of binary stars are not in the model. Close
binaries, for example Alpha Centauri, can differ by some arcminutes in the
past.

## Data and licence

The data comes from CDS catalogue V/50 (Hoffleit and Warren 1991). The NASA
Astronomical Data Center prepared it. It is a United States government work
and is in the public domain. The `bsc5_table` example makes `data/bsc5.bin`
from the CDS file. The example also gives the record layout.

The crate does not contain ESA data. Its tests and the steps that make its
tables do not use ESA data. ESA catalogues serve only for offline comparison.

The code is under the GNU Affero General Public License, version 3 or later.
Refer to `LICENSE`.
