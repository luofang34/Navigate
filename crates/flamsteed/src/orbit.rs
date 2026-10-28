//! Binary-star orbits from the Sixth Catalog of Orbits of Visual Binary Stars.
//!
//! ORB6 is a USNO catalogue and a United States government work in the public
//! domain. The `orb6_table` example writes `data/orbits.bin` and documents
//! the selection and format version 1: orbits of grade 1 to 3 for the AB
//! pair of a system with a catalogue star, where the star's offset from the
//! system's motion can exceed 0.1 arcsecond.
//!
//! The catalogue proper motions of a binary's components are those of the
//! system: the two components of a resolved pair have nearly the same proper
//! motion, much less than their orbital motion. So the straight-line motion
//! of a star stays as it is, and the orbit adds the periodic offset of the
//! star from it, zero at J2000.

use crate::{Float, Star, float};

const ORBITS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/orbits.bin"));
const MAGIC: &[u8; 8] = b"FLAMORBT";
const HEADER_BYTES: usize = 32;
const RECORD_BYTES: usize = 68;
/// Source 2 is ORB6.
const SOURCE_ORB6: u32 = 2;
const ARCSEC: f64 = core::f64::consts::PI / 648_000.0;

/// The embedded orbit records, checked when the crate compiles.
const RECORDS: Option<&[[u8; RECORD_BYTES]]> = parse(ORBITS);
const _: () = assert!(RECORDS.is_some(), "the embedded orbit table is malformed");

/// The orbit of a catalogue star's binary, and the star's share of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Orbit {
    /// The star's offset as a fraction of the relative orbit of B about A.
    pub(crate) fraction: f64,
    pub(crate) period_years: f64,
    /// Periastron time, in Julian years from J2000.
    pub(crate) periastron_years: f64,
    pub(crate) eccentricity: f64,
    pub(crate) semi_major_arcsec: f64,
    pub(crate) inclination: f64,
    pub(crate) node: f64,
    pub(crate) periastron_argument: f64,
}

impl Orbit {
    /// The position of B relative to A, east and north, in arcseconds.
    pub(crate) fn relative_arcsec(&self, years: f64) -> [f64; 2] {
        let periods = (years - self.periastron_years) / self.period_years;
        // `core` has no `rem_euclid` for floats; the cast truncates.
        let fraction = periods - periods as i64 as f64;
        let mean = core::f64::consts::TAU
            * if fraction < 0.0 {
                fraction + 1.0
            } else {
                fraction
            };
        let e = self.eccentricity;
        let mut anomaly = if e < 0.8 { mean } else { core::f64::consts::PI };
        for _ in 0..50 {
            let (sin, cos) = float::sin_cos(anomaly);
            let step = (anomaly - e * sin - mean) / (1.0 - e * cos);
            anomaly -= step;
            if step.abs() < 1e-14 {
                break;
            }
        }
        let (sin_e, cos_e) = float::sin_cos(anomaly);
        let x = cos_e - e;
        let y = Float::sqrt(1.0 - e * e) * sin_e;
        let (sin_w, cos_w) = float::sin_cos(self.periastron_argument);
        let (sin_n, cos_n) = float::sin_cos(self.node);
        let (_, cos_i) = float::sin_cos(self.inclination);
        let a = self.semi_major_arcsec;
        // Thiele-Innes constants; A and F give north, B and G give east.
        let thiele_a = a * (cos_w * cos_n - sin_w * sin_n * cos_i);
        let thiele_b = a * (cos_w * sin_n + sin_w * cos_n * cos_i);
        let thiele_f = a * (-sin_w * cos_n - cos_w * sin_n * cos_i);
        let thiele_g = a * (-sin_w * sin_n + cos_w * cos_n * cos_i);
        [thiele_b * x + thiele_g * y, thiele_a * x + thiele_f * y]
    }
}

/// The orbit of the star with this HR number, if the table has one.
pub(crate) fn orbit_of(hr: u16) -> Option<Orbit> {
    let records = RECORDS?;
    let index = records
        .binary_search_by_key(&hr, |record| u16::from_le_bytes([record[0], record[1]]))
        .ok()?;
    records.get(index).map(decode)
}

impl Star {
    /// The offset of the star from its straight-line motion caused by its
    /// binary orbit, east and north in radians, zero at J2000. `None` if the
    /// orbit table has no orbit for the star.
    ///
    /// The orbits come from ORB6 (grades 1 to 3). The star's share of the
    /// relative orbit uses a mass fraction from the magnitude difference and a
    /// main-sequence mass-luminosity relation, which is uncertain by about 0.1
    /// of the semi-major axis and wrong for giants.
    pub fn orbital_offset_rad_at(&self, years: f64) -> Option<[f64; 2]> {
        let orbit = orbit_of(self.hr()?)?;
        let [east, north] = orbit.relative_arcsec(years);
        let [east0, north0] = orbit.relative_arcsec(0.0);
        let scale = orbit.fraction * ARCSEC;
        Some([(east - east0) * scale, (north - north0) * scale])
    }

    /// The unit direction of the star itself `years` Julian years (TT) after
    /// J2000: [`Star::direction_at`] plus [`Star::orbital_offset_rad_at`]. A
    /// star without an orbit gives [`Star::direction_at`].
    pub fn component_direction_at(&self, years: f64) -> [f64; 3] {
        let direction = self.direction_at(years);
        let Some([east, north]) = self.orbital_offset_rad_at(years) else {
            return direction;
        };
        let [x, y, z] = direction;
        let rho = Float::sqrt(x * x + y * y);
        if rho == 0.0 {
            return direction;
        }
        let east_axis = [-y / rho, x / rho, 0.0];
        let north_axis = [-z * x / rho, -z * y / rho, rho];
        let moved: [f64; 3] =
            core::array::from_fn(|i| direction[i] + east_axis[i] * east + north_axis[i] * north);
        let norm = Float::sqrt(moved[0] * moved[0] + moved[1] * moved[1] + moved[2] * moved[2]);
        moved.map(|c| c / norm)
    }
}

/// The records of a format version 1 orbit file, if the header and the
/// length agree. The build script checks the CRC-32.
const fn parse(bytes: &[u8]) -> Option<&[[u8; RECORD_BYTES]]> {
    let Some((header, body)) = bytes.split_at_checked(HEADER_BYTES) else {
        return None;
    };
    let mut index = 0;
    while index < MAGIC.len() {
        if header[index] != MAGIC[index] {
            return None;
        }
        index += 1;
    }
    let version = u16::from_le_bytes([header[8], header[9]]);
    let record_bytes = u16::from_le_bytes([header[10], header[11]]) as usize;
    let count = u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;
    let source = u32::from_le_bytes([header[28], header[29], header[30], header[31]]);
    let (records, rest) = body.as_chunks::<RECORD_BYTES>();
    if version != 1
        || record_bytes != RECORD_BYTES
        || source != SOURCE_ORB6
        || !rest.is_empty()
        || records.len() != count
    {
        return None;
    }
    Some(records)
}

fn decode(record: &[u8; RECORD_BYTES]) -> Orbit {
    let value = |index: usize| {
        let start = 4 + 8 * index;
        let mut bytes = [0; 8];
        bytes.copy_from_slice(&record[start..start + 8]);
        f64::from_le_bytes(bytes)
    };
    Orbit {
        fraction: value(0),
        period_years: value(1),
        periastron_years: value(2),
        eccentricity: value(3),
        semi_major_arcsec: value(4),
        inclination: value(5),
        node: value(6),
        periastron_argument: value(7),
    }
}

#[cfg(test)]
mod tests;
