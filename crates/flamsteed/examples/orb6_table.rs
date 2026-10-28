//! Pack the binary-star orbits that move Bright Star Catalogue stars.
//!
//! ```text
//! cargo run -p flamsteed --example orb6_table -- catalog orb6orbits.txt orb6ephem.txt \
//!     crates/flamsteed/data/orbits.bin crates/flamsteed/fixtures/orb6-ephemerides.txt
//! ```
//!
//! The inputs are CDS V/50 `catalog` (SHA-256
//! `69797549cc1605aad7ff94e9325e29a1661f2a253917faaa056d9bf20b809afd`) and the
//! Sixth Catalog of Orbits of Visual Binary Stars (USNO, a United States
//! government work in the public domain), from the mirror
//! `http://www.astro.gsu.edu/wds/orb6/`: `orb6orbits.txt` (SHA-256
//! `ac7db7711d638f4d8bc47f9a876fb7294f25436d36942d74eb023aa42325824c`) and
//! `orb6ephem.txt` (SHA-256
//! `4e7b6111422c7c7d7d5afc49c4de923f1f238a1ed586f551fce6bada095c23da`).
//!
//! The table keeps an orbit of grade 1 to 3 for the AB pair of a system that
//! has a catalogue star, and only where the star's offset from the
//! barycentre can exceed 0.1 arcsecond. The catalogue gives the component
//! (A, B, or the combined light when blank). The star's offset is a fraction
//! of the relative orbit: -m for A, 1 - m for B, and b - m for the combined
//! light, where m is the mass fraction of B and b its light fraction. ORB6
//! has no masses. For the systems in `DYNAMICAL_MASSES`, m comes from
//! published dynamical masses. For the other systems, m comes from the
//! magnitude difference with the main-sequence relation M ~ L^(1/4).
//!
//! The file is format version 1, all little-endian. A 32-byte header gives
//! the magic `FLAMORBT`, the format version (u16), the record size (u16), the
//! record count (u32), eight zero bytes, the CRC-32 (IEEE) of the records
//! (u32), and the source (u32, 2 for ORB6). The 68-byte records follow in HR
//! order: HR number (u16), orbit grade (u8), component (u8: 1 A, 2 B, 3
//! combined), then f64 values: the offset fraction, the period in Julian
//! years, the periastron time in Julian years from J2000, the eccentricity,
//! the semi-major axis in arcseconds, and the inclination, the node, and the
//! argument of periastron in radians. The fixture lists the ORB6 ephemerides
//! (position angle in degrees and separation in arcseconds at 2025.0 to
//! 2029.0) of the kept orbits.

use std::{collections::BTreeMap, f64::consts::PI, path::PathBuf};

const HEADER_BYTES: usize = 32;
const RECORD_BYTES: usize = 68;
const MINIMUM_OFFSET_ARCSEC: f64 = 0.1;
/// Dynamical masses of A and B in solar masses, by the HD number of the
/// orbit. The magnitude relation is wrong for a white-dwarf companion: it
/// gives Sirius a mass fraction of 0.09, not 0.33.
const DYNAMICAL_MASSES: [(u32, f64, f64); 3] = [
    // Alpha Centauri, Akeson et al. 2021.
    (128_620, 1.0788, 0.9092),
    // Sirius, Bond et al. 2017.
    (48_915, 2.063, 1.018),
    // Procyon, Bond et al. 2015.
    (61_421, 1.478, 0.592),
];

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("usage: orb6_table <V/50 catalog> <orb6orbits.txt> <orb6ephem.txt> <table> <fixture>")]
    Arguments,
    #[error("cannot read {}", path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot write {}", path.display())]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot format the fixture")]
    Format(#[from] std::fmt::Error),
}

/// A catalogue star that can belong to a binary.
struct Entry {
    hr: u16,
    hd: Option<u32>,
    ads: Option<u32>,
    component: String,
    v_magnitude: f64,
    dmag: Option<f64>,
    ra: f64,
    dec: f64,
}

/// One ORB6 orbit of an AB pair.
struct Orbit {
    key: String,
    reference: String,
    grade: u8,
    hd: Option<u32>,
    ads: Option<u32>,
    magnitudes: Option<(f64, f64)>,
    period_years: f64,
    periastron_years: f64,
    eccentricity: f64,
    semi_major_arcsec: f64,
    inclination: f64,
    node: f64,
    periastron_argument: f64,
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [catalog, orbits, ephemerides, table_path, fixture_path] = args.as_slice() else {
        return Err(Error::Arguments);
    };
    let read = |path: &String| {
        std::fs::read_to_string(path).map_err(|source| Error::Read {
            path: path.into(),
            source,
        })
    };
    let entries: Vec<Entry> = read(catalog)?.lines().filter_map(entry).collect();
    let mut best: BTreeMap<String, Orbit> = BTreeMap::new();
    for orbit in read(orbits)?.lines().filter_map(orbit) {
        // The best grade wins; the later line wins a tie.
        if best
            .get(&orbit.key)
            .is_none_or(|kept| orbit.grade <= kept.grade)
        {
            best.insert(orbit.key.clone(), orbit);
        }
    }
    // An ephemeris row has the pair, the grade, the reference, then position
    // angle and separation for 2025.0 to 2029.0.
    let ephemeris: BTreeMap<String, Vec<f64>> = read(ephemerides)?
        .lines()
        .filter(|line| line.as_bytes().first().is_some_and(u8::is_ascii_digit))
        .filter_map(|line| {
            let key = format!(
                "{} {}",
                line.get(0..26)?.trim_end(),
                line.get(34..45)?.trim()
            );
            let values: Vec<f64> = line
                .get(45..)?
                .split_whitespace()
                .take(10)
                .map_while(|token| token.parse().ok())
                .collect();
            (values.len() == 10).then_some((key, values))
        })
        .collect();

    let mut records: BTreeMap<u16, (Vec<u8>, Vec<f64>)> = BTreeMap::new();
    for orbit in best.values() {
        for (star, component) in members(&entries, orbit) {
            let Some(delta) = orbit
                .magnitudes
                .map(|(primary, secondary)| secondary - primary)
                .or(star.dmag)
            else {
                continue;
            };
            let light = 1.0 / (1.0 + 10f64.powf(0.4 * delta));
            let ratio = 10f64.powf(-0.1 * delta);
            let mass = DYNAMICAL_MASSES
                .iter()
                .find(|(hd, ..)| orbit.hd == Some(*hd))
                .map_or(ratio / (1.0 + ratio), |(_, a, b)| b / (a + b));
            let fraction = match component {
                1 => -mass,
                2 => 1.0 - mass,
                _ => light - mass,
            };
            let reach = fraction.abs() * orbit.semi_major_arcsec * (1.0 + orbit.eccentricity);
            if reach < MINIMUM_OFFSET_ARCSEC {
                continue;
            }
            let Some(lines) = ephemeris.get(&format!("{} {}", orbit.key, orbit.reference)) else {
                continue;
            };
            let mut record = Vec::with_capacity(RECORD_BYTES);
            record.extend_from_slice(&star.hr.to_le_bytes());
            record.push(orbit.grade);
            record.push(component);
            for value in [
                fraction,
                orbit.period_years,
                orbit.periastron_years,
                orbit.eccentricity,
                orbit.semi_major_arcsec,
                orbit.inclination,
                orbit.node,
                orbit.periastron_argument,
            ] {
                record.extend_from_slice(&value.to_le_bytes());
            }
            records.insert(star.hr, (record, lines.clone()));
        }
    }

    let body: Vec<u8> = records
        .values()
        .flat_map(|(record, _)| record.clone())
        .collect();
    let mut file = Vec::with_capacity(HEADER_BYTES + body.len());
    file.extend_from_slice(b"FLAMORBT");
    file.extend_from_slice(&1_u16.to_le_bytes());
    file.extend_from_slice(&(RECORD_BYTES as u16).to_le_bytes());
    file.extend_from_slice(&(records.len() as u32).to_le_bytes());
    file.extend_from_slice(&[0; 8]);
    file.extend_from_slice(&crc32(&body).to_le_bytes());
    file.extend_from_slice(&2_u32.to_le_bytes());
    file.extend_from_slice(&body);
    std::fs::write(table_path, file).map_err(|source| Error::Write {
        path: table_path.into(),
        source,
    })?;

    let mut fixture = String::from(
        "# Generated by crates/flamsteed/examples/orb6_table.rs from orb6ephem.txt.\n\
         # hr theta_2025 rho_2025 theta_2026 rho_2026 theta_2027 rho_2027 theta_2028 rho_2028 theta_2029 rho_2029\n",
    );
    for (hr, (_, values)) in &records {
        use std::fmt::Write as _;
        let values: Vec<String> = values.iter().map(|value| format!("{value:?}")).collect();
        writeln!(fixture, "{hr} {}", values.join(" "))?;
    }
    std::fs::write(fixture_path, fixture).map_err(|source| Error::Write {
        path: fixture_path.into(),
        source,
    })
}

/// The catalogue stars of an orbit's system and their components.
fn members<'a>(entries: &'a [Entry], orbit: &Orbit) -> Vec<(&'a Entry, u8)> {
    let role = |entry: &Entry| match entry.component.as_str() {
        "A" => Some(1),
        "B" => Some(2),
        "" | "AB" => Some(3),
        _ => None,
    };
    let mut found: Vec<&Entry> = entries
        .iter()
        .filter(|entry| {
            orbit.ads.is_some_and(|ads| entry.ads == Some(ads))
                || orbit.hd.is_some_and(|hd| entry.hd == Some(hd))
        })
        .collect();
    // A companion without the ADS number lies within the orbit of its primary.
    let reach = (orbit.semi_major_arcsec * (1.0 + orbit.eccentricity) + 5.0) / 3600.0 * PI / 180.0;
    if let Some(primary) = found.iter().find(|entry| entry.component == "A").copied() {
        for entry in entries {
            let near = separation(primary, entry) < reach;
            if near && entry.component == "B" && !found.iter().any(|kept| kept.hr == entry.hr) {
                found.push(entry);
            }
        }
    }
    found
        .into_iter()
        .filter_map(|entry| Some((entry, role(entry)?)))
        .collect()
}

fn separation(a: &Entry, b: &Entry) -> f64 {
    let cos = a.dec.sin() * b.dec.sin() + a.dec.cos() * b.dec.cos() * (a.ra - b.ra).cos();
    cos.clamp(-1.0, 1.0).acos()
}

fn entry(line: &str) -> Option<Entry> {
    let field = |first: usize, last: usize| {
        line.get(first - 1..last.min(line.len()))
            .unwrap_or("")
            .trim()
    };
    if field(76, 77).is_empty() {
        return None;
    }
    let number = |first, last| field(first, last).parse::<f64>().ok();
    let ra = (number(76, 77)? + number(78, 79)? / 60.0 + number(80, 83)? / 3600.0) * PI / 12.0;
    let sign = if field(84, 84) == "-" { -1.0 } else { 1.0 };
    let dec =
        sign * (number(85, 86)? + number(87, 88)? / 60.0 + number(89, 90)? / 3600.0).to_radians();
    Some(Entry {
        hr: field(1, 4).parse().ok()?,
        hd: field(26, 31).parse().ok(),
        ads: field(45, 49).parse().ok(),
        component: field(50, 51).to_owned(),
        v_magnitude: number(103, 107)?,
        dmag: number(181, 184),
        ra,
        dec,
    })
    .filter(|entry| entry.v_magnitude.is_finite())
}

fn orbit(line: &str) -> Option<Orbit> {
    if !line.as_bytes().first()?.is_ascii_digit() || line.len() < 235 {
        return None;
    }
    let column =
        |first: usize, length: usize| line.get(first - 1..first - 1 + length).unwrap_or("");
    let number = |first, length| column(first, length).trim().parse::<f64>().ok();
    let grade: u8 = column(234, 1).parse().ok()?;
    let designation = column(31, 14).trim_end();
    // Only the AB pair moves a catalogue star; sub-pairs such as Aa,Ab do not.
    let components: String = designation
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_alphabetic() || *c == ',')
        .collect();
    let pair_is_ab = components.is_empty() || components == "BA";
    if !(1..=3).contains(&grade) || !pair_is_ab {
        return None;
    }
    let period_scale = match column(93, 1) {
        "y" => 1.0,
        "d" => 1.0 / 365.25,
        "c" => 100.0,
        _ => return None,
    };
    let axis_scale = match column(115, 1) {
        "a" => 1.0,
        "m" => 1e-3,
        "M" => 60.0,
        _ => return None,
    };
    let periastron = number(163, 12)?;
    let periastron_years = match column(175, 1) {
        // Besselian year to Julian years from J2000.
        "y" => (2_415_020.313_52 + (periastron - 1900.0) * 365.242_198_781 - 2_451_545.0) / 365.25,
        "c" => {
            (2_415_020.313_52 + (periastron * 100.0 - 1900.0) * 365.242_198_781 - 2_451_545.0)
                / 365.25
        }
        "d" => (periastron + 2_400_000.0 - 2_451_545.0) / 365.25,
        "m" => (periastron + 2_400_000.5 - 2_451_545.0) / 365.25,
        _ => return None,
    };
    let magnitude = |first| {
        let flag = column(first + 5, 1);
        if flag == "k" { None } else { number(first, 5) }
    };
    Some(Orbit {
        key: line.get(19..45)?.trim_end().to_owned(),
        reference: column(238, 8).trim().to_owned(),
        grade,
        hd: column(52, 6).trim().parse().ok(),
        ads: column(46, 5).trim().parse().ok(),
        magnitudes: magnitude(67).zip(magnitude(74)),
        period_years: number(81, 12)? * period_scale,
        periastron_years,
        eccentricity: number(188, 8)?,
        semi_major_arcsec: number(106, 9)? * axis_scale,
        inclination: number(126, 8)?.to_radians(),
        node: number(144, 8)?.to_radians(),
        periastron_argument: number(206, 8)?.to_radians(),
    })
}

/// CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320).
fn crc32(bytes: &[u8]) -> u32 {
    !bytes.iter().fold(u32::MAX, |crc, &byte| {
        (0..8).fold(crc ^ u32::from(byte), |crc, _| {
            (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg())
        })
    })
}
