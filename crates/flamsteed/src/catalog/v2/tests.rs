use std::vec::Vec;

use super::super::{Catalog, crc32, v1};
use super::*;
use crate::{Axes, FormatError};

/// A format version 2 record of `star` with the given provenance.
fn record_of(star: &Star, provenance: u16, source_id: u64) -> Record {
    let mut record = [0; RECORD_BYTES];
    let StarId::Hr(hr) = star.id;
    record[0..2].copy_from_slice(&hr.to_le_bytes());
    record[2..4].copy_from_slice(&provenance.to_le_bytes());
    record[4..12].copy_from_slice(&source_id.to_le_bytes());
    for (i, c) in star.position.iter().chain(&star.velocity).enumerate() {
        record[12 + 8 * i..20 + 8 * i].copy_from_slice(&c.to_le_bytes());
    }
    let single = |value: Option<f64>| value.map_or(f32::NAN, |v| v as f32);
    record[60..64].copy_from_slice(&single(star.parallax_mas).to_le_bytes());
    record[64..68].copy_from_slice(&single(star.radial_velocity_km_s).to_le_bytes());
    let hundredths = |value: f64| (value * 100.0).round() as i16;
    record[68..70].copy_from_slice(&hundredths(star.v_magnitude).to_le_bytes());
    let colour = star.b_minus_v.map_or(i16::MIN, hundredths);
    record[70..72].copy_from_slice(&colour.to_le_bytes());
    record
}

/// A format version 2 file of `records` with a correct header.
fn file_of(records: &[Record], epoch: f64) -> Vec<u8> {
    let body: Vec<u8> = records.concat();
    let mut file = Vec::new();
    file.extend_from_slice(b"FLAMSTED");
    file.extend_from_slice(&2_u16.to_le_bytes());
    file.extend_from_slice(&(RECORD_BYTES as u16).to_le_bytes());
    file.extend_from_slice(&u32::try_from(records.len()).unwrap().to_le_bytes());
    file.extend_from_slice(&epoch.to_le_bytes());
    file.extend_from_slice(&crc32(&body).to_le_bytes());
    file.extend_from_slice(&3_u32.to_le_bytes());
    file.extend_from_slice(&body);
    file
}

fn embedded() -> Vec<Star> {
    Catalog::bright_star_catalogue().stars().collect()
}

#[test]
fn version_2_round_trips_the_embedded_stars() {
    let stars = embedded();
    let records: Vec<Record> = stars.iter().map(|star| record_of(star, 1, 0)).collect();
    let file = file_of(&records, 0.0);
    let catalog = Catalog::from_bytes(&file).unwrap();
    assert_eq!(catalog.axes(), Axes::Icrs);
    assert_eq!(catalog.len(), stars.len());
    assert!(catalog.stars().eq(stars.iter().copied()));
    let limit = stars[10].v_magnitude;
    assert_eq!(
        catalog.brighter_than(limit).len(),
        Catalog::bright_star_catalogue().brighter_than(limit).len()
    );
    // f32 motions agree with the f64 directions to the f32 rounding.
    for (motion, star) in catalog.motions::<f32>().zip(&stars).take(100) {
        let [x, y, z] = motion.direction_at(-2000.0);
        let reference = star.direction_at(-2000.0);
        let error = (f64::from(x) - reference[0]).abs()
            + (f64::from(y) - reference[1]).abs()
            + (f64::from(z) - reference[2]).abs();
        assert!(error < 1e-6, "{error}");
    }
}

#[test]
fn version_2_keeps_the_provenance_and_the_absent_values() {
    let stars = embedded();
    let alpha_centauri = stars.iter().find(|star| star.hr() == Some(5459)).unwrap();
    let mut without_parallax = *alpha_centauri;
    without_parallax.parallax_mas = None;
    without_parallax.b_minus_v = None;
    let records = [
        record_of(alpha_centauri, 2, 5_853_498_713_190_525_696),
        record_of(&without_parallax, 1, 0),
    ];
    let file = file_of(&records, 16.0);
    let imported: Vec<Star> = Catalog::from_bytes(&file).unwrap().stars().collect();
    assert_eq!(
        imported[0].astrometry,
        Astrometry::Other {
            provenance: 2,
            source_id: 5_853_498_713_190_525_696
        }
    );
    assert_eq!(imported[0].reference_epoch_years, 16.0);
    // Another catalogue's position already includes the orbit.
    assert_eq!(imported[0].orbital_offset_rad_at(-2000.0), None);
    assert_eq!(imported[1].astrometry, Astrometry::BrightStarCatalogue);
    assert!(imported[1].orbital_offset_rad_at(-2000.0).is_some());
    assert_eq!(imported[1].parallax_mas, None);
    assert_eq!(imported[1].distance_pc(), None);
    assert_eq!(imported[1].b_minus_v, None);
}

#[test]
fn version_2_rejects_records_that_are_not_directions() {
    let stars = embedded();
    let mut records: Vec<Record> = stars[..3].iter().map(|s| record_of(s, 1, 0)).collect();
    records[1][12..20].copy_from_slice(&2.0_f64.to_le_bytes());
    assert_eq!(
        Catalog::from_bytes(&file_of(&records, 0.0)).err(),
        Some(FormatError::Record { index: 1 })
    );
    records[1] = record_of(&stars[1], 1, 0);
    records[2][36..44].copy_from_slice(&f64::NAN.to_le_bytes());
    assert_eq!(
        Catalog::from_bytes(&file_of(&records, 0.0)).err(),
        Some(FormatError::Record { index: 2 })
    );
}

#[test]
fn version_2_needs_its_record_size_and_source() {
    let stars = embedded();
    let records: Vec<Record> = stars[..2].iter().map(|s| record_of(s, 1, 0)).collect();
    let mut file = file_of(&records, 0.0);
    file[10..12].copy_from_slice(&(v1::RECORD_BYTES as u16).to_le_bytes());
    assert_eq!(
        Catalog::from_bytes(&file).err(),
        Some(FormatError::Version {
            version: 2,
            record_bytes: 34
        })
    );
    let mut file = file_of(&records, 0.0);
    file[28..32].copy_from_slice(&1_u32.to_le_bytes());
    assert_eq!(
        Catalog::from_bytes(&file).err(),
        Some(FormatError::Source { source: 1 })
    );
}
