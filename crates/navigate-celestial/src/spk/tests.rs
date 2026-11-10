use super::*;
use crate::{Correction, moon_ecef_with, observe_body, sun_ecef_with};

type TestSegmentSpec<'a> = (&'a str, i32, i32, i32, f64, f64, [[f64; 2]; 3]);

/// Build a valid little-endian `DAF/SPK` Type 2 byte buffer with up to 5 segments.
fn build_test_spk(segments_spec: &[TestSegmentSpec<'_>]) -> Vec<u8> {
    let mut bytes = vec![0_u8; 3 * 1024];
    bytes[0..8].copy_from_slice(b"DAF/SPK ");
    bytes[8..12].copy_from_slice(&2_i32.to_le_bytes());
    bytes[12..16].copy_from_slice(&6_i32.to_le_bytes());
    bytes[76..80].copy_from_slice(&2_i32.to_le_bytes());
    bytes[80..84].copy_from_slice(&2_i32.to_le_bytes());
    bytes[88..96].copy_from_slice(b"LTL-IEEE");

    let sum_off = 1024_usize;
    let name_off = 2048_usize;
    bytes[sum_off..sum_off + 8].copy_from_slice(&0.0_f64.to_le_bytes());
    bytes[sum_off + 8..sum_off + 16].copy_from_slice(&0.0_f64.to_le_bytes());
    bytes[sum_off + 16..sum_off + 24].copy_from_slice(&(segments_spec.len() as f64).to_le_bytes());

    let mut next_word = 385_i32;
    let mut payload_words: Vec<f64> = Vec::new();
    for (idx, &(name, target, center, frame, start_s, end_s, coeffs_km)) in
        segments_spec.iter().enumerate()
    {
        let mid_s = 0.5 * (start_s + end_s);
        let radius_s = 0.5 * (end_s - start_s);
        let rsize = 8_usize;
        let seg_words = [
            mid_s,
            radius_s,
            coeffs_km[0][0],
            coeffs_km[0][1],
            coeffs_km[1][0],
            coeffs_km[1][1],
            coeffs_km[2][0],
            coeffs_km[2][1],
            start_s,
            end_s - start_s,
            rsize as f64,
            1.0,
        ];
        let start_word = next_word;
        let end_word = start_word + (seg_words.len() as i32) - 1;
        next_word = end_word + 1;
        payload_words.extend_from_slice(&seg_words);

        let e_off = sum_off + 24 + idx * 40;
        bytes[e_off..e_off + 8].copy_from_slice(&start_s.to_le_bytes());
        bytes[e_off + 8..e_off + 16].copy_from_slice(&end_s.to_le_bytes());
        let ic = [target, center, frame, 2_i32, start_word, end_word];
        for (k, &val) in ic.iter().enumerate() {
            let off = e_off + 16 + k * 4;
            bytes[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
        let n_off = name_off + idx * 40;
        let mut padded = [b' '; 40];
        let b = name.as_bytes();
        let copy_len = b.len().min(40);
        padded[..copy_len].copy_from_slice(&b[..copy_len]);
        bytes[n_off..n_off + 40].copy_from_slice(&padded);
    }
    for w in payload_words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    let rem = bytes.len() % 1024;
    if rem != 0 {
        bytes.resize(bytes.len() + (1024 - rem), 0);
    }
    bytes
}

#[test]
fn spk_ephemeris_evaluates_chained_segments_and_mars_barycenter_fallback() {
    let spk_bytes = build_test_spk(&[
        ("SUN", 10, 0, 1, -10_000.0, 10_000.0, [[0.0, 0.0]; 3]),
        (
            "EMB",
            3,
            0,
            1,
            -10_000.0,
            10_000.0,
            [[149_597_870.7, 0.0], [0.0, 297_800.0], [0.0, 0.0]],
        ),
        (
            "EARTH",
            399,
            3,
            1,
            -10_000.0,
            10_000.0,
            [[-4_670.0, 0.0], [0.0, -120.0], [0.0, 0.0]],
        ),
        (
            "MOON",
            301,
            3,
            1,
            -10_000.0,
            10_000.0,
            [[379_700.0, 0.0], [0.0, 10_000.0], [0.0, 0.0]],
        ),
        (
            "MARS_BC",
            4,
            0,
            1,
            -10_000.0,
            10_000.0,
            [[-227_900_000.0, 0.0], [0.0, -240_000.0], [0.0, 0.0]],
        ),
    ]);
    let spk = SpkEphemeris::from_bytes(&spk_bytes).unwrap();
    let epoch = Epoch::from_tdb_seconds(0.0);

    let earth = spk.body_state(399, epoch).unwrap();
    assert!((earth.position_m[0] - (149_597_870.7 - 4_670.0) * 1_000.0).abs() < 1e-3);
    assert!((earth.velocity_m_s[1] - (297_800.0 - 120.0) * 1_000.0 / 10_000.0).abs() < 1e-6);

    let mars = spk.body_state(499, epoch).unwrap();
    assert!((mars.position_m[0] - (-227_900_000.0 * 1_000.0)).abs() < 1e-3);

    let observer = State {
        position_m: [6_378_137.0, 0.0, 0.0],
        velocity_m_s: [0.0; 3],
        epoch,
        frame: FrameId::EARTH_ITRF93,
    };
    let sight_mars = observe_body(&spk, observer, 499, Correction::VacuumReception).unwrap();
    assert!(sight_mars.coordinate_range_m > 3.0e11);

    let sun_fixed = sun_ecef_with(&spk, epoch, Some([6_378_137.0, 0.0, 0.0])).unwrap();
    let moon_fixed = moon_ecef_with(&spk, epoch, Some([6_378_137.0, 0.0, 0.0])).unwrap();
    assert!((Vector3::from(sun_fixed.direction).norm() - 1.0).abs() < 1e-12);
    assert!((Vector3::from(moon_fixed.direction).norm() - 1.0).abs() < 1e-12);
}
