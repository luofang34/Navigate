use super::*;

fn record(lat: f64, lon: f64, seed: f32) -> Record {
    let mut descriptor: Vec<f32> = (0..8).map(|i| (seed + i as f32).sin()).collect();
    let norm = descriptor.iter().map(|v| v * v).sum::<f32>().sqrt();
    descriptor.iter_mut().for_each(|v| *v /= norm);
    Record {
        centre: [lat, lon],
        descriptor,
    }
}

#[test]
fn a_shard_round_trips_its_records_in_curve_order() {
    let cell = Cell { x: 297, y: 384 };
    let records: Vec<Record> = (0..20)
        .map(|i| {
            record(
                40.90 + f64::from(i % 5) * 0.01,
                -75.40 + f64::from(i / 5) * 0.01,
                i as f32,
            )
        })
        .collect();
    let bytes = encode_shard(cell, 4000.0, 8, &records).unwrap();
    assert_eq!(bytes.len(), 32 + 20 * (8 + 16));
    let (read_cell, side, read) = decode_shard("test", &bytes).unwrap();
    assert_eq!((read_cell, side), (cell, 4000.0));
    for original in &records {
        let copy = read
            .iter()
            .find(|r| {
                (r.centre[0] - original.centre[0]).abs() < 1e-6
                    && (r.centre[1] - original.centre[1]).abs() < 1e-6
            })
            .unwrap();
        for (a, b) in copy.descriptor.iter().zip(&original.descriptor) {
            assert!((a - b).abs() < 2e-3);
        }
    }
    let first = read[0].centre;
    let second = read[1].centre;
    assert!(
        crate::region::distance_m(first, second) < 1_600.0,
        "neighbours on the curve are near"
    );
}

#[test]
fn a_truncated_or_foreign_shard_is_refused() {
    let bytes = encode_shard(Cell { x: 1, y: 2 }, 160.0, 8, &[record(40.5, -74.5, 1.0)]).unwrap();
    assert!(decode_shard("t", &bytes[..bytes.len() - 1]).is_err());
    let mut foreign = bytes.clone();
    foreign[0] = b'X';
    assert!(decode_shard("t", &foreign).is_err());
    assert!(encode_shard(Cell { x: 1, y: 2 }, 160.0, 9, &[record(40.5, -74.5, 1.0)]).is_err());
}
