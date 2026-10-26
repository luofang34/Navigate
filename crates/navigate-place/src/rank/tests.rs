use super::*;

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

#[test]
fn the_best_record_ranks_first_and_near_duplicates_are_dropped() {
    let records = vec![
        Record {
            centre: [40.0, -75.0],
            descriptor: unit(&[1.0, 0.0, 0.0]),
        },
        Record {
            centre: [40.0001, -75.0],
            descriptor: unit(&[0.99, 0.1, 0.0]),
        },
        Record {
            centre: [40.1, -75.0],
            descriptor: unit(&[0.0, 1.0, 0.0]),
        },
        Record {
            centre: [40.2, -75.0],
            descriptor: unit(&[0.5, 0.5, 0.0]),
        },
    ];
    let queries = vec![
        Descriptor {
            values: unit(&[0.0, 0.0, 1.0]),
        },
        Descriptor {
            values: unit(&[1.0, 0.05, 0.0]),
        },
    ];
    let ranked = rank(&queries, &records, |_| true, 3, 100.0);
    assert_eq!(ranked.len(), 3);
    assert_eq!(ranked[0].centre, [40.0, -75.0]);
    assert_eq!(ranked[0].query, 1);
    assert!(
        ranked.iter().all(|r| r.centre != [40.0001, -75.0]),
        "the near duplicate is dropped"
    );
    let only_north = rank(&queries, &records, |c| c[0] > 40.05, 5, 100.0);
    assert_eq!(only_north.len(), 2);
}
