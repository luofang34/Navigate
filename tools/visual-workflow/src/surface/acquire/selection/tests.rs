use super::*;
use nalgebra::Vector2;
fn pairs(xs: &[f64], y: f64) -> Vec<PixelMatch> {
    xs.iter()
        .map(|x| PixelMatch {
            reference: Vector2::new(*x, y),
            query: Vector2::new(0.0, 0.0),
        })
        .collect()
}
#[test]
fn median_seed_and_clamped_neighbors_match_the_fixed_search_order() {
    let seed = seed(&pairs(&[20.0, 30.0, 40.0], 4096.0), (6275, 6384))
        .unwrap()
        .unwrap();
    assert_eq!(seed, [0, 3072, 2048, 2048]);
    assert_eq!(
        neighbors(seed, (6275, 6384)),
        vec![
            [0, 3072, 2048, 2048],
            [0, 2048, 2048, 2048],
            [1024, 3072, 2048, 2048],
            [0, 4096, 2048, 2048],
            [1024, 2048, 2048, 2048],
            [1024, 4096, 2048, 2048]
        ]
    );
}
#[test]
fn lattice_ties_use_even_rounding_and_empty_evidence_has_no_proposal() {
    assert_eq!(
        seed(&pairs(&[2560.0], 1536.0), (6000, 6000)).unwrap(),
        Some([2048, 0, 2048, 2048])
    );
    assert_eq!(seed(&[], (6000, 6000)).unwrap(), None);
    assert!(seed(&[], (2000, 6000)).is_err());
    assert!(seed(&pairs(&[f64::NAN], 0.0), (6000, 6000)).is_err());
    assert_eq!(neighbors([0, 0, 2048, 2048], (2048, 2048)).len(), 1);
}

#[test]
fn finer_search_stays_inside_parent_in_fixed_order() {
    let areas = children([2048, 2048, 2048, 2048]);
    assert_eq!(areas.len(), 9);
    assert_eq!(areas[0], [2560, 2560, 1024, 1024]);
    assert_eq!(areas[2], [2048, 2560, 1024, 1024]);
    for [x, y, w, h] in areas {
        assert!(x >= 2048 && y >= 2048 && x + w <= 4096 && y + h <= 4096);
    }
}
