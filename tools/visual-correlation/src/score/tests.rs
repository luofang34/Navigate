//! Compare every lag with a double precision direct sum.
use super::*;
#[cfg(test)]
fn grid(w: usize, h: usize, c: usize, holes: bool) -> Grid {
    let mask: Vec<_> = (0..w * h).map(|i| !holes || i % 7 != 0).collect();
    let values = (0..w * h * c)
        .map(|i| {
            if mask[i / c] {
                ((i * 17 + 3) as f32).sin() * 0.2
            } else {
                f32::NAN
            }
        })
        .collect();
    Grid::new(w, h, c, values, mask).unwrap()
}
#[test]
fn linear_padding_masks_and_all_signed_lags_match_f64() {
    for (qw, qh, mw, mh) in [(5, 4, 8, 7), (4, 7, 9, 6), (3, 3, 2, 4), (6, 2, 3, 5)] {
        let q = grid(qw, qh, 3, true);
        let m = grid(mw, mh, 3, true);
        let lags: Vec<_> = (-(qh as i32) - 1..=mh as i32 + 1)
            .flat_map(|y| (-(qw as i32) - 1..=mw as i32 + 1).map(move |x| [x, y]))
            .collect();
        let oracle = direct(&q, &m, &lags, 2, true);
        let actual = Correlator::new(&q, &m)
            .unwrap()
            .score(&q, &lags, 2)
            .unwrap();
        for (a, b) in oracle.iter().zip(actual) {
            assert_eq!(a.count, b.count);
            assert_eq!(a.value.is_some(), b.value.is_some());
            if let (Some(x), Some(y)) = (a.value, b.value) {
                assert!((x - y).abs() < 2e-6, "{x} {y}");
            }
        }
    }
}
#[test]
fn zero_support_has_no_peak_even_with_nan_payloads() {
    let q = Grid::new(5, 3, 2, vec![f32::NAN; 30], vec![false; 15]).unwrap();
    let m = grid(7, 4, 2, false);
    let lags = vec![[0, 0], [-4, -2], [6, 3]];
    for v in Correlator::new(&q, &m)
        .unwrap()
        .score(&q, &lags, 1)
        .unwrap()
    {
        assert_eq!(v.count, 0);
        assert!(v.value.is_none());
    }
}
#[test]
fn edge_peak_uses_positive_map_lag_without_wraparound() {
    let q = grid(3, 4, 2, false);
    let mut m = Grid::new(8, 7, 2, vec![0.0; 112], vec![true; 56]).unwrap();
    for y in 0..4 {
        for x in 0..3 {
            for c in 0..2 {
                m.values[((y + 3) * 8 + x + 5) * 2 + c] = q.values[(y * 3 + x) * 2 + c];
            }
        }
    }
    let lags: Vec<_> = (0..7).flat_map(|y| (0..8).map(move |x| [x, y])).collect();
    let scores = Correlator::new(&q, &m)
        .unwrap()
        .score(&q, &lags, 12)
        .unwrap();
    let best = scores
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.value.map(|v| (i, v)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    assert_eq!(lags[best.0], [5, 3]);
}
#[test]
fn equal_peaks_remain_ties_within_numeric_error() {
    let q = Grid::new(4, 3, 2, vec![0.5; 24], vec![true; 12]).unwrap();
    let m = Grid::new(7, 6, 2, vec![0.5; 84], vec![true; 42]).unwrap();
    let scores = Correlator::new(&q, &m)
        .unwrap()
        .score(&q, &[[0, 0], [1, 1], [3, 3]], 12)
        .unwrap();
    for score in scores {
        assert!((score.value.unwrap() - 0.5).abs() < 2e-6);
    }
}
