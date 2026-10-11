use super::*;

#[test]
fn bicubic_matches_grid_sample_at_interior_and_padded_edges() {
    let data: Vec<_> = (0..HEIGHT / 8)
        .flat_map(|y| (0..WIDTH / 8).map(move |x| (x + 3 * y) as f32))
        .collect();
    for (point, expected) in [
        ([0.23, -0.2], -0.172_809),
        ([10.4, 9.7], 39.397_97),
        ([99.2, 70.8], 281.498_7),
    ] {
        assert!((Bicubic::new(point).sample(&data) - expected).abs() < 0.02);
    }
}

#[test]
fn peak_selection_rejects_dustbin_and_suppresses_nearby_lower_peak() {
    let mut logits = vec![-20.0; 65 * CELLS];
    for cell in 0..CELLS {
        logits[64 * CELLS + cell] = 20.0;
    }
    assert!(peaks(&heatmap(&logits), &vec![1.0; CELLS]).is_empty());
    let mut heat = vec![0.0; WIDTH * HEIGHT];
    heat[100 * WIDTH + 100] = 0.9;
    heat[100 * WIDTH + 101] = 0.8;
    heat[100 * WIDTH + 110] = 0.7;
    let selected = peaks(&heat, &vec![1.0; CELLS]);
    assert_eq!(selected.len(), 2);
    assert_eq!((selected[0].0, selected[0].1), (100, 100));
    assert_eq!((selected[1].0, selected[1].1), (110, 100));
}

#[test]
fn uniform_input_normalizes_to_finite_zero() {
    assert!(
        normalize_resized(&vec![0.5; WIDTH * 600])
            .iter()
            .all(|v| v.abs() < 1e-6)
    );
}

#[test]
fn reused_weights_preserve_interpolation_bits_at_edges_and_interior() {
    let data: Vec<_> = (0..CELLS).map(|i| (i as f32 * 0.137).sin()).collect();
    for point in [[-0.5_f32, -0.5], [0.23, -0.2], [42.71, 39.28], [99.4, 71.4]] {
        let x = point[0].floor() as i32;
        let y = point[1].floor() as i32;
        let mut expected = 0.0;
        for dy in -1..=2 {
            for dx in -1..=2 {
                expected += sample(&data, x + dx, y + dy)
                    * cubic(point[0] - (x + dx) as f32)
                    * cubic(point[1] - (y + dy) as f32);
            }
        }
        assert_eq!(Bicubic::new(point).sample(&data), expected);
    }
}

#[test]
fn cached_softmax_preserves_heatmap_bits() {
    let logits: Vec<_> = (0..65 * CELLS)
        .map(|i| (i as f32 * 0.237).sin() * 20.0)
        .collect();
    let actual = heatmap(&logits);
    for cell in [0, 197, CELLS - 1] {
        let max = (0..65)
            .map(|c| logits[c * CELLS + cell])
            .fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = (0..65)
            .map(|c| (logits[c * CELLS + cell] - max).exp())
            .sum();
        for c in 0..64 {
            let x = (cell % (WIDTH / 8)) * 8 + c % 8;
            let y = (cell / (WIDTH / 8)) * 8 + c / 8;
            assert_eq!(
                actual[y * WIDTH + x],
                (logits[c * CELLS + cell] - max).exp() / sum
            );
        }
    }
}
