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
        assert!((bicubic(&data, point) - expected).abs() < 0.02);
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
