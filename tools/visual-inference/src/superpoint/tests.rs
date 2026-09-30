use super::*;
#[test]
fn normalization_is_independent_of_export_aspect_ratio() {
    let point = [271.0, 179.0];
    let source = [640, 480];
    let target = [640, 360];
    let mapped = normalized_pixel(point, source, target);
    for i in 0..2 {
        let original = (point[i] - source[i] as f32 / 2.0) / (640.0 * 0.7);
        let exported = (mapped[i] - target[i] as f32 / 2.0) / (640.0 * 0.7);
        assert!((original - exported).abs() < 1e-6);
    }
}
#[test]
fn suppression_preserves_separated_peaks() {
    let mut heat = vec![0.0; 32 * 32];
    heat[10 * 32 + 10] = 1.0;
    heat[11 * 32 + 11] = 0.5;
    heat[20 * 32 + 20] = 0.75;
    let kept = suppress(&heat, 32, 32);
    assert_eq!(kept[10 * 32 + 10], 1.0);
    assert_eq!(kept[11 * 32 + 11], 0.0);
    assert_eq!(kept[20 * 32 + 20], 0.75);
}

#[test]
fn separable_maximum_matches_square_support_at_borders_and_ties() {
    for (width, height) in [(1, 1), (3, 2), (8, 11), (19, 13), (80, 45)] {
        let input: Vec<f32> = (0..width * height)
            .map(|i| ((i * 73 + i / width * 11) % 97) as f32 / 97.0)
            .collect();
        let actual = maximum(&input, width, height);
        for y in 0..height {
            for x in 0..width {
                let mut expected = 0.0_f32;
                for yy in y.saturating_sub(4)..=(y + 4).min(height - 1) {
                    for xx in x.saturating_sub(4)..=(x + 4).min(width - 1) {
                        expected = expected.max(input[yy * width + xx]);
                    }
                }
                assert_eq!(actual[y * width + x], expected);
            }
        }
    }
}
