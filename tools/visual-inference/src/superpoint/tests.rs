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
