use super::*;

#[test]
fn short_buffers_and_strides_are_refused() {
    let data = [0_u8; 10];
    assert!(
        ImageView::new(
            4,
            3,
            Pixels::Gray8 {
                data: &data,
                stride: 4
            }
        )
        .is_err()
    );
    assert!(
        ImageView::new(
            4,
            2,
            Pixels::Gray8 {
                data: &data,
                stride: 3
            }
        )
        .is_err()
    );
    assert!(
        ImageView::new(
            4,
            2,
            Pixels::Gray8 {
                data: &data,
                stride: 5
            }
        )
        .is_ok()
    );
    let wide = [0_u16; 8];
    let bad_bits = Pixels::Gray16 {
        data: &wide,
        stride: 4,
        significant_bits: 17,
    };
    assert!(ImageView::new(4, 2, bad_bits).is_err());
}

#[test]
fn luma_reads_each_format() {
    let rgb = [255_u8, 255, 255, 0, 0, 0];
    let view = ImageView::new(
        2,
        1,
        Pixels::Rgb8 {
            data: &rgb,
            stride: 6,
        },
    )
    .unwrap();
    assert!((view.luma(0, 0).unwrap() - 1.0).abs() < 1e-6);
    assert_eq!(view.luma(1, 0), Some(0.0));
    assert_eq!(view.luma(2, 0), None);
    let counts = [16383_u16, 0];
    let lwir = ImageView::new(
        2,
        1,
        Pixels::Gray16 {
            data: &counts,
            stride: 2,
            significant_bits: 14,
        },
    )
    .unwrap();
    assert!((lwir.luma(0, 0).unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn percentile_stretch_spans_the_narrow_raw_range_of_a_thermal_scene() {
    // A 14-bit scene that uses only counts 8000 to 8200.
    let mut counts = [0_u16; 64 * 64];
    for (i, c) in counts.iter_mut().enumerate() {
        *c = 8000 + (i % 201) as u16;
    }
    let view = ImageView::new(
        64,
        64,
        Pixels::Gray16 {
            data: &counts,
            stride: 64,
            significant_bits: 14,
        },
    )
    .unwrap();
    let mut out = [0_u8; 64 * 64];
    view.to_gray8_percentile(0.01, 0.99, &mut out).unwrap();
    let (min, max) = out
        .iter()
        .fold((255, 0), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    assert_eq!((min, max), (0, 255));
    let mut again = [0_u8; 64 * 64];
    view.to_gray8_percentile(0.01, 0.99, &mut again).unwrap();
    assert_eq!(out, again, "the stretch is deterministic");
}

#[test]
fn percentile_stretch_checks_its_arguments() {
    let data = [0_u8; 16];
    let view = ImageView::new(
        4,
        4,
        Pixels::Gray8 {
            data: &data,
            stride: 4,
        },
    )
    .unwrap();
    let mut out = [0_u8; 15];
    assert!(view.to_gray8_percentile(0.0, 1.0, &mut out).is_err());
    let mut out = [0_u8; 16];
    assert!(view.to_gray8_percentile(0.5, 0.5, &mut out).is_err());
    assert!(view.to_gray8_percentile(0.0, 1.0, &mut out).is_ok());
}
