use super::mask_blocks;
use std::cell::Cell;

fn per_pixel(depth: &mut [f32], width: usize, valid: impl Fn(usize, usize, f32) -> bool) {
    for (i, d) in depth.iter_mut().enumerate() {
        if *d > 0.0 && !valid(i % width, i / width, *d) {
            *d = 0.0;
        }
    }
}

#[test]
fn block_mask_never_keeps_rejected_depth_and_samples_few_points() {
    let (width, height) = (961, 541);
    let mut depth: Vec<f32> = (0..width * height)
        .map(|i| {
            if (i % width) < 7 {
                0.0
            } else {
                50.0 + (i % 13) as f32
            }
        })
        .collect();
    // Coverage edges at arbitrary pixel positions, as tile boundaries project into the image.
    let valid = |x: usize, y: usize, _: f32| x * 3 + y * 2 < 1900 || (x > 700 && y < 123);
    let mut expected = depth.clone();
    per_pixel(&mut expected, width, valid);
    let samples = Cell::new(0_usize);
    mask_blocks(&mut depth, width, height, |x, y, d| {
        samples.set(samples.get() + 1);
        valid(x, y, d)
    });
    assert_eq!(depth, expected);
    assert!(
        samples.get() * 8 < width * height,
        "{} samples",
        samples.get()
    );
}

#[test]
fn empty_depth_stays_empty() {
    let mut depth = vec![0.0_f32; 64 * 48];
    mask_blocks(&mut depth, 64, 48, |_, _, _| true);
    assert!(depth.iter().all(|d| *d == 0.0));
}
