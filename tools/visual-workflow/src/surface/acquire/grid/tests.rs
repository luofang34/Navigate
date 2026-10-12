use super::*;

#[test]
fn grid_covers_every_pixel_including_non_aligned_edges() {
    for size in [(4000, 4000), (511, 513), (32, 32)] {
        let side = if size.0 == 4000 { 512 } else { 32 };
        let grid = windows(size, side, 4096).unwrap();
        let mut coverage = vec![false; size.0 as usize * size.1 as usize];
        let mut unique = std::collections::BTreeSet::new();
        for &[x, y, w, h] in &grid {
            assert!(unique.insert([x, y, w, h]));
            assert!(x + w <= size.0 && y + h <= size.1);
            for py in y..y + h {
                for px in x..x + w {
                    coverage[(py * size.0 + px) as usize] = true;
                }
            }
        }
        assert!(coverage.into_iter().all(|p| p));
        assert!(
            grid.windows(2)
                .all(|pair| (pair[0][1], pair[0][0]) < (pair[1][1], pair[1][0]))
        );
        assert!(grid.contains(&[size.0 - side, size.1 - side, side, side]));
    }
}

#[test]
fn exact_candidate_bound_and_invalid_sizes_are_enforced() {
    assert_eq!(windows((4000, 4000), 512, 900).unwrap().len(), 225);
    assert!(windows((4000, 4000), 512, 899).is_err());
    assert_eq!(windows((4000, 4000), 1024, 196).unwrap().len(), 49);
    for (size, side) in [
        ((4000, 4000), 0),
        ((4000, 4000), 31),
        ((4000, 4000), 513),
        ((4000, 4000), 4096),
        ((8193, 4000), 512),
        ((8192, 8192), 32),
    ] {
        assert!(windows(size, side, usize::MAX).is_err());
    }
}
