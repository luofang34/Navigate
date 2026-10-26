//! Order along a Hilbert curve, so that near records sit near each other in a shard.

/// Position of `(x, y)` along the Hilbert curve of a `side` × `side` grid, where
/// `side` is a power of two.
pub(crate) fn index(side: u32, mut x: u32, mut y: u32) -> u64 {
    let mut d = 0_u64;
    let mut s = side / 2;
    while s > 0 {
        let rx = u32::from(x & s > 0);
        let ry = u32::from(y & s > 0);
        d += u64::from(s) * u64::from(s) * u64::from((3 * rx) ^ ry);
        if ry == 0 {
            if rx == 1 {
                x = side - 1 - x;
                y = side - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::index;

    #[test]
    fn the_curve_visits_each_cell_once_and_steps_to_neighbours() {
        let side = 8;
        let mut order = vec![(0, 0); 64];
        for x in 0..side {
            for y in 0..side {
                order[index(side, x, y) as usize] = (x, y);
            }
        }
        for pair in order.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            assert_eq!(a.0.abs_diff(b.0) + a.1.abs_diff(b.1), 1, "{a:?} {b:?}");
        }
    }
}
