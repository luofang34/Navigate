//! Position embeddings resized from the training grid to the input grid.
//!
//! The pinned ONNX export resizes the patch embeddings to the input grid with
//! bicubic interpolation (coefficient −0.75, `align_corners = false`, scale
//! `side / grid`) and keeps the class embedding as it is. The parity test checks it.

const A: f64 = -0.75;

fn cubic(t: f64) -> [f64; 4] {
    let w = |x: f64| {
        let x = x.abs();
        if x <= 1.0 {
            ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
        } else {
            0.0
        }
    };
    [w(t + 1.0), w(t), w(1.0 - t), w(2.0 - t)]
}

/// The class embedding, then the `grid` × `grid` patch embeddings of `table` resized to
/// `side` × `side`, each `width` values.
pub(super) fn resize(table: &[f32], grid: usize, side: usize, width: usize) -> Vec<f32> {
    let mut out = table[..width].to_vec();
    if side == grid {
        out.extend_from_slice(&table[width..]);
        return out;
    }
    let scale = side as f64 / grid as f64;
    let last = grid as i64 - 1;
    let taps = |dst: usize| {
        let src = (dst as f64 + 0.5) / scale - 0.5;
        let base = src.floor();
        let weights = cubic(src - base);
        let index = |k: i64| (base as i64 - 1 + k).clamp(0, last) as usize;
        [0, 1, 2, 3].map(|k| (index(k), weights[k as usize]))
    };
    let patch = |y: usize, x: usize| &table[(1 + y * grid + x) * width..(2 + y * grid + x) * width];
    for y in 0..side {
        let rows = taps(y);
        for x in 0..side {
            let columns = taps(x);
            let mut value = vec![0.0_f64; width];
            for (ry, wy) in rows {
                for (cx, wx) in columns {
                    for (v, p) in value.iter_mut().zip(patch(ry, cx)) {
                        *v += wy * wx * f64::from(*p);
                    }
                }
            }
            out.extend(value.into_iter().map(|v| v as f32));
        }
    }
    out
}
