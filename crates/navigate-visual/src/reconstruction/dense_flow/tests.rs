//! Dense pair association controls.

use super::*;
#[test]
fn interpolates_affine_flow_and_excludes_occlusion_edges() {
    let mut pairs: Vec<_> = (0..9)
        .flat_map(|y| {
            (0..9).map(move |x| {
                let x = x as f64 * 10.0;
                let y = y as f64 * 10.0;
                PixelMatch {
                    reference: [x, y].into(),
                    query: [x * 1.1 + y * 0.02 + 3.0, y * 0.9 - x * 0.04 - 2.0].into(),
                }
            })
        })
        .collect();
    let at = Vector2::new(42.0, 39.0);
    let found = Flow::new(&pairs).at(at).unwrap_or_default();
    assert!(
        (found
            - Vector2::new(
                at.x * 1.1 + at.y * 0.02 + 3.0,
                at.y * 0.9 - at.x * 0.04 - 2.0
            ))
        .norm()
            < 1e-9
    );
    for p in &mut pairs {
        if p.reference[0] > 40.0 {
            p.query[0] += 15.0;
        }
    }
    assert!(Flow::new(&pairs).at(at).is_none());
}
