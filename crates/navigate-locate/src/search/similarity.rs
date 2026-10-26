//! A 2-D similarity fit of matches between a photo projection and a map window.
use nalgebra::Vector2;
use navigate_visual::PixelMatch;

/// Largest distance of an inlier from the fitted similarity, window pixels.
const INLIER_PX: f64 = 4.0;

/// A 2-D similarity `q ↦ a·q + b` in complex form, from the projection to the window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Similarity {
    a: Vector2<f64>,
    b: Vector2<f64>,
}

impl Similarity {
    /// The window pixel of a projection pixel.
    pub fn apply(&self, p: Vector2<f64>) -> Vector2<f64> {
        Vector2::new(
            self.a.x * p.x - self.a.y * p.y + self.b.x,
            self.a.y * p.x + self.a.x * p.y + self.b.y,
        )
    }

    /// Rotation from the projection to the window, degrees clockwise on the map.
    pub fn angle_deg(&self) -> f64 {
        self.a.y.atan2(self.a.x).to_degrees()
    }

    fn through(p: [Vector2<f64>; 2], q: [Vector2<f64>; 2]) -> Option<Self> {
        let (dp, dq) = (p[1] - p[0], q[1] - q[0]);
        let norm = dp.norm_squared();
        if norm < 100.0 {
            return None;
        }
        let a = Vector2::new(dq.x * dp.x + dq.y * dp.y, dq.y * dp.x - dq.x * dp.y) / norm;
        // The projection uses a guessed height: allow a scale error, but not a mirror
        // or a large turn, since the heading hypothesis is near.
        if !(0.6..=1.7).contains(&a.norm()) || a.y.atan2(a.x).abs() > 0.6 {
            return None;
        }
        let b = q[0]
            - Self {
                a,
                b: Vector2::zeros(),
            }
            .apply(p[0]);
        Some(Self { a, b })
    }
}

/// The similarity with the most inliers, by a fixed-seed sample of match pairs.
pub fn fit(matches: &[PixelMatch]) -> Option<(usize, Similarity)> {
    if matches.len() < 6 {
        return None;
    }
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = |n: usize| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((state >> 33) as usize) % n
    };
    let mut best: Option<(usize, Similarity)> = None;
    for _ in 0..400 {
        let (m, n) = (&matches[next(matches.len())], &matches[next(matches.len())]);
        let Some(model) = Similarity::through([m.query, n.query], [m.reference, n.reference])
        else {
            continue;
        };
        let inliers = matches
            .iter()
            .filter(|c| (model.apply(c.query) - c.reference).norm() < INLIER_PX)
            .count();
        if best.is_none_or(|(b, _)| inliers > b) {
            best = Some((inliers, model));
        }
    }
    best
}
