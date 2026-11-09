//! A table of catalogue star pairs sorted by their separation angle.
//!
//! The angle between two stars does not change with the attitude or with
//! the Earth rotation. Thus a table that [`PairCatalogue::new`] makes from
//! the catalogue of one epoch serves all frames near that epoch. Proper
//! motion and differential aberration change an angle by less than about
//! 10 arcseconds for stars in one field of view over some years. The
//! identification tolerance must include this change.

use nalgebra::Vector3;
use navigate_celestial::StarSight;

use crate::StarTrackError;

/// Most pairs of one table. The limit keeps the memory below about 64 MB.
pub const MAX_PAIRS: usize = 8_000_000;

/// One star of a [`PairCatalogue`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TableStar {
    pub(crate) hr: u16,
    pub(crate) unit: Vector3<f64>,
}

/// Two stars and the angle between them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Pair {
    pub(crate) angle_rad: f32,
    pub(crate) first: u16,
    pub(crate) second: u16,
}

/// The catalogue star pairs closer than a maximum angle, sorted by angle,
/// for identification without an attitude prior.
///
/// Make the table once. Each frame then needs only the search.
#[derive(Debug, Clone, PartialEq)]
pub struct PairCatalogue {
    stars: Vec<TableStar>,
    pairs: Vec<Pair>,
    max_angle_rad: f64,
}

impl PairCatalogue {
    /// The pairs of the stars of `stars` at least as bright as
    /// `faintest_v_magnitude` that are closer than `max_angle_rad`.
    ///
    /// Use the diagonal field of view of the camera for `max_angle_rad`,
    /// and the magnitude limit that the camera detects reliably. The cost
    /// is proportional to the square of the star count.
    ///
    /// # Errors
    /// Refuses an angle outside `(0, π]`, a magnitude that is not finite,
    /// and more than 65 535 stars. Returns [`StarTrackError::TooManyPairs`]
    /// when the table has more than [`MAX_PAIRS`] pairs.
    pub fn new(
        stars: &[StarSight],
        max_angle_rad: f64,
        faintest_v_magnitude: f64,
    ) -> Result<Self, StarTrackError> {
        let angle_ok = max_angle_rad > 0.0 && max_angle_rad <= std::f64::consts::PI;
        if !angle_ok || !faintest_v_magnitude.is_finite() {
            return Err(StarTrackError::Invalid {
                field: "pair table angle or magnitude",
            });
        }
        let stars: Vec<TableStar> = stars
            .iter()
            .filter(|s| s.v_magnitude <= faintest_v_magnitude)
            .map(|s| TableStar {
                hr: s.hr,
                unit: Vector3::from(s.direction).normalize(),
            })
            .collect();
        if stars.len() > usize::from(u16::MAX) {
            return Err(StarTrackError::Invalid {
                field: "pair table star count",
            });
        }
        let min_cos = max_angle_rad.cos();
        let mut pairs = Vec::new();
        for (i, a) in stars.iter().enumerate() {
            for (j, b) in stars.iter().enumerate().skip(i + 1) {
                let cos = a.unit.dot(&b.unit);
                if cos < min_cos {
                    continue;
                }
                if pairs.len() == MAX_PAIRS {
                    return Err(StarTrackError::TooManyPairs { limit: MAX_PAIRS });
                }
                pairs.push(Pair {
                    angle_rad: a.unit.angle(&b.unit) as f32,
                    first: i as u16,
                    second: j as u16,
                });
            }
        }
        pairs.sort_by(|p, q| {
            p.angle_rad
                .total_cmp(&q.angle_rad)
                .then(p.first.cmp(&q.first))
                .then(p.second.cmp(&q.second))
        });
        Ok(Self {
            stars,
            pairs,
            max_angle_rad,
        })
    }

    /// Stars in the table.
    #[must_use]
    pub fn star_count(&self) -> usize {
        self.stars.len()
    }

    /// Pairs in the table.
    #[must_use]
    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }

    /// The largest angle of a pair in radians.
    #[must_use]
    pub fn max_angle_rad(&self) -> f64 {
        self.max_angle_rad
    }

    /// The star at `index`.
    pub(crate) fn star(&self, index: u16) -> Option<&TableStar> {
        self.stars.get(usize::from(index))
    }

    /// The angle between the stars at `a` and `b`.
    pub(crate) fn angle(&self, a: u16, b: u16) -> Option<f64> {
        Some(self.star(a)?.unit.angle(&self.star(b)?.unit))
    }

    /// The pairs with an angle inside `angle_rad ± tolerance_rad`, found by
    /// binary search.
    pub(crate) fn within(&self, angle_rad: f64, tolerance_rad: f64) -> &[Pair] {
        let low = (angle_rad - tolerance_rad) as f32;
        let high = (angle_rad + tolerance_rad) as f32;
        let start = self.pairs.partition_point(|p| p.angle_rad < low);
        let end = self.pairs.partition_point(|p| p.angle_rad <= high);
        self.pairs.get(start..end.max(start)).unwrap_or(&[])
    }
}
