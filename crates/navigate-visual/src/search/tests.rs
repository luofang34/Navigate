#![allow(clippy::expect_used)]
use super::*;
use nalgebra::UnitQuaternion;

fn prior(sigma_m: f64) -> SearchPrior {
    SearchPrior {
        center: CameraPose {
            position: Vector3::new(100.0, -50.0, 900.0),
            orientation: UnitQuaternion::identity(),
        },
        position_covariance_m2: Matrix3::from_diagonal(&Vector3::new(
            sigma_m * sigma_m,
            sigma_m * sigma_m,
            25.0,
        )),
        attitude_sigma_rad: 0.02,
    }
}

#[test]
fn uncertainty_selects_the_tier() {
    let config = SearchConfig::default();
    assert_eq!(prior(5.0).tier(&config), SearchTier::Local);
    assert_eq!(prior(100.0).tier(&config), SearchTier::Region);
    assert_eq!(prior(5_000.0).tier(&config), SearchTier::Global);
    assert!(prior(5_000.0).candidates(&config).is_empty());
    let local = prior(5.0).candidates(&config);
    assert_eq!(local.len(), 1);
    assert_eq!(local[0].position, prior(5.0).center.position);
}

#[test]
fn region_candidates_are_bounded_ordered_and_inside_the_ellipse() {
    let config = SearchConfig::default();
    let p = prior(100.0);
    let candidates = p.candidates(&config);
    assert!(!candidates.is_empty() && candidates.len() <= config.max_candidates);
    assert_eq!(
        candidates[0].position, p.center.position,
        "the center is first"
    );
    let inverse = p
        .position_covariance_m2
        .fixed_view::<2, 2>(0, 0)
        .into_owned()
        .try_inverse()
        .expect("invertible");
    let distances: Vec<f64> = candidates
        .iter()
        .map(|c| {
            let d = (c.position - p.center.position).xy();
            (d.transpose() * inverse * d)[(0, 0)]
        })
        .collect();
    assert!(distances.windows(2).all(|w| w[0] <= w[1] + 1e-12));
    let limit = config.sigma_scale * config.sigma_scale;
    assert!(distances.iter().all(|d| *d <= limit + 1e-9));
}

#[test]
fn a_singular_or_non_finite_covariance_is_invalid() {
    let mut p = prior(10.0);
    p.position_covariance_m2 = Matrix3::zeros();
    assert!(p.validate().is_err());
    let mut p = prior(10.0);
    p.position_covariance_m2[(0, 0)] = f64::NAN;
    assert!(p.validate().is_err());
    assert!(prior(10.0).validate().is_ok());
}
