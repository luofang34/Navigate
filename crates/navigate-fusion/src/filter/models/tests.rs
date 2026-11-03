use super::*;

#[test]
fn range_jacobian_points_from_station_to_vehicle() {
    let model = RangeModel::new(Vector3::new(1000.0, 0.0, 0.0), 1000.0, 25.0);
    let x = Vec6::zeros();
    let h = model.jacobian(&x);
    assert!(h.is_some_and(|h| (h[(0, 0)] + 1.0).abs() < 1e-12));
    assert_eq!(model.innovation(&x)[0], 0.0);
}

#[test]
fn range_jacobian_is_undefined_at_the_station() {
    let model = RangeModel::new(Vector3::zeros(), 0.0, 25.0);
    assert!(model.jacobian(&Vec6::zeros()).is_none());
}

mod altitude {
    use navigate_contract::GeodeticPosition;
    use navigate_geodesy::{LocalTangentPlane, NedOffset};

    use super::*;

    fn plane() -> LocalTangentPlane {
        let origin = GeodeticPosition::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 500.0);
        LocalTangentPlane::new(origin, navigate_geodesy::BodyEllipsoid::WGS84)
            .expect("plausible origin")
    }

    fn state_at(plane: &LocalTangentPlane, position: &GeodeticPosition) -> Vec6 {
        let offset = plane.to_ned(position);
        Vec6::new(offset.north_m, offset.east_m, offset.down_m, 0.0, 0.0, 0.0)
    }

    #[test]
    fn altitude_at_the_origin_reads_the_negative_down_axis() {
        let plane = plane();
        let model = AltitudeModel::new(plane, 600.0, 4.0);
        let x = Vec6::new(0.0, 0.0, -100.0, 0.0, 0.0, 0.0);
        assert!(model.innovation(&x)[0].abs() < 1e-6);
        let h = model.jacobian(&x).expect("defined");
        assert!((h[(0, 2)] + 1.0).abs() < 1e-12);
        assert!(h[(0, 0)].abs() < 1e-12 && h[(0, 1)].abs() < 1e-12);
        assert!((model.noise()[(0, 0)] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn altitude_50_km_from_the_origin_has_no_curvature_error() {
        let plane = plane();
        let on_plane = plane.from_ned(&NedOffset::new(50_000.0, 0.0, 0.0));
        let truth = GeodeticPosition::new(on_plane.latitude_rad, on_plane.longitude_rad, 500.0);
        let x = state_at(&plane, &truth);
        // A flat-plane altitude `origin altitude − down` is wrong by the
        // curvature drop, approximately d²/(2R) = 196 m.
        let flat_error_m = (500.0 - x[2]) - truth.altitude_m;
        assert!((flat_error_m.abs() - 196.0).abs() < 2.0, "{flat_error_m}");
        let model = AltitudeModel::new(plane, 500.0, 4.0);
        assert!(model.innovation(&x)[0].abs() < 1e-6);
        // The local up direction tilts north by d/R ≈ 0.0078 rad.
        let h = model.jacobian(&x).expect("defined");
        assert!((h[(0, 0)] - 50_000.0 / 6_371_000.0).abs() < 1e-4, "{h}");
        assert!(h[(0, 1)].abs() < 1e-9);
        assert!(h[(0, 2)] < -0.9999);
        let norm = (h[(0, 0)].powi(2) + h[(0, 1)].powi(2) + h[(0, 2)].powi(2)).sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
    }
}

mod magnetic {
    use nalgebra::{Matrix3, Vector3};
    use navigate_contract::GeodeticPosition;
    use navigate_geodesy::{LocalTangentPlane, NedOffset};

    use super::*;
    use crate::config::FusionConfig;
    use crate::filter::Mat6;
    use crate::rejection::RejectionReason;

    /// 50 nT anomaly at the linearization point.
    const MAP_T: f64 = 50e-9;
    /// 3 nT 1-sigma.
    const VARIANCE_T2: f64 = 9e-18;

    fn plane() -> LocalTangentPlane {
        let origin = GeodeticPosition::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 500.0);
        LocalTangentPlane::new(origin, navigate_geodesy::BodyEllipsoid::WGS84)
            .expect("plausible origin")
    }

    fn position_cov(variance_m2: f64) -> Mat6 {
        let mut p = Mat6::zeros();
        p.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&Matrix3::from_diagonal_element(variance_m2));
        p
    }

    fn map(
        linearization_point: GeodeticPosition,
        gradient_ned_t_per_m: [f64; 3],
    ) -> MapLinearization {
        MapLinearization {
            linearization_point,
            map_t_at_lin: MAP_T,
            gradient_ned_t_per_m,
            valid_radius_m: 500.0,
        }
    }

    #[test]
    fn survey_scale_gradient_near_the_origin() {
        let plane = plane();
        // 0.05 nT/m north. Ten meters north reads 0.5 nT above the map.
        let lin = map(plane.origin(), [5e-11, 0.0, 0.0]);
        let model = MagneticAnomalyModel::new(&plane, &lin, MAP_T + 5e-10, VARIANCE_T2)
            .expect("plausible point");
        let x = Vec6::new(10.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(model.innovation(&x)[0].abs() < 1e-22);
        let h = model.jacobian(&x).expect("linear");
        assert!((h[(0, 0)] - 5e-11).abs() < 1e-24);
        assert!(h[(0, 1)].abs() < 1e-24 && h[(0, 2)].abs() < 1e-24);
        assert!((model.noise()[(0, 0)] - VARIANCE_T2).abs() < 1e-30);
        // η = 25 · (5e-11)² / 9e-18 ≈ 6.9e-3, above the 1e-3 floor.
        let config = FusionConfig::default();
        assert!(model.admissible(&x, &position_cov(25.0), &config).is_ok());
    }

    #[test]
    fn gradient_50_km_from_the_origin_is_rotated_into_the_plane() {
        let plane = plane();
        let lin_point = plane.from_ned(&NedOffset::new(48_000.0, 3_000.0, 0.0));
        let local_lin = LocalTangentPlane::new(lin_point, navigate_geodesy::BodyEllipsoid::WGS84)
            .expect("plausible point");
        let g_local = [3e-11, -2e-11, 4e-11];
        let truth = local_lin.from_ned(&NedOffset::new(200.0, -150.0, -30.0));
        let local_offset = local_lin.to_ned(&truth);
        let g = Vector3::from(g_local);
        let local_offset_v = Vector3::new(
            local_offset.north_m,
            local_offset.east_m,
            local_offset.down_m,
        );
        let measured = MAP_T + g.dot(&local_offset_v);
        let model =
            MagneticAnomalyModel::new(&plane, &map(lin_point, g_local), measured, VARIANCE_T2)
                .expect("plausible point");
        let truth_ned = plane.to_ned(&truth);
        let lin_ned = plane.to_ned(&lin_point);
        let x = Vec6::new(
            truth_ned.north_m,
            truth_ned.east_m,
            truth_ned.down_m,
            0.0,
            0.0,
            0.0,
        );
        assert!(
            model.innovation(&x)[0].abs() < 1e-19,
            "{}",
            model.innovation(&x)
        );
        // Without the rotation, the plane offset would mix the axes and
        // leave an error far above the rounding level.
        let plane_offset = Vector3::new(
            truth_ned.north_m - lin_ned.north_m,
            truth_ned.east_m - lin_ned.east_m,
            truth_ned.down_m - lin_ned.down_m,
        );
        assert!((g.dot(&plane_offset) - g.dot(&local_offset_v)).abs() > 1e-12);
        let h = model.jacobian(&x).expect("linear");
        let h3 = Vector3::new(h[(0, 0)], h[(0, 1)], h[(0, 2)]);
        assert!((h3.norm() - g.norm()).abs() < 1e-24);
        assert!((h3 - g).norm() > 1e-14);
    }

    #[test]
    fn large_covariance_exceeds_the_linearization_region() {
        let plane = plane();
        let model = MagneticAnomalyModel::new(
            &plane,
            &map(plane.origin(), [5e-11, 0.0, 0.0]),
            MAP_T,
            VARIANCE_T2,
        )
        .expect("plausible point");
        // 1-sigma 1 km; the 2-sigma margin is 2 km against a 500 m radius.
        let reason = model
            .admissible(&Vec6::zeros(), &position_cov(1e6), &FusionConfig::default())
            .expect_err("outside the region");
        assert!(matches!(
            reason,
            RejectionReason::LinearizationRegionExceeded { distance_plus_ksigma_m, valid_radius_m }
                if (distance_plus_ksigma_m - 2_000.0).abs() < 1e-6 && valid_radius_m == 500.0
        ));
    }

    #[test]
    fn large_offset_with_small_covariance_exceeds_the_linearization_region() {
        let plane = plane();
        let lin_point = plane.from_ned(&NedOffset::new(1_000.0, 0.0, 0.0));
        let model = MagneticAnomalyModel::new(
            &plane,
            &map(lin_point, [5e-11, 0.0, 0.0]),
            MAP_T,
            VARIANCE_T2,
        )
        .expect("plausible point");
        let reason = model
            .admissible(
                &Vec6::zeros(),
                &position_cov(25.0),
                &FusionConfig::default(),
            )
            .expect_err("outside the region");
        assert!(matches!(
            reason,
            RejectionReason::LinearizationRegionExceeded { distance_plus_ksigma_m, .. }
                if (distance_plus_ksigma_m - 1_010.0).abs() < 0.5
        ));
    }

    #[test]
    fn flat_cell_is_an_uninformative_gradient() {
        let plane = plane();
        let model = MagneticAnomalyModel::new(
            &plane,
            &map(plane.origin(), [1e-13, 0.0, 0.0]),
            MAP_T,
            VARIANCE_T2,
        )
        .expect("plausible point");
        let reason = model
            .admissible(
                &Vec6::zeros(),
                &position_cov(25.0),
                &FusionConfig::default(),
            )
            .expect_err("uninformative");
        assert!(matches!(
            reason,
            RejectionReason::UninformativeGradient { information_ratio, floor }
                if (information_ratio - 25.0 * 1e-26 / 9e-18).abs() < 1e-12 && floor == 1e-3
        ));
    }

    #[test]
    fn implausible_limits_in_the_configuration_fail_closed() {
        let plane = plane();
        let model = MagneticAnomalyModel::new(
            &plane,
            &map(plane.origin(), [5e-11, 0.0, 0.0]),
            MAP_T,
            VARIANCE_T2,
        )
        .expect("plausible point");
        let p = position_cov(25.0);
        for k in [-1.0, f64::NAN, f64::INFINITY] {
            let config = FusionConfig {
                linearization_sigma_multiplier: k,
                ..FusionConfig::default()
            };
            assert!(matches!(
                model.admissible(&Vec6::zeros(), &p, &config),
                Err(RejectionReason::LinearizationRegionExceeded { .. })
            ));
        }
        for floor in [-1.0, f64::NAN, f64::INFINITY] {
            let config = FusionConfig {
                min_information_ratio: floor,
                ..FusionConfig::default()
            };
            assert!(matches!(
                model.admissible(&Vec6::zeros(), &p, &config),
                Err(RejectionReason::UninformativeGradient { .. })
            ));
        }
    }
}
