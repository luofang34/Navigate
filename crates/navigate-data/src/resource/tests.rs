use super::*;

#[cfg(test)]
struct Flat(f64);

impl GroundHeight for Flat {
    fn ground_m(&self, _: [f64; 2]) -> Option<f64> {
        Some(self.0)
    }
}

#[cfg(test)]
fn camera(depression_deg: f64) -> PlannedCamera {
    PlannedCamera {
        focal_px: 800.0,
        width_px: 1024,
        height_px: 768,
        depression_rad: depression_deg.to_radians(),
    }
}

#[cfg(test)]
fn request(altitude_m: f64, depression_deg: f64) -> ResourceRequest {
    ResourceRequest {
        legs: alloc::vec![PlannedLeg {
            from: [41.70, -1.05, altitude_m],
            to: [41.70, -0.80, altitude_m],
        }],
        cameras: alloc::vec![camera(depression_deg)],
        search_radius_m: 100.0,
        deviation_m: 200.0,
        max_range_m: 8_000.0,
        imagery: LayerLimits::imagery(),
        elevation: LayerLimits::terrarium(),
        elevation_offset: 3,
    }
}

#[cfg(test)]
fn imagery(plan: &ResourcePlan) -> Vec<ZoomBand> {
    plan.legs
        .iter()
        .flat_map(|l| l.bands.iter().copied())
        .filter(|b| b.layer == LayerKind::Imagery)
        .collect()
}

#[test]
fn the_zoom_rule_is_the_renderer_rule_for_256_pixel_tiles() {
    // maplibre: desired = log2(focal / (512 × d)), d in Mercator units of
    // C × cos φ metres, for 512-pixel tiles.
    let (focal, slant, lat) = (900.0_f64, 1_500.0_f64, 41.7_f64);
    let world = EQUATOR_M * cos(lat.to_radians());
    let maplibre_512 = (focal / (512.0 * (slant / world))).log2();
    assert!((texel_zoom(focal, slant, lat, 512) - maplibre_512).abs() < 1e-9);
    assert!((texel_zoom(focal, slant, lat, 256) - (maplibre_512 + 1.0)).abs() < 1e-9);
    let z = texel_zoom(focal, slant, lat, 256);
    assert!((slant_at_zoom(focal, 16, lat, 256) / slant - 2_f64.powf(z - 16.0)).abs() < 1e-9);
    // One texel is one pixel at that zoom.
    let texel = crate::tile::ground_sample_distance_m(lat, 16, 256);
    let pixel = slant_at_zoom(focal, 16, lat, 256) / focal;
    assert!((texel - pixel).abs() < 1e-9, "{texel} {pixel}");
}

#[test]
fn a_higher_leg_needs_coarser_tiles_and_costs_less_per_kilometre() {
    let low = plan_resources(&request(800.0, 90.0), &Flat(300.0)).unwrap();
    let high = plan_resources(&request(3_300.0, 90.0), &Flat(300.0)).unwrap();
    let finest = |p: &ResourcePlan| imagery(p).iter().map(|b| b.zoom).max().unwrap();
    assert!(
        finest(&high) < finest(&low),
        "{} {}",
        finest(&high),
        finest(&low)
    );
    let per_km = |p: &ResourcePlan| p.bytes / (p.legs[0].length_m / 1000.0);
    assert!(
        per_km(&high) < per_km(&low),
        "{} {}",
        per_km(&high),
        per_km(&low)
    );
    assert_eq!(low.legs[0].agl_m, 500.0);
    assert_eq!(low.legs[0].altitude_band, 2);
    assert_eq!(high.legs[0].altitude_band, 4);
}

#[test]
fn an_oblique_far_field_uses_coarser_levels_than_its_near_field() {
    let plan = plan_resources(&request(1_300.0, 20.0), &Flat(300.0)).unwrap();
    let bands = imagery(&plan);
    assert!(bands.len() >= 3, "{bands:?}");
    for pair in bands.windows(2) {
        let [fine, coarse] = [pair[0], pair[1]];
        assert!(fine.zoom > coarse.zoom);
        assert!(fine.far_m <= coarse.near_m + 1e-6, "{fine:?} {coarse:?}");
        assert!(fine.half_width_m <= coarse.half_width_m);
    }
    let nadir = plan_resources(&request(1_300.0, 90.0), &Flat(300.0)).unwrap();
    assert!(imagery(&nadir).len() <= 2, "{:?}", imagery(&nadir));
}

#[test]
fn elevation_follows_imagery_at_the_offset_and_the_strip_includes_the_margins() {
    let plan = plan_resources(&request(1_300.0, 90.0), &Flat(300.0)).unwrap();
    let leg = &plan.legs[0];
    let fine = imagery(&plan)[0];
    assert!(
        leg.bands
            .iter()
            .any(|b| b.layer == LayerKind::Elevation && b.zoom == fine.zoom - 3)
    );
    assert!(fine.half_width_m > 300.0);
    assert!(leg.bytes > 0.0 && (plan.bytes - leg.bytes).abs() < 1e-6);
}

#[test]
fn unusable_requests_are_refused() {
    struct Unknown;
    impl GroundHeight for Unknown {
        fn ground_m(&self, _: [f64; 2]) -> Option<f64> {
            None
        }
    }
    assert!(plan_resources(&request(800.0, 90.0), &Unknown).is_err());
    assert!(plan_resources(&request(800.0, 90.0), &Flat(900.0)).is_err());
    let mut no_camera = request(800.0, 90.0);
    no_camera.cameras.clear();
    assert!(plan_resources(&no_camera, &Flat(0.0)).is_err());
    let mut nan = request(800.0, 90.0);
    nan.search_radius_m = f64::NAN;
    assert!(plan_resources(&nan, &Flat(0.0)).is_err());
}
