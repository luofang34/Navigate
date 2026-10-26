use super::*;

const PHONE: [u32; 2] = [1600, 1200];
const PHONE_FOCAL: f64 = 1202.0;
const DRONE: [u32; 2] = [640, 360];
const DRONE_FOCAL: f64 = 459.0;

fn index_over(region: &Region) -> Vec<(f64, Vec<Cell>)> {
    let all = cells(&region.grown(20_000.0));
    vec![(4_000.0, all.clone()), (160.0, all)]
}

fn request(region: Region, image_px: [u32; 2], focal_px: f64) -> SearchRequest {
    SearchRequest {
        region,
        image_px,
        focal_px,
        heading: None,
        height: None,
        pitch: None,
    }
}

#[test]
fn an_aircraft_photo_uses_the_kilometre_crops_and_a_drone_frame_the_small_ones() {
    let region = Region::Circle {
        centre: [40.9, -75.1],
        radius_m: 10_000.0,
    };
    let scales = index_over(&region);
    let mut aircraft = request(region.clone(), PHONE, PHONE_FOCAL);
    aircraft.height = Some(HeightPrior {
        m: 1_000.0,
        sigma_m: 100.0,
    });
    aircraft.pitch = Some(PitchPrior {
        deg: 21.0,
        sigma_deg: 3.0,
    });
    let plan_a = plan(&aircraft, &scales);
    assert_eq!(plan_a.scales.len(), 1);
    assert_eq!(plan_a.scales[0].side_m, 4_000.0);
    let mut drone = request(region, DRONE, DRONE_FOCAL);
    drone.height = Some(HeightPrior {
        m: 100.0,
        sigma_m: 5.0,
    });
    drone.pitch = Some(PitchPrior {
        deg: 88.0,
        sigma_deg: 2.0,
    });
    let plan_d = plan(&drone, &scales);
    assert_eq!(plan_d.scales.len(), 1);
    assert_eq!(plan_d.scales[0].side_m, 160.0);
    assert_eq!(plan_d.fallback, Fallback::None);
}

#[test]
fn unknown_priors_give_more_hypotheses_and_known_ones_fewer() {
    let region = Region::Circle {
        centre: [40.9, -75.1],
        radius_m: 5_000.0,
    };
    let scales = index_over(&region);
    let open = plan(&request(region.clone(), PHONE, PHONE_FOCAL), &scales);
    assert_eq!(
        open.scales.len(),
        2,
        "without height both crop sizes are plausible"
    );
    assert!(open.scales.iter().all(|s| s.headings_deg.len() == 8));
    let mut known = request(region, PHONE, PHONE_FOCAL);
    known.heading = Some(HeadingPrior {
        deg: 210.0,
        sigma_deg: 10.0,
    });
    known.height = Some(HeightPrior {
        m: 900.0,
        sigma_m: 50.0,
    });
    known.pitch = Some(PitchPrior {
        deg: 20.0,
        sigma_deg: 2.0,
    });
    let narrow = plan(&known, &scales);
    assert_eq!(narrow.canvases, 1);
    assert!(narrow.canvases * 10 < open.canvases);
    assert!(
        narrow.scales[0].verify < open.scales[0].verify,
        "an unknown heading checks more places"
    );
}

#[test]
fn a_wide_heading_prior_gives_a_fan_around_it() {
    let fan = headings(Some(HeadingPrior {
        deg: 90.0,
        sigma_deg: 40.0,
    }));
    assert!(fan.contains(&90.0));
    assert!(fan.len() >= 3 && fan.len() < 8, "{fan:?}");
}

#[test]
fn a_region_outside_the_index_falls_back_by_its_size() {
    let indexed = Region::Circle {
        centre: [40.9, -75.1],
        radius_m: 5_000.0,
    };
    let scales = index_over(&indexed);
    let far_small = Region::Circle {
        centre: [52.98, 36.05],
        radius_m: 800.0,
    };
    assert_eq!(
        plan(&request(far_small, PHONE, PHONE_FOCAL), &scales).fallback,
        Fallback::DenseGrid
    );
    let far_large = Region::Circle {
        centre: [52.98, 36.05],
        radius_m: 10_000.0,
    };
    assert!(matches!(
        plan(&request(far_large, PHONE, PHONE_FOCAL), &scales).fallback,
        Fallback::Uncovered { .. }
    ));
}

#[test]
fn the_footprint_of_a_straight_down_view_is_centred_below_the_camera() {
    let f = Footprint::of(DRONE, DRONE_FOCAL, 100.0, 90.0);
    assert!((f.near_m + f.far_m).abs() < 1.0, "{f:?}");
    assert!((f.far_m - f.near_m - 78.0).abs() < 3.0, "{f:?}");
    let oblique = Footprint::of(PHONE, PHONE_FOCAL, 1_000.0, 21.0);
    assert!(
        (oblique.near_m - 916.0).abs() < 5.0 && oblique.far_m == 6_000.0,
        "{oblique:?}"
    );
}
