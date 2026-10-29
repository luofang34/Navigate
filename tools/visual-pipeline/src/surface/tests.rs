use super::*;

fn intrinsics(width: u32, height: u32, focal: f64) -> Matrix3<f64> {
    Matrix3::new(
        focal,
        0.0,
        f64::from(width) * 0.5 - 0.5,
        0.0,
        focal,
        f64::from(height) * 0.5 - 0.5,
        0.0,
        0.0,
        1.0,
    )
}

fn mixed_shoreline_surface() -> Surface {
    let water_tile = RgbImage::from_pixel(256, 256, Rgb([26, 72, 146]));
    let land_tile = RgbImage::from_pixel(256, 256, Rgb([66, 122, 54]));
    let mut shore_tile = RgbImage::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let color = if x < 225 {
                Rgb([26, 72 + (y / 64) as u8, 146])
            } else if x < 228 {
                Rgb([232, 224, 196])
            } else {
                Rgb([66, 122 + (y / 64) as u8, 54])
            };
            shore_tile.put_pixel(x, y, color);
        }
    }
    let mut surface = Surface::synthetic(46.6, 7.9, 120.0, water_tile.clone());
    let (cx, cy, _, _) = mercator_tile_coords(46.6, 7.9, surface.imagery_zoom);
    for dy in -5_i32..=5 {
        for dx in -5_i32..=5 {
            let tx = (cx as i32 + dx) as u32;
            let ty = (cy as i32 + dy) as u32;
            let img = if tx < cx {
                water_tile.clone()
            } else if tx == cx {
                shore_tile.clone()
            } else {
                land_tile.clone()
            };
            let mut elev = vec![120.0_f32; 256 * 256];
            if tx > cx {
                elev.fill(220.0);
            } else if tx == cx {
                for y in 0..256 {
                    for x in 226..256 {
                        elev[(y * 256 + x) as usize] = 220.0;
                    }
                }
            }
            surface.imagery_tiles.insert((tx, ty), img);
            surface.dem_tiles.insert((tx, ty), elev);
        }
    }
    surface.min_elev_m = 120.0;
    surface.max_elev_m = 220.0;
    surface
}

#[test]
fn raycast_pixels_matches_full_surface_render_and_subpixel_lifting() {
    let surface = mixed_shoreline_surface();
    let eye = surface.local(46.6, 7.9, 1350.0);
    // Pitch near the horizon so top rows see sky and lower rows hit water/shore/land.
    let w2c = surface.camera_rotation(46.6, 7.9, 75.0, 14.0, 2.0);
    let size = [64, 48];
    let k = intrinsics(size[0], size[1], 62.0);
    let (full_rgb, full_depth, full_pts) = surface.render(eye, &w2c, &k, size);

    let mut keypoints = Vec::new();
    for y in (0..size[1]).step_by(5) {
        for x in (0..size[0]).step_by(7) {
            keypoints.push([f64::from(x), f64::from(y)]);
        }
    }
    let hits = surface.raycast_pixels(eye, &w2c, &k, &keypoints);
    assert_eq!(hits.len(), keypoints.len());

    let mut hit_count = 0_usize;
    let mut sky_count = 0_usize;
    for (&[u, v], hit) in keypoints.iter().zip(&hits) {
        let (x, y) = (u as u32, v as u32);
        let idx = (y * size[0] + x) as usize;
        if full_depth[idx] > 0.0 {
            hit_count += 1;
            let h = hit.expect("expected terrain hit matching Surface::render");
            assert_eq!(h.rgb, *full_rgb.get_pixel(x, y));
            assert!((h.depth_m - full_depth[idx]).abs() < 1e-5);
            for (a, b) in h.world_enu.into_iter().zip(full_pts[idx]) {
                assert!((a - b).abs() < 1e-6);
            }
        } else {
            sky_count += 1;
            assert!(hit.is_none(), "expected sky miss at ({x}, {y})");
        }
    }
    assert!(hit_count > 10 && sky_count > 5);

    // Subpixel keypoints project back onto their subpixel rays and lie on the DEM.
    let subpixels = [[18.35, 36.65], [32.75, 41.25], [47.125, 38.875]];
    let sub_hits = surface.raycast_pixels(eye, &w2c, &k, &subpixels);
    let eye_v = Vector3::new(eye[0], eye[1], eye[2]);
    for (&[u, v], maybe_hit) in subpixels.iter().zip(&sub_hits) {
        let h = maybe_hit.expect("expected subpixel ground hit");
        let rel = w2c * (Vector3::from(h.world_enu) - eye_v);
        let px = k[(0, 0)] * rel.x / rel.z + k[(0, 2)];
        let py = k[(1, 1)] * rel.y / rel.z + k[(1, 2)];
        assert!((px - u).abs() < 1e-6 && (py - v).abs() < 1e-6);
        let [lat, lon, alt] = surface.geodetic(h.world_enu);
        let dem = surface.sample_dem(lat, lon).unwrap();
        assert!((alt - dem).abs() < 5.0, "alt {alt} vs dem {dem}");
    }
    assert_eq!(
        surface.raycast_pixels(eye, &w2c, &k, &[[f64::NAN, 5.0]]),
        vec![None]
    );
}

#[test]
fn render_adaptive_agrees_with_full_render_and_cuts_rays_over_fourfold() {
    let surface = mixed_shoreline_surface();
    let eye = surface.local(46.6, 7.9, 1500.0);
    let size = [128, 96];

    // 1. Mixed water/shoreline/cliff terrain view with stride = 4 cuts rays > 4x.
    let w2c_ground = surface.camera_rotation(46.6, 7.9, 0.0, 55.0, -2.0);
    let k_ground = intrinsics(size[0], size[1], 120.0);
    let (full_g_rgb, full_g_depth, full_g_pts) = surface.render(eye, &w2c_ground, &k_ground, size);
    let ((adap_g_rgb, adap_g_depth, adap_g_pts), stats_ground) =
        surface.render_adaptive(eye, &w2c_ground, &k_ground, size, 4, 8, 0.04);
    assert!(
        stats_ground.rays_cast * 4 < stats_ground.total_pixels,
        "expected > 4x ray reduction on mixed terrain, cast {} of {}",
        stats_ground.rays_cast,
        stats_ground.total_pixels
    );
    for idx in (0..full_g_depth.len()).step_by(13) {
        let x = (idx as u32) % size[0];
        let y = (idx as u32) / size[0];
        let fp = full_g_rgb.get_pixel(x, y).0;
        let ap = adap_g_rgb.get_pixel(x, y).0;
        for (ac, fc) in ap.into_iter().zip(fp) {
            assert!((i16::from(ac) - i16::from(fc)).abs() <= 4);
        }
        assert!((adap_g_depth[idx] - full_g_depth[idx]).abs() / full_g_depth[idx] < 0.01);
        let enu_err = (Vector3::from(adap_g_pts[idx]) - Vector3::from(full_g_pts[idx])).norm();
        assert!(enu_err < f64::from(full_g_depth[idx]) * 0.01);
    }

    // 2. Oblique horizon view includes sky, water, bright shoreline, and elevated land.
    let w2c = surface.camera_rotation(46.6, 7.9, 0.0, 4.0, -1.0);
    let k = intrinsics(size[0], size[1], 140.0);

    let (full_rgb, full_depth, full_pts) = surface.render(eye, &w2c, &k, size);

    let ((stride1_rgb, stride1_depth, _), stride1_stats) =
        surface.render_adaptive(eye, &w2c, &k, size, 1, 8, 0.03);
    assert_eq!(stride1_stats.rays_cast, stride1_stats.total_pixels);
    assert_eq!(stride1_rgb.as_raw(), full_rgb.as_raw());
    assert_eq!(stride1_depth, full_depth);

    let ((adap_rgb, adap_depth, adap_pts), stats) =
        surface.render_adaptive(eye, &w2c, &k, size, 5, 8, 0.30);
    assert_eq!(stats.total_pixels, (size[0] * size[1]) as usize);
    assert!(
        stats.rays_cast * 4 < stats.total_pixels,
        "expected > 4x ray reduction, cast {} of {}",
        stats.rays_cast,
        stats.total_pixels
    );

    let (w, h) = (size[0], size[1]);
    let mut horizon_transitions = 0_usize;
    let mut shoreline_edges = 0_usize;

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = (y * w + x) as usize;
            let fd = full_depth[idx];
            let ad = adap_depth[idx];
            if fd == 0.0 {
                assert_eq!(ad, 0.0, "sky pixel ({x}, {y}) should remain zero depth");
                assert_eq!(*adap_rgb.get_pixel(x, y), Rgb([0, 0, 0]));
                if full_depth[((y + 1) * w + x) as usize] > 0.0 {
                    horizon_transitions += 1;
                }
                continue;
            }

            let fp = full_rgb.get_pixel(x, y).0;
            let ap = adap_rgb.get_pixel(x, y).0;
            let right_p = full_rgb.get_pixel(x + 1, y).0;
            if (i16::from(fp[0]) - i16::from(right_p[0])).abs() > 60 {
                shoreline_edges += 1;
                assert_eq!(
                    ap, fp,
                    "shoreline edge pixel ({x}, {y}) must cast exact ray"
                );
            }
            for (ac, fc) in ap.into_iter().zip(fp) {
                assert!(
                    (i16::from(ac) - i16::from(fc)).abs() <= 4,
                    "rgb mismatch at ({x}, {y}): {ap:?} vs {fp:?}"
                );
            }
            let rel_err = (ad - fd).abs() / fd;
            assert!(
                rel_err < 0.015,
                "depth mismatch at ({x}, {y}): {ad} vs {fd}"
            );
            let enu_err = (Vector3::from(adap_pts[idx]) - Vector3::from(full_pts[idx])).norm();
            assert!(
                enu_err < fd as f64 * 0.015,
                "world_enu mismatch at ({x}, {y}): {enu_err} m at depth {fd} m"
            );
        }
    }
    assert!(
        horizon_transitions > 20,
        "expected visible sky/horizon transition"
    );
    assert!(
        shoreline_edges > 10,
        "expected visible high-gradient shoreline"
    );
}
