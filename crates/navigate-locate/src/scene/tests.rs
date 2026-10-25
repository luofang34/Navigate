use super::*;

/// A tile source with one encoded image for every tile of a kind.
struct Constant {
    imagery: Vec<u8>,
    terrain: Vec<u8>,
    missing: Option<(u32, u32)>,
}

impl TileSource for Constant {
    async fn fetch(
        &self,
        kind: TileKind,
        _z: u32,
        x: u32,
        y: u32,
    ) -> Result<Option<Vec<u8>>, String> {
        if self.missing == Some((x, y)) {
            return Ok(None);
        }
        Ok(Some(match kind {
            TileKind::Imagery => self.imagery.clone(),
            TileKind::Terrain => self.terrain.clone(),
        }))
    }
    fn identity(&self) -> String {
        "constant".into()
    }
}

fn png(image: RgbImage) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(image)
        .write_to(&mut out, image::ImageFormat::Png)
        .unwrap();
    out.into_inner()
}

fn request(max_bytes: usize) -> SceneRequest {
    SceneRequest {
        anchor: [52.982, 36.06],
        radius_m: 300.0,
        imagery_zoom: 17,
        terrain_zoom: 13,
        imagery_tile_pixels: 64,
        max_bytes,
        concurrency: 4,
    }
}

fn source() -> Constant {
    // Terrarium 190 m: R·256 + G + B/256 − 32768 = 32958, so R = 128 and G = 190.
    let terrain = RgbImage::from_pixel(256, 256, image::Rgb([128, 190, 0]));
    let imagery = RgbImage::from_fn(256, 256, |x, y| image::Rgb([x as u8, y as u8, 7]));
    Constant {
        imagery: png(imagery),
        terrain: png(terrain),
        missing: None,
    }
}

#[test]
fn the_scene_holds_every_tile_and_the_terrarium_height() {
    let mut last = (0, 0);
    let scene = pollster::block_on(build(&source(), &request(64 << 20), |done, total| {
        last = (done, total)
    }))
    .unwrap();
    assert!(scene.imagery.coverage.iter().all(|&c| c == 255));
    assert_eq!(height_at(&scene.terrain, [52.982, 36.06]), Some(190.0));
    assert_eq!(last.0, last.1, "the last progress call reports all tiles");
}

#[test]
fn a_missing_tile_leaves_no_coverage() {
    let mut constant = source();
    let grid = TileGrid::around([52.982, 36.06], 300.0, 17);
    constant.missing = Some((grid.x[0], grid.y[0]));
    let scene = pollster::block_on(build(&constant, &request(64 << 20), |_, _| {})).unwrap();
    assert!(scene.imagery.coverage.contains(&0));
}

#[test]
fn an_area_over_the_memory_limit_is_refused_before_fetching() {
    let result = pollster::block_on(build(&source(), &request(1000), |_, _| {}));
    assert!(matches!(
        result,
        Err(LocateError::Memory { limit: 1000, .. })
    ));
}

#[test]
fn a_tile_that_is_not_an_image_names_the_tile() {
    let constant = Constant {
        imagery: b"no".to_vec(),
        ..source()
    };
    let result = pollster::block_on(build(&constant, &request(64 << 20), |_, _| {}));
    assert!(matches!(
        result,
        Err(LocateError::Tile {
            kind: "imagery",
            ..
        })
    ));
}
