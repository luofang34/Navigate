#![allow(clippy::expect_used)]
use super::*;
use gdal::{DriverManager, raster::Buffer};

#[test]
fn source_pixels_and_nodata_survive_reprojection() {
    let tile = Tile(16, 19214, 24677);
    let step = 40_075_016.685_578_49 / (2_f64.powi(16) * 512.0);
    let mut dataset = DriverManager::get_driver_by_name("MEM")
        .expect("driver")
        .create_with_band_type::<u8, _>("", 514, 514, 3)
        .expect("dataset");
    dataset
        .set_geo_transform(&[
            (f64::from(tile.1) * 512.0 - 1.0) * step - 20_037_508.342_789_244,
            step,
            0.0,
            20_037_508.342_789_244 - (f64::from(tile.2) * 512.0 - 1.0) * step,
            0.0,
            -step,
        ])
        .expect("transform");
    dataset
        .set_spatial_ref(&SpatialRef::from_epsg(3857).expect("projection"))
        .expect("projection");
    for index in 1..=3 {
        let mut pixels = vec![(index * 50) as u8; 514 * 514];
        pixels[257 * 514 + 257] = 0;
        let mut band = dataset.rasterband(index).expect("band");
        band.write((0, 0), (514, 514), &mut Buffer::new((514, 514), pixels))
            .expect("write");
        band.set_no_data_value(Some(0.0)).expect("nodata");
    }
    let mut raster = GdalRaster::from_dataset(dataset).expect("raster");
    let image = raster.tile_blocking(tile).expect("render");
    assert_eq!(image.get_pixel(20, 20).0, [50, 100, 150, 255]);
    assert_eq!(image.get_pixel(256, 256)[3], 0);
    let outside = raster
        .tile_blocking(Tile(16, 19220, 24680))
        .expect("outside");
    assert!(outside.pixels().all(|p| p[3] == 0));
}
