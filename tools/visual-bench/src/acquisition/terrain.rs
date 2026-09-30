use crate::package::MapPackage;
pub(super) struct Terrain {
    frame: navigate_visual::LocalFrame,
    tiles: Vec<([u32; 3], image::RgbaImage)>,
}
impl Terrain {
    pub fn new(package: &MapPackage) -> Self {
        let mut tiles: Vec<_> = package
            .tiles
            .iter()
            .filter_map(|t| t.elevation.as_ref().map(|image| (t.xyz, image.clone())))
            .collect();
        tiles.sort_by(|a, b| b.0[0].cmp(&a.0[0]));
        Self {
            frame: package.frame,
            tiles,
        }
    }
    pub fn elevation(&self, east: f64, north: f64) -> Option<f64> {
        let [lat, lon, _] = self
            .frame
            .geodetic(nalgebra::Vector3::new(east, north, 0.0));
        for ([z, x, y], tile) in &self.tiles {
            let n = 2_f64.powf(f64::from(*z));
            let px = ((lon + 180.0) / 360.0 * n - f64::from(*x)) * 256.0;
            let py = ((1.0 - lat.to_radians().tan().asinh() / std::f64::consts::PI) / 2.0 * n
                - f64::from(*y))
                * 256.0;
            if !(0.0..256.0).contains(&px) || !(0.0..256.0).contains(&py) {
                continue;
            }
            let p = tile.get_pixel(px.floor() as u32, py.floor() as u32).0;
            if p[3] == 255 {
                return Some(
                    f64::from(p[0]) * 256.0 + f64::from(p[1]) + f64::from(p[2]) / 256.0 - 32768.0,
                );
            }
        }
        None
    }
}

#[cfg(test)]
mod tests;
