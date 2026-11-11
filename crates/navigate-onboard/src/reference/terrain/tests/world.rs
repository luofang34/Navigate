//! The world of the tests: a package of synthetic imagery and relief in a
//! directory store, and a camera that renders frames over it.

use std::collections::BTreeSet;
use std::io::Cursor;
use std::sync::Arc;

use image::{GrayImage, ImageFormat, Rgb, RgbImage};
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use navigate_camera::CameraMount;
use navigate_data::corridor::corridor_tiles;
use navigate_data::product::{KindName, ProductKind, ValidityEnvelope, open_chunk, write_list};
use navigate_data::tile::tile_lat_lon;
use navigate_data::{
    Accuracy, DayNumber, Descriptor, DirectoryStore, Encoding, HorizontalDatum, LayerKey,
    LayerKind, LayerSpec, Package, PackageWriter, PlannedLeg, Revision, Sha256Digest, SourceKey,
    SurfaceModel, TileAddress, VerticalDatum,
};
use navigate_visual::{CameraModel, CameraPose, LocalFrame};
use navigate_visual_render::{
    KeyframeSpec, SceneRequest, SparseSceneSampler, build_keyframes, scene_around,
};

/// Centre of the world, `[lat, lon]` in degrees.
pub(crate) const CENTRE: [f64; 2] = [41.70, -1.00];
/// Imagery zoom of the world: 3.6 m pixels at its latitude.
mod elevation_error;
mod texture;
pub(crate) use elevation_error::DemError;
pub(crate) use texture::Texture;

pub(crate) const IMAGERY_ZOOM: u32 = 15;
/// Elevation zoom of the world.
pub(crate) const ELEVATION_ZOOM: u32 = 12;
/// Zoom of the correlation cells of the imagery: 57 m cells, so that a
/// short flight crosses some.
pub(crate) const CORRELATION_ZOOM: u32 = 19;
/// Mean height of the ground above the ellipsoid, metres.
pub(crate) const GROUND_M: f64 = 250.0;
/// The day of the flight.
pub(crate) const TODAY: DayNumber = DayNumber(9_800);

pub(crate) const IMAGERY: LayerKey = LayerKey {
    kind: LayerKind::Imagery,
    source: SourceKey::Synthetic,
};
pub(crate) const ELEVATION: LayerKey = LayerKey {
    kind: LayerKind::Elevation,
    source: SourceKey::Synthetic,
};

/// How the layers of a test world are declared.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WorldSpec {
    /// Stated accuracy: imagery horizontal and elevation vertical, mm.
    pub accuracy_mm: Option<(u32, u32)>,
    pub imagery_encoding: Encoding,
    pub imagery_tile_pixels: u32,
    pub elevation: bool,
    pub vertical_datum: Option<VerticalDatum>,
    /// Half width of the imagery about [`CENTRE`], metres.
    pub radius_m: f64,
    /// Zoom of the correlation cells of the imagery.
    pub correlation_zoom: u32,
    /// Zoom of the imagery.
    pub imagery_zoom: u32,
    /// Height of the hills from valley to top, metres.
    pub relief_m: f64,
    /// The error of the elevation of the package against the ground of
    /// the frames.
    pub dem_error: DemError,
    /// What the ground looks like.
    pub texture: Texture,
}

impl Default for WorldSpec {
    fn default() -> Self {
        Self {
            accuracy_mm: Some((1_500, 2_000)),
            imagery_encoding: Encoding::Png,
            imagery_tile_pixels: 256,
            elevation: true,
            vertical_datum: Some(VerticalDatum::Ellipsoidal),
            radius_m: 3_000.0,
            correlation_zoom: CORRELATION_ZOOM,
            imagery_zoom: IMAGERY_ZOOM,
            relief_m: 80.0,
            dem_error: DemError::None,
            texture: Texture::Farmland,
        }
    }
}

/// A package in a directory store.
pub(crate) struct World {
    pub spec: WorldSpec,
    /// The directory of the store; it lives as long as the world.
    pub dir: tempfile::TempDir,
    pub store: Arc<DirectoryStore>,
    pub package: Sha256Digest,
}

/// The keyframes of a test world: a leg to the north through the centre,
/// at the height of the flights, for the nadir camera of the tests.
pub(crate) struct Keyframes {
    /// The digest of the product list that names the keyframe chunk.
    pub list: Sha256Digest,
    /// The planned positions, `[lat, lon, height]`.
    pub poses: Vec<[f64; 3]>,
}

/// Distance between the keyframes of a test world, metres.
pub(crate) const KEYFRAME_SPACING_M: f64 = 100.0;
/// The envelope of each keyframe of a test world.
pub(crate) const ENVELOPE: ValidityEnvelope = ValidityEnvelope {
    lateral_mm: 60_000,
    vertical_mm: 40_000,
    heading_urad: 150_000,
};

/// A value in `[0, 1)` for each lattice point.
pub(crate) fn lattice(x: i64, y: i64, seed: u64) -> f64 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ seed;
    h ^= h >> 31;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^= h >> 32;
    (h >> 11) as f64 / (1_u64 << 53) as f64
}

/// Smooth noise with features of `cell` pixels.
fn noise(x: f64, y: f64, cell: f64, seed: u64) -> f64 {
    let (fx, fy) = (x / cell, y / cell);
    let (ix, iy) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - ix, fy - iy);
    let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let at = |dx: i64, dy: i64| lattice(ix as i64 + dx, iy as i64 + dy, seed);
    (1.0 - sy) * ((1.0 - sx) * at(0, 0) + sx * at(1, 0))
        + sy * ((1.0 - sx) * at(0, 1) + sx * at(1, 1))
}

/// The luminance of the ground at one global imagery pixel: fields and
/// blocks of 50 m to 250 m, which a camera with 5 m pixels resolves.

/// The height of the ground above the ellipsoid at one global elevation
/// pixel: hills of `relief_m` from valley to top, a kilometre apart.
pub(crate) fn ground_height_m(gx: f64, gy: f64, relief_m: f64) -> f64 {
    GROUND_M + relief_m * (noise(gx, gy, 40.0, 7) - 0.5)
}

fn png(image: &RgbImage) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    image.write_to(&mut out, ImageFormat::Png).unwrap();
    out.into_inner()
}

fn imagery_tile(a: TileAddress, pixels: u32, texture: Texture) -> Vec<u8> {
    png(&RgbImage::from_fn(pixels, pixels, |x, y| {
        let v = texture.luma(f64::from(a.x * pixels + x), f64::from(a.y * pixels + y));
        Rgb([v, v, v])
    }))
}

fn elevation_tile(a: TileAddress, relief_m: f64, error: DemError) -> Vec<u8> {
    png(&RgbImage::from_fn(256, 256, |x, y| {
        let (gx, gy) = (f64::from(a.x * 256 + x), f64::from(a.y * 256 + y));
        let h = ground_height_m(gx, gy, relief_m) + error.at(gx, gy) + 32_768.0;
        Rgb([
            (h / 256.0) as u8,
            (h % 256.0) as u8,
            (h.fract() * 256.0) as u8,
        ])
    }))
}

fn layer(key: LayerKey, spec: &WorldSpec) -> LayerSpec {
    let imagery = key.kind == LayerKind::Imagery;
    let accuracy = spec
        .accuracy_mm
        .map_or(Accuracy::unknown(), |(h, v)| Accuracy {
            horizontal_mm: imagery.then_some(h),
            vertical_mm: (!imagery).then_some(v),
            acquired: Some(DayNumber(TODAY.0 - 400)),
            age_growth_mm_per_year: Some(0),
            surface: SurfaceModel::BareEarth,
        });
    LayerSpec {
        key,
        revision: Revision::Release(1),
        horizontal_datum: HorizontalDatum::Wgs84,
        vertical_datum: if imagery { None } else { spec.vertical_datum },
        encoding: if imagery {
            spec.imagery_encoding
        } else {
            Encoding::TerrariumPng
        },
        tile_pixels: if imagery {
            spec.imagery_tile_pixels
        } else {
            256
        },
        accuracy,
        correlation_zoom: if imagery { spec.correlation_zoom } else { 16 },
    }
}

impl World {
    /// Writes the package of `spec` into a new temporary directory.
    pub(crate) fn build(spec: &WorldSpec) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut store = DirectoryStore::new(dir.path());
        let mut writer = PackageWriter::new();
        writer.add_layer(layer(IMAGERY, spec)).unwrap();
        let mut tiles = BTreeSet::new();
        corridor_tiles(&[CENTRE], spec.radius_m, spec.imagery_zoom, &mut tiles).unwrap();
        for a in tiles {
            let tile = imagery_tile(a, spec.imagery_tile_pixels, spec.texture);
            pollster::block_on(writer.add_tile(&mut store, IMAGERY, a, &tile)).unwrap();
        }
        if spec.elevation {
            writer.add_layer(layer(ELEVATION, spec)).unwrap();
            let mut tiles = BTreeSet::new();
            corridor_tiles(
                &[CENTRE],
                spec.radius_m + 2_000.0,
                ELEVATION_ZOOM,
                &mut tiles,
            )
            .unwrap();
            for a in tiles {
                let tile = elevation_tile(a, spec.relief_m, spec.dem_error);
                pollster::block_on(writer.add_tile(&mut store, ELEVATION, a, &tile)).unwrap();
            }
        }
        let package = pollster::block_on(writer.seal(&mut store, &Descriptor::default())).unwrap();
        Self {
            spec: *spec,
            dir,
            store: Arc::new(store),
            package,
        }
    }

    /// Builds keyframes for a camera of `camera` at `height_m` along a
    /// leg of 300 m to the north through the centre, and the product list
    /// of the package.
    pub(crate) fn keyframes(&self, camera: CameraModel, height_m: f64) -> Keyframes {
        let mut store = (*self.store).clone();
        let half = 150.0 / 6_364_000.0_f64;
        let [lat, lon] = CENTRE;
        let leg = PlannedLeg {
            from: [lat - half.to_degrees(), lon, height_m],
            to: [lat + half.to_degrees(), lon, height_m],
        };
        let spec = KeyframeSpec {
            imagery_layer: IMAGERY,
            imagery_zoom: self.spec.imagery_zoom,
            elevation_layer: ELEVATION,
            elevation_zoom: ELEVATION_ZOOM,
            camera,
            depression_rad: core::f64::consts::FRAC_PI_2,
            spacing_m: KEYFRAME_SPACING_M,
            envelope: ENVELOPE,
            far_m: 2_500.0,
            steps: 16,
            refinements: 10,
            stride: 4,
            luma_edge: u8::MAX,
            depth_edge: 0.02,
            altitude_band: 0,
        };
        let package = self.open();
        let (chunk, count) =
            pollster::block_on(build_keyframes(&package, &mut store, &[leg], &spec, None)).unwrap();
        let kind = KindName::Known(ProductKind::Keyframes);
        let list =
            pollster::block_on(write_list(&mut store, self.package, &[(kind, chunk)])).unwrap();
        let chunk = pollster::block_on(open_chunk(&store, chunk)).unwrap();
        let poses: Vec<[f64; 3]> = chunk
            .entries
            .iter()
            .map(|e| {
                let ([lat, lon], height, _) = e.pose.unwrap().to_si();
                [lat, lon, height]
            })
            .collect();
        assert_eq!(poses.len(), count);
        Keyframes { list, poses }
    }

    /// The opened package, for a builder of derived products.
    pub(crate) fn open(&self) -> Package<navigate_data::DirectoryFile> {
        pollster::block_on(Package::open(&*self.store, self.package)).unwrap()
    }
}

/// East, north, and up of a point, as columns, in the Earth-fixed axes.
fn enu_axes(lat_deg: f64, lon_deg: f64) -> Matrix3<f64> {
    let (sp, cp) = lat_deg.to_radians().sin_cos();
    let (sl, cl) = lon_deg.to_radians().sin_cos();
    Matrix3::from_columns(&[
        Vector3::new(-sl, cl, 0.0),
        Vector3::new(-sp * cl, -sp * sl, cp),
        Vector3::new(cp * cl, cp * sl, sp),
    ])
}

/// The Earth-fixed position of a WGS84 point.
fn ecef([lat_deg, lon_deg, h]: [f64; 3]) -> Vector3<f64> {
    let (a, e2) = (6_378_137.0, 6.694_379_990_141_316e-3);
    let (sp, cp) = lat_deg.to_radians().sin_cos();
    let (sl, cl) = lon_deg.to_radians().sin_cos();
    let n = a / (1.0 - e2 * sp * sp).sqrt();
    Vector3::new(
        (n + h) * cp * cl,
        (n + h) * cp * sl,
        (n * (1.0 - e2) + h) * sp,
    )
}

/// The true pose, in the tangent frame at `anchor`, of a camera on a
/// vehicle whose body reference point is at `body` (`[lat, lon]` in
/// degrees, height in metres) with the body-to-NED attitude `body_to_ned`.
/// The computation goes through Earth-fixed axes with its own formulas, so
/// it does not share the frame conversions of the navigator.
pub(crate) fn true_camera(
    anchor: [f64; 2],
    body: [f64; 3],
    body_to_ned: &UnitQuaternion<f64>,
    mount: &CameraMount,
) -> CameraPose {
    let enu = enu_axes(body[0], body[1]);
    let ecef_from_ned = Matrix3::from_columns(&[
        enu.column(1).into_owned(),
        enu.column(0).into_owned(),
        -enu.column(2),
    ]);
    let ecef_from_body = ecef_from_ned * body_to_ned.to_rotation_matrix().into_inner();
    let eye = ecef(body) + ecef_from_body * mount.lever_arm_body_m;
    // Camera axes are right, down, forward; eye axes are right, up, back.
    let body_from_camera = mount
        .cam_from_body
        .inverse()
        .to_rotation_matrix()
        .into_inner();
    let flip = Matrix3::from_diagonal(&Vector3::new(1.0, -1.0, -1.0));
    let frame_from_ecef = enu_axes(anchor[0], anchor[1]).transpose();
    let orientation = frame_from_ecef * ecef_from_body * body_from_camera * flip;
    CameraPose {
        position: frame_from_ecef * (eye - ecef([anchor[0], anchor[1], 0.0])),
        orientation: UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(
            orientation,
        )),
    }
}

/// The camera of the tests: 160 by 120 pixels, 56° across.
pub(crate) fn camera_model() -> CameraModel {
    CameraModel {
        width: 160,
        height: 120,
        fx: 150.0,
        fy: 150.0,
        cx: 79.5,
        cy: 59.5,
    }
}

/// The frame that `camera` on a vehicle at `body` sees: the renderer
/// casts one ray for each second pixel over the package, and the sensor
/// adds a gain, an offset, and noise of two counts.
pub(crate) fn frame_at(
    world: &World,
    camera: &CameraModel,
    body: [f64; 3],
    body_to_ned: &UnitQuaternion<f64>,
    mount: &CameraMount,
    seed: u64,
) -> Vec<u8> {
    frame_with_rays(world, camera, body, body_to_ned, mount, (seed, 2))
}

/// The frame of [`frame_at`] with one ray for each block of `stride`
/// pixels on a side: a stride of one casts a ray for each pixel.
pub(crate) fn frame_with_rays(
    world: &World,
    camera: &CameraModel,
    body: [f64; 3],
    body_to_ned: &UnitQuaternion<f64>,
    mount: &CameraMount,
    (seed, stride): (u64, u32),
) -> Vec<u8> {
    let request = SceneRequest {
        imagery_layer: IMAGERY,
        imagery_zoom: world.spec.imagery_zoom,
        elevation_layer: ELEVATION,
        elevation_zoom: ELEVATION_ZOOM,
        centre: [body[0], body[1]],
        half_side_m: world.spec.radius_m.min(2_400.0),
    };
    let package = world.open();
    let window = pollster::block_on(scene_around(&package, &*world.store, &request, None)).unwrap();
    let pose = true_camera([body[0], body[1]], body, body_to_ned, mount);
    let (view, _) = SparseSceneSampler::new(&window.scene, 2_500.0, 16, 10)
        .unwrap()
        .render_adaptive(camera, &pose, stride, u8::MAX, 0.05)
        .unwrap();
    let image: GrayImage = view.image;
    image
        .as_raw()
        .iter()
        .enumerate()
        .map(|(i, luma)| {
            let noise = 4.0 * (lattice(i as i64, seed as i64, 11) - 0.5);
            (0.9 * f64::from(*luma) + 8.0 + noise)
                .round()
                .clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// The centre of the tile of `zoom` that holds `point`, and the
/// north-south size of that tile in metres.
pub(crate) fn correlation_cell(point: [f64; 2], zoom: u32) -> ([f64; 2], f64) {
    let cell = TileAddress::containing(point, zoom).unwrap();
    let (x, y) = (f64::from(cell.x), f64::from(cell.y));
    let centre = tile_lat_lon(x + 0.5, y + 0.5, zoom);
    let [north, _] = tile_lat_lon(x, y, zoom);
    let [south, _] = tile_lat_lon(x, y + 1.0, zoom);
    (centre, (north - south).to_radians() * 6_364_000.0)
}

/// A check of [`true_camera`] against the frame of the renderer: a camera
/// at the anchor that looks down sees the anchor at its principal point.
#[test]
fn the_true_camera_of_a_level_vehicle_looks_at_the_point_below_it() {
    let level = UnitQuaternion::identity();
    let body = [CENTRE[0], CENTRE[1], 1_000.0];
    let pose = true_camera(CENTRE, body, &level, &CameraMount::nadir());
    assert!((pose.position - Vector3::new(0.0, 0.0, 1_000.0)).norm() < 1e-6);
    let frame = LocalFrame::wgs84_enu(CENTRE[0], CENTRE[1]).unwrap();
    assert!((frame.local(body).unwrap() - pose.position).norm() < 1e-6);
    let pixel = camera_model().project(&pose, Vector3::zeros()).unwrap();
    assert!((pixel.x - 79.5).abs() < 1e-9 && (pixel.y - 59.5).abs() < 1e-9);
    // A point 100 m north of the anchor is up the image: the nose is north.
    let north = camera_model()
        .project(&pose, Vector3::new(0.0, 100.0, 0.0))
        .unwrap();
    assert!(north.y < 59.5 && (north.x - 79.5).abs() < 1e-9);
}
