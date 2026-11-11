//! Errors of the elevation model through the real matcher: a package whose
//! elevation is the true ground plus a tilted plane or a smooth random
//! field. A free 6-DoF solve takes a tilt of that error under the
//! footprint as a turned camera, so the fix moves horizontally by about
//! the height above the ground times the tilt, and the fix must state it
//! ([`crate::reference::terrain::budget::tilt_sigma_m`]).
//!
//! The matches are slow without optimization, so these tests run in an
//! optimized build only.

use std::sync::Arc;

use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use navigate_camera::CameraMount;

use super::fixes::belly;
use super::flight::{CAMERA, Flight, RATE, at, fast, navigator};
use super::world::{
    CENTRE, CORRELATION_ZOOM, DemError, GROUND_M, TODAY, World, WorldSpec, correlation_cell,
};
use crate::reference::terrain::budget::{MapSigmas, ViewGeometry, tilt_sigma_m};
use crate::{Navigator, ReferenceNavigator, TerrainConfig, TerrainMapRequest};

/// Positions 1 km apart: their footprints do not overlap at 400 m above
/// the ground and overlap little at 1.6 km.
const SPACING_M: f64 = 1_000.0;
/// The one-sigma of the elevation fields and of the stated elevation.
const DEM_SIGMA_M: f64 = 5.0;
/// The feature size of the fields and the length that the navigator
/// assumes.
const LENGTH_M: f64 = 300.0;
/// The narrower extent of the test camera for each metre of range.
const FIELD: f64 = 120.0 / 150.0;

/// One fix: the horizontal error, north and east in metres, and the
/// horizontal covariance that the fix states.
pub(super) type Fix = (Vector2<f64>, Matrix2<f64>);

/// The world of `error` with the elevation one-sigma [`DEM_SIGMA_M`]
/// stated, and correlation cells of `zoom`.
pub(super) fn world(error: DemError, zoom: u32) -> World {
    World::build(&WorldSpec {
        accuracy_mm: Some((1_500, (DEM_SIGMA_M * 1_000.0) as u32)),
        dem_error: error,
        correlation_zoom: zoom,
        ..WorldSpec::default()
    })
}

/// Six random elevation fields of [`DEM_SIGMA_M`] and [`LENGTH_M`].
pub(super) fn fields() -> Vec<DemError> {
    (0..6)
        .map(|seed| DemError::Field {
            sigma_m: DEM_SIGMA_M,
            length_m: LENGTH_M,
            seed: 100 + seed,
        })
        .collect()
}

/// Loads `world` with the elevation correlation length `length_m`.
fn load(navigator: &mut ReferenceNavigator, world: &World, length_m: f64) {
    let request = TerrainMapRequest {
        config: TerrainConfig {
            dem_correlation_length_m: length_m,
            ..fast()
        },
        ..TerrainMapRequest::new(world.package, TODAY)
    };
    navigator
        .load_terrain_map(Arc::clone(&world.store), &request, None)
        .unwrap();
}

/// A hover `agl_m` above the mean ground at the position `k` of a 3 by 3
/// grid about the centre.
pub(super) fn hover(k: u32, agl_m: f64) -> Flight {
    let north = f64::from(k % 3) * SPACING_M - SPACING_M;
    let east = f64::from(k / 3) * SPACING_M - SPACING_M;
    Flight {
        start: [
            CENTRE[0] + north / 111_100.0,
            CENTRE[1] + east / 83_000.0,
            GROUND_M + agl_m,
        ],
        speed_m_s: 0.0,
    }
}

/// The mount of a camera `off_deg` from the nadir, looking north or east.
pub(super) fn mount(off_deg: f64, east: bool) -> CameraMount {
    let yaw = if east {
        core::f64::consts::FRAC_PI_2
    } else {
        0.0
    };
    CameraMount::rotated_forward(0.0, -(90.0 - off_deg).to_radians(), yaw)
}

/// The fix of one frame over `map`, rendered over `truth`; `None` when
/// the frame gave no fix.
pub(super) fn fix(maps: (&World, &World), flight: &Flight, mount: CameraMount) -> Option<Fix> {
    fix_rendered(maps, flight, (mount, mount))
}

/// [`fix`] with the frame rendered through the mount `rendered` of a
/// camera declared on `declared`.
fn fix_rendered(
    (map, truth): (&World, &World),
    flight: &Flight,
    (declared, rendered): (CameraMount, CameraMount),
) -> Option<Fix> {
    let mut navigator = navigator(flight, declared, [5.0, -5.0, 2.0], 20.0);
    load(&mut navigator, map, LENGTH_M);
    navigator.set_terrain_match_inspection(true);
    let mut sink = Vec::new();
    for step in 1..=RATE {
        navigator.ingest(&flight.increment(step), &mut sink).ok()?;
    }
    navigator
        .ingest(&flight.frame(truth, &rendered, RATE, 0), &mut sink)
        .ok()?;
    let fix = navigator.last_terrain_match(CAMERA)?.fix()?;
    let [lat, lon, _] = flight.body(RATE);
    let p = fix.position;
    let error = Vector2::new(
        (p.latitude_rad - lat.to_radians()) * 6_364_000.0,
        (p.longitude_rad - lon.to_radians()) * 6_378_137.0 * lat.to_radians().cos(),
    );
    // The map error of the cell is declared apart from the covariance; one
    // fix carries all of it.
    let [nn, ne, _, ee, _, _] = fix.covariance_ned.upper_triangle();
    let [map_n, map_e, _] = fix.map_sigma_m.unwrap_or([0.0; 3]);
    Some((
        error,
        Matrix2::new(nn + map_n * map_n, ne, ne, ee + map_e * map_e),
    ))
}

/// The fixes of the grid over each of `maps`, seen at `off_deg` from
/// `agl_m`.
pub(super) fn grid(maps: &[World], truth: &World, (off_deg, agl_m): (f64, f64)) -> Vec<Fix> {
    let mut fixes = Vec::new();
    for (i, map) in maps.iter().enumerate() {
        for k in 0..9 {
            let east = (k + i as u32) % 2 == 1;
            fixes.extend(fix((map, truth), &hover(k, agl_m), mount(off_deg, east)));
        }
    }
    fixes
}

/// The 99.9 % point of `χ²(k)/n` for `k = dof·n`, by Wilson–Hilferty.
fn upper_band(n: usize, dof: usize) -> f64 {
    let k = (dof * n) as f64;
    let a = 2.0 / (9.0 * k);
    k * (1.0 - a + 3.09 * a.sqrt()).powi(3) / n as f64
}

/// The mean of the squared horizontal error per axis.
fn mean_square(fixes: &[Fix]) -> f64 {
    fixes
        .iter()
        .map(|(e, _)| e.norm_squared() / 2.0)
        .sum::<f64>()
        / fixes.len() as f64
}

/// The mean normalized horizontal error.
fn mean_nees(fixes: &[Fix]) -> f64 {
    fixes
        .iter()
        .filter_map(|(e, c)| Some((e.transpose() * c.try_inverse()? * e)[(0, 0)]))
        .sum::<f64>()
        / fixes.len() as f64
}

/// The tilt term of a view `off_deg` from the nadir, `agl_m` above flat
/// ground.
fn stated_tilt_m(off_deg: f64, agl_m: f64) -> f64 {
    let map = MapSigmas {
        horizontal_m: 0.0,
        height_m: 0.0,
        elevation_m: DEM_SIGMA_M,
        elevation_length_m: LENGTH_M,
    };
    let off = off_deg.to_radians();
    let view = ViewGeometry {
        off_nadir_rad: off,
        ground_direction: [1.0, 0.0],
        range_m: agl_m / off.cos(),
        field_ratio: FIELD,
    };
    tilt_sigma_m(&map, &view)
}

/// A tilted plane of elevation error moves the fix against the slope by
/// about the height above the ground times the slope: 4 m from 400 m and
/// 16 m from 1.6 km for 1 %. The single frames give 0.6 to 1.0 of that,
/// and the shift grows with the height.
#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn a_tilted_elevation_error_moves_the_fix_by_the_height_times_the_slope() {
    let truth = world(DemError::None, CORRELATION_ZOOM);
    let slope = DemError::Slope {
        north: 0.01,
        east: 0.0,
    };
    let plane = world(slope, CORRELATION_ZOOM);
    let mut shifts = Vec::new();
    for agl_m in [400.0, 1_600.0] {
        let flight = hover(4, agl_m);
        let nadir = mount(0.0, false);
        let base = fix((&truth, &truth), &flight, nadir).unwrap().0;
        let tilted = fix((&plane, &truth), &flight, nadir).unwrap().0;
        let shift = tilted - base;
        let expected = 0.01 * agl_m;
        tracing::info!(agl_m, north = shift.x, east = shift.y, "a 1 % plane");
        let along = (-1.25 * expected..=-0.5 * expected).contains(&shift.x);
        assert!(along, "{agl_m}: {shift}");
        assert!(shift.y.abs() < 0.3 * expected + 1.0, "{agl_m}: {shift}");
        shifts.push(-shift.x);
    }
    assert!(shifts[1] > 2.5 * shifts[0], "{shifts:?}");
}

/// What a free 6-DoF solve does not take horizontally: a uniform
/// elevation error moves the fix vertically, and an error of the mount
/// angle turns the solved attitude. Over the grid at 800 m above the
/// ground, a 10 m offset of the elevation and a 5 mrad error of the pitch
/// of a camera 30° off the nadir each move the fix by less than 1.5 m on
/// average, where the relief and the range would give 6 m and 5 m.
#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn a_uniform_elevation_error_and_a_mount_error_hardly_move_the_fix_horizontally() {
    let truth = world(DemError::None, CORRELATION_ZOOM);
    let raised = world(DemError::Offset(10.0), CORRELATION_ZOOM);
    let oblique = mount(30.0, false);
    let pitched = CameraMount::rotated_forward(0.0, -60.0_f64.to_radians() + 0.005, 0.0);
    let (mut offset, mut turned, mut n) = (Vector2::zeros(), Vector2::zeros(), 0.0);
    for k in 0..9 {
        let flight = hover(k, 800.0);
        let base = fix((&truth, &truth), &flight, oblique);
        let high = fix((&raised, &truth), &flight, oblique);
        let off = fix_rendered((&truth, &truth), &flight, (oblique, pitched));
        if let (Some(b), Some(h), Some(o)) = (base, high, off) {
            offset += h.0 - b.0;
            turned += o.0 - b.0;
            n += 1.0;
        }
    }
    let (offset, turned) = (offset / n, turned / n);
    tracing::info!(
        n,
        ?offset,
        ?turned,
        "a uniform elevation error and a mount error"
    );
    assert!(n >= 6.0, "{n}");
    assert!(
        offset.norm() < 1.5 && turned.norm() < 1.5,
        "{offset} {turned}"
    );
}

/// Over random elevation fields of 5 m and 300 m, the normalized error of
/// one fix stays under the band at the nadir, 30° and 45° and from 400 m
/// and 1.6 km. From 400 m the error that a field adds to a fix is of the
/// size of the tilt term and within it, so the term is not vacuous. From
/// 1.6 km the match error of the test camera, pixels of 10 m on the
/// ground, is larger than the term, and the difference of two noisy
/// means does not measure the term.
#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn one_fix_states_the_error_of_a_random_elevation_field() {
    super::flight::show_events();
    let truth = world(DemError::None, CORRELATION_ZOOM);
    let maps: Vec<World> = fields()
        .into_iter()
        .map(|f| world(f, CORRELATION_ZOOM))
        .collect();
    let exact = [world(DemError::None, CORRELATION_ZOOM)];
    for agl_m in [400.0, 1_600.0] {
        for off_deg in [0.0, 30.0, 45.0] {
            let with_field = grid(&maps, &truth, (off_deg, agl_m));
            let without = grid(&exact, &truth, (off_deg, agl_m));
            assert!(
                with_field.len() >= 8 && without.len() >= 4,
                "{agl_m} {off_deg}"
            );
            let added = mean_square(&with_field) - mean_square(&without);
            let tilt = stated_tilt_m(off_deg, agl_m);
            let nees = mean_nees(&with_field);
            let band = upper_band(with_field.len(), 2);
            tracing::info!(
                agl_m,
                off_deg,
                added,
                tilt,
                nees,
                band,
                n = with_field.len(),
                "one fix over a random field"
            );
            assert!(nees <= band, "{agl_m} {off_deg}: {nees} > {band}");
            if agl_m < 1_000.0 {
                let within = (0.25 * tilt * tilt..=1.5 * tilt * tilt).contains(&added);
                assert!(within, "{agl_m} {off_deg}: {added} {tilt}");
            }
        }
    }
}

/// A flight across correlation cells of zoom 16, about 470 m, at 400 m
/// above the ground over a random elevation field of 300 m: one fix in
/// each cell. The footprints of the fixes of adjacent cells, 320 m wide,
/// do not overlap, and their tilt errors are near independent at that
/// spacing; the position NEES of the filter after the flight stays in
/// band over six fields.
#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn fixes_of_adjacent_cells_over_one_elevation_field_keep_the_filter_in_band() {
    super::flight::show_events();
    let zoom = 16;
    let truth = world(DemError::None, zoom);
    let (_, cell_m) = correlation_cell(CENTRE, zoom);
    // A cell each 12 s, a frame each second.
    let flight = super::fixes::north_through_cells(zoom, 24.0, GROUND_M + 400.0, -1.0);
    let steps = (8.0 * cell_m / flight.speed_m_s * RATE as f64) as u64;
    let fields = fields();
    let mut sum = 0.0;
    for field in &fields {
        let map = world(*field, zoom);
        let mut navigator = navigator(&flight, belly(), [-20.0, 30.0, -8.0], 40.0);
        load(&mut navigator, &map, LENGTH_M);
        let mut sink = Vec::new();
        for step in 1..=steps {
            navigator
                .ingest(&flight.increment(step), &mut sink)
                .unwrap();
            if step.is_multiple_of(RATE) {
                let frame = flight.frame(&truth, &belly(), step, 0);
                navigator.ingest(&frame, &mut sink).unwrap();
            }
        }
        let solution = navigator.solution(at(steps)).unwrap();
        let error = flight.error_m(&solution, steps);
        // North, east, up against north, east, down: flip the vertical.
        let e = Vector3::new(error.x, error.y, -error.z);
        let [a, b, c, d, f, g] = solution.position_cov.upper_triangle();
        let p = Matrix3::new(a, b, c, b, d, f, c, f, g);
        let nees = (e.transpose() * p.try_inverse().unwrap() * e)[(0, 0)];
        tracing::info!(
            nees,
            north = error.x,
            east = error.y,
            "a flight across cells"
        );
        sum += nees;
    }
    let mean = sum / fields.len() as f64;
    let band = upper_band(fields.len(), 3);
    tracing::info!(mean, band, "a flight across cells over six fields");
    assert!(mean <= band, "{mean} > {band}");
}
