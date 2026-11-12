//! Frames with no content are found before any work is done for them, and
//! frames with a little content are still matched.

use image::GrayImage;
use navigate_camera::CameraMount;

use super::fixes::{HEIGHT_M, belly, terrain};
use super::flight::{CAMERA, Flight, RATE, at, load, navigator, show_events};
use super::world::{CENTRE, World, WorldSpec, lattice};
use crate::reference::terrain::content::Content;
use crate::{Navigator, Outcome, SensorData, SensorInput, TerrainConfig};

/// Noise with a one-sigma of one count for the pixel `index` of the frame
/// `seed`: the sum of twelve uniform numbers.
#[cfg(test)]
fn noise(index: usize, seed: u64) -> f64 {
    (0..12).map(|k| lattice(index as i64, k, seed)).sum::<f64>() - 6.0
}

/// A frame of a dark scene: an offset of ten counts and noise of `sigma`.
#[cfg(test)]
pub(super) fn dark(sigma: f64, seed: u64) -> Vec<u8> {
    (0..160 * 120)
        .map(|i| (10.0 + sigma * noise(i, seed)).round().clamp(0.0, 255.0) as u8)
        .collect()
}

/// The frame `luma` with its contrast about the mid grey scaled by
/// `contrast`, and new noise of two counts.
#[cfg(test)]
fn faded(luma: &[u8], contrast: f64, seed: u64) -> Vec<u8> {
    luma.iter()
        .enumerate()
        .map(|(i, v)| {
            let value = 128.0 + contrast * (f64::from(*v) - 128.0) + 2.0 * noise(i, seed);
            value.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

#[cfg(test)]
fn structure(luma: &[u8]) -> f64 {
    Content::of(&GrayImage::from_raw(160, 120, luma.to_vec()).unwrap()).structure()
}

/// The basis of the default limit: the structure of noise alone is one at
/// each noise level, and the structure of the poorest frame that still
/// gives a fix is several times the limit.
#[test]
fn noise_has_the_structure_of_one_and_ground_has_much_more() {
    show_events();
    let limit = TerrainConfig::default().min_structure;
    for sigma in [1.0, 3.0, 10.0] {
        let value = structure(&dark(sigma, 5));
        tracing::info!(sigma, structure = value, "a frame of noise");
        assert!((0.8..1.3).contains(&value), "{sigma}: {value}");
        assert!(
            Content::of(&GrayImage::from_raw(160, 120, dark(sigma, 5)).unwrap())
                .is_featureless(limit)
        );
    }
    // A frame with no noise at all is flat.
    let flat = GrayImage::from_pixel(160, 120, image::Luma([10]));
    assert!(Content::of(&flat).is_featureless(limit));

    let world = World::build(&WorldSpec::default());
    let hover = Flight {
        start: [CENTRE[0], CENTRE[1], HEIGHT_M],
        speed_m_s: 0.0,
    };
    let frame = hover.frame(&world, &belly(), RATE, 0);
    let SensorData::TerrainFrame { luma, .. } = &frame.data else {
        panic!("a terrain frame");
    };
    // The outcome of the frame at each contrast, with a navigator that is
    // new for each.
    let mut poorest_fused = f64::INFINITY;
    for contrast in [1.0, 0.5, 0.25, 0.12, 0.06] {
        let pixels = faded(luma, contrast, 9);
        let value = structure(&pixels);
        let mut navigator = navigator(&hover, belly(), [15.0, -10.0, 5.0], 20.0);
        load(&mut navigator, &world);
        let mut sink = Vec::new();
        for step in 1..=RATE {
            navigator.ingest(&hover.increment(step), &mut sink).unwrap();
        }
        let input = SensorInput {
            data: SensorData::TerrainFrame {
                width: 160,
                height: 120,
                luma: pixels,
            },
            ..frame.clone()
        };
        navigator.ingest(&input, &mut sink).unwrap();
        let record = terrain(&sink).remove(0);
        tracing::info!(contrast, structure = value, outcome = ?record, "a frame of ground");
        if record.0 == Outcome::Accepted {
            poorest_fused = poorest_fused.min(value);
        }
        // No frame of ground at these contrasts is taken for noise.
        assert_ne!(record.0, Outcome::Unused("featureless_frame"), "{contrast}");
    }
    assert!(poorest_fused > 2.0 * limit, "{poorest_fused}");
}

#[test]
fn a_frame_of_noise_is_recorded_and_costs_no_match() {
    let world = World::build(&WorldSpec::default());
    let hover = Flight {
        start: [CENTRE[0], CENTRE[1], HEIGHT_M],
        speed_m_s: 0.0,
    };
    let mut navigator = navigator(&hover, belly(), [15.0, -10.0, 5.0], 20.0);
    load(&mut navigator, &world);
    let mut sink = Vec::new();
    for step in 1..=RATE {
        navigator.ingest(&hover.increment(step), &mut sink).unwrap();
    }
    let before = navigator.solution(at(RATE)).unwrap();
    let night = SensorInput {
        sensor: CAMERA,
        acquired_at: at(RATE),
        delivered_at: at(RATE),
        data: SensorData::TerrainFrame {
            width: 160,
            height: 120,
            luma: dark(3.0, 1),
        },
    };
    navigator.ingest(&night, &mut sink).unwrap();
    // No reference view was made, and the camera took no working camera.
    assert_eq!(
        terrain(&sink),
        [(Outcome::Unused("featureless_frame"), None)]
    );
    assert_eq!(navigator.terrain_working_camera(CAMERA), None);
    assert_eq!(navigator.solution(at(RATE)).unwrap(), before);
}

/// The block contrast of a 160 by 120 frame.
#[cfg(test)]
fn block(luma: &[u8]) -> f64 {
    Content::of(&GrayImage::from_raw(160, 120, luma.to_vec()).unwrap()).block
}

/// A camera that looks 45° ahead of the nadir.
#[cfg(test)]
fn oblique() -> CameraMount {
    CameraMount::rotated_forward(0.0, -45.0_f64.to_radians(), 0.0)
}

/// The outcome of the frame `luma` of `world` from `hover`, for a camera
/// on `mount`.
#[cfg(test)]
fn outcome(
    world: &World,
    hover: &Flight,
    mount: CameraMount,
    frame: &SensorInput,
    luma: Vec<u8>,
) -> Outcome {
    let mut navigator = navigator(hover, mount, [15.0, -10.0, 5.0], 20.0);
    load(&mut navigator, world);
    let mut sink = Vec::new();
    for step in 1..=RATE {
        navigator.ingest(&hover.increment(step), &mut sink).unwrap();
    }
    let input = SensorInput {
        data: SensorData::TerrainFrame {
            width: 160,
            height: 120,
            luma,
        },
        ..frame.clone()
    };
    navigator.ingest(&input, &mut sink).unwrap();
    terrain(&sink).remove(0).0
}

/// The frame of `luma` faded to `target` counts of block contrast, by
/// bisection: the block contrast grows with the contrast.
#[cfg(test)]
fn faded_to(luma: &[u8], target: f64) -> Vec<u8> {
    let (mut low, mut high) = (0.0, 1.0);
    for _ in 0..30 {
        let middle = 0.5 * (low + high);
        if block(&faded(luma, middle, 9)) < target {
            low = middle;
        } else {
            high = middle;
        }
    }
    faded(luma, high, 9)
}

/// The frame of a camera on `mount` over the centre of `world`.
#[cfg(test)]
fn frame_of(world: &World, mount: &CameraMount) -> (Flight, SensorInput, Vec<u8>) {
    let hover = Flight {
        start: [CENTRE[0], CENTRE[1], HEIGHT_M],
        speed_m_s: 0.0,
    };
    let frame = hover.frame(world, mount, RATE, 0);
    let SensorData::TerrainFrame { luma, .. } = &frame.data else {
        panic!("a terrain frame");
    };
    let luma = luma.clone();
    (hover, frame, luma)
}

/// A frame of a camera 45° off the nadir, faded to just under the block
/// contrast floor, is refused as faint without a match; faded to just
/// over it, it is matched.
#[test]
fn the_faint_floor_of_an_oblique_camera_refuses_just_under_it_and_matches_just_over_it() {
    show_events();
    let floor = TerrainConfig::default().min_block_contrast;
    let world = World::build(&WorldSpec::default());
    let (hover, frame, luma) = frame_of(&world, &oblique());
    let under = faded_to(&luma, 0.95 * floor);
    let over = faded_to(&luma, 1.05 * floor);
    tracing::info!(
        under = block(&under),
        over = block(&over),
        "the faded frames"
    );
    assert!(block(&under) < floor && block(&over) > floor);
    assert_eq!(
        outcome(&world, &hover, oblique(), &frame, under),
        Outcome::Unused("faint_frame")
    );
    assert_ne!(
        outcome(&world, &hover, oblique(), &frame, over),
        Outcome::Unused("faint_frame")
    );
}

/// A camera at the nadir has no floor: a frame under the floor of an
/// oblique camera goes to the matcher, which decides. The camera of this
/// test has 160 by 120 pixels, and the matcher finds too little in its
/// faint frames; the camera of `tests::geometry` fixes at these
/// contrasts.
#[test]
fn a_camera_at_the_nadir_gives_a_frame_under_the_oblique_floor_to_the_matcher() {
    show_events();
    let floor = TerrainConfig::default().min_block_contrast;
    let world = World::build(&WorldSpec::default());
    let (hover, frame, luma) = frame_of(&world, &belly());
    for part in [0.75, 0.5] {
        let faint = faded_to(&luma, part * floor);
        assert!(block(&faint) < (part + 0.1) * floor);
        let outcome = outcome(&world, &hover, belly(), &frame, faint);
        tracing::info!(part, ?outcome, "a faint frame at the nadir");
        assert_ne!(outcome, Outcome::Unused("faint_frame"));
        assert_ne!(outcome, Outcome::Unused("featureless_frame"));
    }
}

#[test]
fn the_floor_rises_from_thirty_degrees_to_forty_five() {
    let config = TerrainConfig::default();
    let at = |degrees: f64| config.faint_floor(degrees.to_radians());
    assert_eq!([at(0.0), at(15.0), at(30.0)], [0.0, 0.0, 0.0]);
    assert!((at(37.5) - 2.0).abs() < 1e-9);
    assert!((at(45.0) - 4.0).abs() < 1e-9 && (at(70.0) - 4.0).abs() < 1e-9);
    // The angles must be in order.
    let reversed = TerrainConfig {
        faint_floor_from_rad: 1.0,
        faint_floor_full_rad: 0.5,
        ..config
    };
    assert!(reversed.validate().is_err() && config.validate().is_ok());
}
