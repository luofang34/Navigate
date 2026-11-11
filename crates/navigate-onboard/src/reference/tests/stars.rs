//! Star frames: identified and offered once the navigator has a state,
//! recorded and not identified before it, and taken without the prior or
//! the lock of a state that the navigator forgot.

use super::*;
use navigate_startrack::{StarCentroid, catalogue_ecef};

/// The catalogue stars that a zenith camera on a level vehicle at the
/// alignment sees at `acquired_at`, as ideal centroids.
#[cfg(test)]
fn zenith_centroids(acquired_at: MonotonicNanos) -> Vec<StarCentroid> {
    let config = StarCatalogueConfig::with_utc_at_zero(UTC_AT_ZERO);
    let catalogue = catalogue_ecef(config.epoch(acquired_at), 5.5).unwrap();
    let ecef_from_body = UnitQuaternion::from_matrix(
        &navigate_fusion::inertial::ned_to_body_fixed_dcm(LAT.to_radians(), LON.to_radians()),
    );
    let cam_from_ecef = CameraMount::zenith().cam_from_body * ecef_from_body.inverse();
    let k = star_imager().intrinsics;
    catalogue
        .iter()
        .filter_map(|s| k.project(&(cam_from_ecef * Vector3::from(s.direction))))
        .filter(|p| (5.0..506.0).contains(&p.x) && (5.0..506.0).contains(&p.y))
        .map(|pixel: Vector2<f64>| StarCentroid {
            pixel,
            flux: 50_000.0,
            pixel_count: 9,
            peak: 5_000.0,
            saturated: false,
            length_px: 0.0,
            covariance_px2: Matrix2::identity() * 0.01,
        })
        .collect()
}

#[test]
fn a_star_frame_is_identified_and_offered_as_lines_of_sight() {
    let mut navigator = aligned(&[SensorKind::StarCamera]);
    let mut sink = Vec::new();
    run(&mut navigator, 1, None, &mut sink);
    let mid = at(RATE);
    let centroids = zenith_centroids(mid);
    assert!(centroids.len() >= 6, "{}", centroids.len());
    let exposure = SensorInput {
        sensor: STARS,
        acquired_at: mid,
        delivered_at: mid,
        data: SensorData::StarExposureMid,
    };
    navigator.ingest(&exposure, &mut sink).unwrap();
    let frame = SensorInput {
        sensor: STARS,
        acquired_at: mid,
        delivered_at: mid,
        data: SensorData::StarCentroids { centroids },
    };
    navigator.ingest(&frame, &mut sink).unwrap();
    let accepted = sink
        .iter()
        .filter(|r| r.measurement == "star_line_of_sight" && r.outcome == Outcome::Accepted)
        .count();
    assert!(accepted >= 4, "{sink:?}");
    let empty = SensorInput {
        sensor: STARS,
        acquired_at: mid,
        delivered_at: mid,
        data: SensorData::StarCentroids { centroids: vec![] },
    };
    sink.clear();
    navigator.ingest(&empty, &mut sink).unwrap();
    assert!(
        matches!(
            sink.as_slice(),
            [AcceptanceRecord {
                outcome: Outcome::Unused(_),
                ..
            }]
        ),
        "{sink:?}"
    );
}

#[test]
fn every_line_of_sight_of_a_second_frame_is_accepted() {
    let mut navigator = aligned(&[SensorKind::StarCamera]);
    let mut sink = Vec::new();
    run(&mut navigator, 1, None, &mut sink);
    let mut per_frame = Vec::new();
    for second in [1_u64, 2] {
        if second == 2 {
            for step in RATE + 1..=2 * RATE {
                navigator.ingest(&rest_increment(step), &mut sink).unwrap();
            }
        }
        let mid = at(second * RATE);
        let exposure = SensorInput {
            sensor: STARS,
            acquired_at: mid,
            delivered_at: mid,
            data: SensorData::StarExposureMid,
        };
        navigator.ingest(&exposure, &mut sink).unwrap();
        let centroids = zenith_centroids(mid);
        let frame = SensorInput {
            sensor: STARS,
            acquired_at: mid,
            delivered_at: mid,
            data: SensorData::StarCentroids { centroids },
        };
        sink.clear();
        navigator.ingest(&frame, &mut sink).unwrap();
        per_frame.push(sink.clone());
    }
    let [first, second] = [&per_frame[0], &per_frame[1]];
    assert!(first.len() >= 4, "{first:?}");
    assert!(
        first.iter().all(|r| r.context == Some("lost_in_space")),
        "{first:?}"
    );
    assert!(second.len() >= 4, "{second:?}");
    let accepted = second
        .iter()
        .filter(|r| r.outcome == Outcome::Accepted)
        .count();
    assert_eq!(accepted, second.len(), "{second:?}");
    assert!(
        second.iter().all(|r| r.context == Some("tracking")),
        "{second:?}"
    );
}

/// The exposure mark and the frame of the star camera at `mid`.
#[cfg(test)]
fn exposure_and_frame(mid: MonotonicNanos) -> [SensorInput; 2] {
    let input = |data| SensorInput {
        sensor: STARS,
        acquired_at: mid,
        delivered_at: mid,
        data,
    };
    [
        input(SensorData::StarExposureMid),
        input(SensorData::StarCentroids {
            centroids: zenith_centroids(mid),
        }),
    ]
}

/// The records of the frame at the end of second `second`.
#[cfg(test)]
fn frame_records(navigator: &mut ReferenceNavigator, second: u64) -> Vec<AcceptanceRecord> {
    let mut sink = Vec::new();
    for input in exposure_and_frame(at(second * RATE)) {
        navigator.ingest(&input, &mut sink).unwrap();
    }
    sink
}

/// The state of `aligned` at `step`: level, at the place, at rest.
#[cfg(test)]
fn transfer(step: u64) -> Alignment {
    Alignment {
        position: GeodeticPosition::new(LAT.to_radians(), LON.to_radians(), HEIGHT),
        velocity: NedVelocity::new(0.0, 0.0, 0.0),
        attitude: AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
        position_cov: SymmetricCov3::from_diagonal(4.0, 4.0, 9.0),
        velocity_cov: SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
        attitude_cov: SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-5),
        at: at(step),
    }
}

/// Before the state a star frame is recorded as taken by the alignment:
/// it is not identified, so it holds no lock for the frames after the
/// state.
#[test]
fn a_star_frame_before_the_state_is_recorded_and_not_identified() {
    let stars = StarCatalogueConfig::with_utc_at_zero(UTC_AT_ZERO);
    let mut navigator =
        ReferenceNavigator::new(suite(&[SensorKind::StarCamera]), CLOCK, Some(stars)).unwrap();
    let records = frame_records(&mut navigator, 1);
    assert!(
        matches!(
            records.as_slice(),
            [AcceptanceRecord {
                measurement: "star_line_of_sight",
                outcome: Outcome::Unused("alignment"),
                context: None,
                ..
            }]
        ),
        "{records:?}"
    );
    navigator.align(&transfer(RATE)).unwrap();
    let records = frame_records(&mut navigator, 1);
    assert!(records.len() >= 4, "{records:?}");
    assert!(records.iter().all(|r| r.context == Some("lost_in_space")));
}

/// A new state forgets the prior that the star camera held from the last
/// state and the lock that its priors kept: the next frame is identified
/// without a prior.
#[test]
fn a_new_state_forgets_the_star_prior_and_lock() {
    let mut navigator = aligned(&[SensorKind::StarCamera]);
    let mut sink = Vec::new();
    run(&mut navigator, 1, None, &mut sink);
    frame_records(&mut navigator, 1);
    for step in RATE + 1..=2 * RATE {
        navigator.ingest(&rest_increment(step), &mut sink).unwrap();
    }
    assert!(
        frame_records(&mut navigator, 2)
            .iter()
            .all(|r| r.context == Some("tracking"))
    );
    for step in 2 * RATE + 1..=3 * RATE {
        navigator.ingest(&rest_increment(step), &mut sink).unwrap();
    }
    let [exposure, frame] = exposure_and_frame(at(3 * RATE));
    navigator.ingest(&exposure, &mut sink).unwrap();
    navigator.align(&transfer(3 * RATE)).unwrap();
    sink.clear();
    navigator.ingest(&frame, &mut sink).unwrap();
    assert!(sink.len() >= 4, "{sink:?}");
    assert!(
        sink.iter().all(|r| r.context == Some("lost_in_space")),
        "{sink:?}"
    );
    // A realignment forgets them too: before the next state the frame is
    // the alignment's, and after it the frame is identified anew.
    navigator.realign().unwrap();
    let records = frame_records(&mut navigator, 3);
    assert!(
        records
            .iter()
            .all(|r| r.outcome == Outcome::Unused("alignment"))
    );
    navigator.align(&transfer(3 * RATE)).unwrap();
    let records = frame_records(&mut navigator, 3);
    assert!(records.iter().all(|r| r.context == Some("lost_in_space")));
}
