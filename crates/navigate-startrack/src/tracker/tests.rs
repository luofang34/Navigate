use super::*;
use crate::synthetic::{
    BACKGROUND, RENDER_MAGNITUDE, SIZE, catalogue, pair_table, prior, render, run, sources, truth,
};

/// A star image at the true attitude.
fn stars() -> Vec<u16> {
    render(&sources(&catalogue(RENDER_MAGNITUDE), &truth()), 5.0, 41)
}

/// A frame without stars, such as one behind a cloud or with a lens cap.
fn blank() -> Vec<u16> {
    vec![BACKGROUND as u16; (SIZE * SIZE) as usize]
}

/// Runs one frame through the tracker and records its outcome.
fn frame(tracker: &mut StarTracker, data: &[u16], prior: AttitudePrior) -> bool {
    let solved = run(data, tracker.mode(Some(prior), pair_table())).is_ok();
    tracker.record(solved);
    solved
}

fn good_prior() -> AttitudePrior {
    prior(0.2_f64.to_radians(), 0.5_f64.to_radians())
}

#[test]
fn the_first_frame_and_a_lost_lock_use_lost_in_space() {
    let mut tracker = StarTracker::new(TrackerConfig::default()).expect("tracker");
    assert_eq!(tracker.lock(), TrackerLock::Lost);
    assert!(matches!(
        tracker.mode(Some(good_prior()), pair_table()),
        IdentifyMode::LostInSpace(_)
    ));
    assert!(frame(&mut tracker, &stars(), good_prior()));
    assert_eq!(tracker.lock(), TrackerLock::Locked { failed_frames: 0 });
    assert!(matches!(
        tracker.mode(Some(good_prior()), pair_table()),
        IdentifyMode::Prior(_)
    ));
    for failed_frames in 1..TrackerConfig::default().max_failed_frames {
        assert!(!frame(&mut tracker, &blank(), good_prior()));
        assert_eq!(tracker.lock(), TrackerLock::Locked { failed_frames });
    }
    assert!(!frame(&mut tracker, &blank(), good_prior()));
    assert_eq!(tracker.lock(), TrackerLock::Lost);
    assert!(matches!(
        tracker.mode(Some(good_prior()), pair_table()),
        IdentifyMode::LostInSpace(_)
    ));
}

#[test]
fn a_wrong_prior_is_dropped_after_the_failed_frames() {
    let mut tracker = StarTracker::new(TrackerConfig::default()).expect("tracker");
    assert!(frame(&mut tracker, &stars(), good_prior()));
    // The host's attitude has drifted 20° while it claims 0.5°.
    let wrong = prior(20.0_f64.to_radians(), 0.5_f64.to_radians());
    let data = stars();
    let outcomes: Vec<bool> = (0..4).map(|_| frame(&mut tracker, &data, wrong)).collect();
    // Three frames fail with the wrong prior, then lost-in-space solves.
    assert_eq!(outcomes, [false, false, false, true]);
    assert_eq!(tracker.lock(), TrackerLock::Locked { failed_frames: 0 });
}

#[test]
fn a_wide_or_missing_prior_uses_lost_in_space() {
    let mut tracker = StarTracker::new(TrackerConfig::default()).expect("tracker");
    tracker.record(true);
    let wide = prior(0.2_f64.to_radians(), 5.0_f64.to_radians());
    assert!(matches!(
        tracker.mode(Some(wide), pair_table()),
        IdentifyMode::LostInSpace(_)
    ));
    assert!(matches!(
        tracker.mode(None, pair_table()),
        IdentifyMode::LostInSpace(_)
    ));
    assert!(frame(&mut tracker, &stars(), wide));
}

#[test]
fn an_invalid_prior_limit_is_refused() {
    for max_prior_sigma_rad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let config = TrackerConfig {
            max_prior_sigma_rad,
            ..TrackerConfig::default()
        };
        assert!(matches!(
            StarTracker::new(config),
            Err(StarTrackError::Invalid { .. })
        ));
    }
}
