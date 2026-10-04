//! Long-run memory bounds and event-queue overflow.

use crate::support::*;
use nalgebra::Vector3;
use navigate_visual_session::{
    AnchorDecision, MapPose, SessionConfig, SessionError, SessionLimits, SessionUsage,
};
use std::io::Write;
use std::time::{Duration, Instant};

/// Peak use, queue overflow, and session processing time over one run.
struct Run {
    peak: SessionUsage,
    dropped: u64,
    full_revision: bool,
    motion: Duration,
    anchors: Duration,
    worst_anchor: Duration,
}

impl Run {
    /// One frame of a lawnmower pattern: 40 frames per line, 30 m between
    /// lines, with a seek every 300 frames.
    fn step(&mut self, flight: &mut Flight, i: u64) {
        if i > 0 && i.is_multiple_of(300) {
            flight.seek();
        }
        let line = i / 40;
        let along = (i % 40) as f64 * 8.0;
        let x = if line.is_multiple_of(2) {
            along
        } else {
            312.0 - along
        };
        let started = Instant::now();
        let k = flight.fly_to(nadir(Vector3::new(x, 30.0 * line as f64, 100.0), 0.0));
        self.motion += started.elapsed();
        if i.is_multiple_of(25) {
            let observation = anchor(flight, &k, Vector3::zeros());
            let started = Instant::now();
            let decision = flight.session.submit_anchor(observation).unwrap();
            let elapsed = started.elapsed();
            self.anchors += elapsed;
            self.worst_anchor = self.worst_anchor.max(elapsed);
            assert!(
                !matches!(decision, AnchorDecision::Rejected(_)),
                "frame {i}: {decision:?}"
            );
        }
        // The session no longer keeps old frames, so the test drops their pixels too.
        if flight.truth.len() > 400 {
            flight.truth.drain(..100);
        }
        self.peak = highest(self.peak, flight.session.usage());
        if i % 500 == 499 {
            let batch = flight.session.drain_events();
            self.dropped += batch.dropped;
            self.full_revision |= batch.full_revision_required;
        }
    }
}

#[test]
fn a_long_run_keeps_every_store_within_its_limit() {
    let limits = SessionLimits {
        max_frames: 300,
        max_keyframes: 24,
        max_closures: 8,
        max_bias_cells: 4,
        max_events: 32,
        max_descriptor_len: 64,
        max_ledger_cells: 8,
    };
    let mut flight = Flight::new(SessionConfig {
        limits,
        ..SessionConfig::standard()
    });
    flight.heading_bias_rad = 0.001;
    let mut run = Run {
        peak: flight.session.usage(),
        dropped: 0,
        full_revision: false,
        motion: Duration::ZERO,
        anchors: Duration::ZERO,
        worst_anchor: Duration::ZERO,
    };
    for i in 0..2_000_u64 {
        run.step(&mut flight, i);
    }
    let (peak, mut out) = (run.peak, std::io::stderr());
    writeln!(out, "processing time: frame and motion {:?} per frame, including synthetic frame hashing; anchor {:?} mean, {:?} worst", run.motion / 2_000, run.anchors / 80, run.worst_anchor).ok();
    assert!(
        peak.frames <= 300 && peak.keyframes <= 24 && peak.anchors <= 24,
        "{peak:?}"
    );
    assert!(
        peak.events <= 32 && peak.bias_cells <= 4 && peak.ledger_cells <= 8,
        "{peak:?}"
    );
    assert!(
        peak.segments <= 3,
        "seeks do not accumulate segments: {peak:?}"
    );
    assert!(
        run.dropped > 0 && run.full_revision,
        "an overflowing queue asks for a full re-projection"
    );
    let old = key(1, 10);
    assert!(
        matches!(
            flight.session.pose_at(&old),
            Err(SessionError::UnknownFrame(_))
        ),
        "a removed frame is reported, not invented"
    );
    let (newest, truth, _) = flight.last().clone();
    let MapPose::Located { pose, .. } = flight.session.pose_at(&newest).unwrap().map else {
        panic!("the newest frame stays located after eviction");
    };
    let error = (pose.translation.vector.xy() - truth.translation.vector.xy()).norm();
    assert!(
        error < 6.0,
        "evicted keyframes keep their map reference: {error} m"
    );
}

fn highest(a: SessionUsage, b: SessionUsage) -> SessionUsage {
    SessionUsage {
        frames: a.frames.max(b.frames),
        keyframes: a.keyframes.max(b.keyframes),
        anchors: a.anchors.max(b.anchors),
        pending_anchors: a.pending_anchors.max(b.pending_anchors),
        closures: a.closures.max(b.closures),
        bias_cells: a.bias_cells.max(b.bias_cells),
        segments: a.segments.max(b.segments),
        events: a.events.max(b.events),
        ledger_cells: a.ledger_cells.max(b.ledger_cells),
    }
}
