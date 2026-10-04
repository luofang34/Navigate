//! Synthetic flights over flat ground with real visual geometry checks.

use image::GrayImage;
use nalgebra::{Translation3, UnitQuaternion, Vector2, Vector3};
use navigate_contract::ClockDomainId;
use navigate_visual::{
    CameraModel, Estimate, Frame, FrameStamp, LocalFrame, LocalizerConfig, MapRevision, PixelMatch,
    PosePrior, PoseVerifier, ReferenceView,
};
use navigate_visual_session::{
    Calibration, CameraMount, CaptureTime, ContinuityEpoch, FrameKey, FrameRecord, LensModel,
    MapReliability, MediaTime, Pose, RegionChange, RelativeMotion, ScaleSource, SessionConfig,
    StreamId, SurfaceModel, VerticalReference, VisualSession, from_camera, to_camera,
};
use std::rc::Rc;

pub const FRAME_NS: u64 = 1_000_000_000;

pub fn camera() -> CameraModel {
    CameraModel {
        width: 320,
        height: 240,
        fx: 260.0,
        fy: 260.0,
        cx: 159.5,
        cy: 119.5,
    }
}

pub fn map() -> MapRevision {
    MapRevision {
        release_id: "synthetic".into(),
        manifest_sha256: "b".repeat(64),
    }
}

pub fn local_frame() -> LocalFrame {
    LocalFrame::anchor_mercator(47.0, 11.0).unwrap()
}

pub fn calibration() -> Calibration {
    Calibration {
        intrinsics: camera(),
        lens: LensModel::Undistorted {
            residual_rms_px: 0.3,
        },
        mount: CameraMount::Fixed {
            body_from_camera: Pose::translation(0.0, 0.0, -0.2),
        },
        map_vertical: VerticalReference::Wgs84Ellipsoid,
    }
}

pub fn reliability() -> MapReliability {
    MapReliability {
        horizontal_m: 1.0,
        vertical_m: 2.0,
        imagery_age_years: Some(2.0),
        age_growth_m_per_year: 0.2,
        unknown_age_years: 10.0,
        surface: SurfaceModel::BareEarth {
            max_object_height_m: 2.0,
        },
        change: RegionChange::Unchanged,
    }
}

/// A camera at `position` that looks straight down with heading `yaw`.
pub fn nadir(position: Vector3<f64>, yaw: f64) -> Pose {
    Pose::from_parts(
        Translation3::from(position),
        UnitQuaternion::from_euler_angles(0.0, 0.0, yaw),
    )
}

fn ray(camera: &CameraModel, pose: &Pose, pixel: Vector2<f64>) -> Vector3<f64> {
    pose.rotation
        * Vector3::new(
            (pixel.x - camera.cx) / camera.fx,
            (camera.cy - pixel.y) / camera.fy,
            -1.0,
        )
}

/// Depth of the plane `z = ground` seen from `pose`, rendered as a reference.
pub fn ground_view(pose: &Pose, ground: f64) -> ReferenceView {
    let c = camera();
    let mut depth = Vec::with_capacity((c.width * c.height) as usize);
    for v in 0..c.height {
        for u in 0..c.width {
            let d = ray(&c, pose, Vector2::new(f64::from(u), f64::from(v)));
            let t = (ground - pose.translation.vector.z) / d.z;
            depth.push(if t > 0.0 { t as f32 } else { 0.0 });
        }
    }
    ReferenceView {
        map: map(),
        frame: local_frame(),
        pose: to_camera(pose),
        image: GrayImage::new(c.width, c.height),
        depth_m: depth,
    }
}

/// Matches of ground points between two true camera poses.
pub fn ground_matches(reference: &Pose, query: &Pose, ground: f64) -> Vec<PixelMatch> {
    let c = camera();
    let mut pairs = Vec::new();
    for row in 0..8 {
        for col in 0..10 {
            let pixel = Vector2::new(20.0 + 31.0 * f64::from(col), 15.0 + 30.0 * f64::from(row));
            let d = ray(&c, reference, pixel);
            let world = reference.translation.vector
                + d * ((ground - reference.translation.vector.z) / d.z);
            if let Some(q) = c.project(&to_camera(query), world)
                && q.x > 2.0
                && q.y > 2.0
                && q.x < 317.0
                && q.y < 237.0
            {
                pairs.push(PixelMatch {
                    reference: pixel,
                    query: q,
                });
            }
        }
    }
    pairs
}

pub fn image_frame(sequence: u64) -> Frame {
    let c = camera();
    Frame {
        stamp: FrameStamp {
            sequence,
            capture_time_ns: sequence * FRAME_NS + 1,
        },
        camera: c,
        image: GrayImage::new(c.width, c.height),
    }
}

pub fn key(epoch: u32, index: u64) -> FrameKey {
    FrameKey {
        stream: StreamId(1),
        continuity: ContinuityEpoch(epoch),
        index,
    }
}

pub fn record(key: FrameKey, frame: &Frame) -> FrameRecord {
    let capture = CaptureTime {
        clock: ClockDomainId::new(7),
        at_ns: frame.stamp.capture_time_ns,
        error_bound_ns: 1_000_000,
    };
    let media = MediaTime {
        pts: i64::try_from(frame.stamp.sequence).unwrap() * 3_000,
        timebase_num: 1,
        timebase_den: 30_000,
    };
    FrameRecord::from_frame(key, Some(media), capture, frame)
}

/// A map match of `truth` against a map whose ground is moved by `map_shift`.
pub fn map_estimate(frame: &Frame, truth: &Pose, map_shift: Vector3<f64>) -> Estimate {
    let shifted = Pose::from_parts(
        Translation3::from(truth.translation.vector + map_shift),
        truth.rotation,
    );
    let render_at = Pose::from_parts(
        Translation3::from(shifted.translation.vector + Vector3::new(4.0, -3.0, 2.0)),
        shifted.rotation,
    );
    let reference = ground_view(&render_at, map_shift.z);
    // Map points carry the shift, so the solved camera carries it as well.
    let pairs = ground_matches(&render_at, &shifted, map_shift.z);
    let prior = PosePrior {
        pose: to_camera(&render_at),
        position_radius_m: 50.0,
        attitude_radius_rad: 0.5,
    };
    PoseVerifier::new(LocalizerConfig::default())
        .unwrap()
        .verify(frame, &reference, &prior, &pairs, "synthetic-ground")
        .unwrap()
}

/// A flight that feeds frames, motion with declared drift, and optional anchors.
pub struct Flight {
    pub session: VisualSession,
    pub truth: Vec<(FrameKey, Pose, Rc<Frame>)>,
    /// Measured odometry pose of the newest frame, with injected drift.
    measured: Option<Pose>,
    pub heading_bias_rad: f64,
    pub scale_error: f64,
    epoch: u32,
    next_index: u64,
    sequence: u64,
}

impl Flight {
    pub fn new(config: SessionConfig) -> Self {
        Self {
            session: VisualSession::new(config, calibration()).unwrap(),
            truth: Vec::new(),
            measured: None,
            heading_bias_rad: 0.0,
            scale_error: 0.0,
            epoch: 0,
            next_index: 0,
            sequence: 0,
        }
    }

    pub fn standard() -> Self {
        Self::new(SessionConfig::standard())
    }

    /// Start a new continuity epoch, as after a seek.
    pub fn seek(&mut self) {
        self.epoch += 1;
        self.next_index = 0;
        self.measured = None;
    }

    /// Observe one frame at `pose`. Motion from the previous frame of the epoch
    /// carries the injected scale error and heading bias.
    pub fn fly_to(&mut self, pose: Pose) -> FrameKey {
        let frame = image_frame(self.sequence);
        self.sequence += 1;
        let k = key(self.epoch, self.next_index);
        self.next_index += 1;
        self.session.observe_frame(record(k, &frame)).unwrap();
        let previous = self
            .truth
            .iter()
            .rev()
            .find(|(pk, _, _)| pk.track() == k.track())
            .cloned();
        match (previous, self.measured) {
            (Some((pk, prev_truth, prev_frame)), Some(measured)) => {
                let mut step = prev_truth.inv_mul(&pose);
                step.translation.vector *= 1.0 + self.scale_error;
                step.rotation = UnitQuaternion::from_euler_angles(0.0, 0.0, self.heading_bias_rad)
                    * step.rotation;
                let motion = RelativeMotion {
                    from: pk,
                    to: k,
                    from_to: step,
                    scale: ScaleSource::MapDepth { map: map() },
                    from_observation_sha256: prev_frame.evidence_sha256(),
                    to_observation_sha256: frame.evidence_sha256(),
                };
                self.session.apply_motion(motion).unwrap();
                self.measured = Some(measured * step);
            }
            _ => self.measured = Some(Pose::identity()),
        }
        self.truth.push((k, pose, Rc::new(frame)));
        k
    }

    pub fn last(&self) -> &(FrameKey, Pose, Rc<Frame>) {
        self.truth.last().unwrap()
    }

    pub fn find(&self, k: &FrameKey) -> &(FrameKey, Pose, Rc<Frame>) {
        self.truth.iter().find(|(x, _, _)| x == k).unwrap()
    }
}

/// Horizontal distance between the map pose of a frame and its true pose.
pub fn error_m(session: &VisualSession, k: &FrameKey, truth: &Pose) -> Option<f64> {
    match session.pose_at(k).unwrap().map {
        navigate_visual_session::MapPose::Located { pose, .. } => {
            Some((pose.translation.vector.xy() - truth.translation.vector.xy()).norm())
        }
        navigate_visual_session::MapPose::Unlocated(_) => None,
    }
}

pub fn anchor(
    flight: &Flight,
    k: &FrameKey,
    map_shift: Vector3<f64>,
) -> navigate_visual_session::AnchorObservation {
    let (_, truth, frame) = flight.find(k);
    let estimate = map_estimate(frame, truth, map_shift);
    navigate_visual_session::AnchorObservation::from_estimate(*k, &estimate, reliability())
}

pub fn odometry_of(session: &VisualSession, k: &FrameKey) -> Pose {
    session.pose_at(k).unwrap().odometry.unwrap().pose
}

pub fn pose_of(camera: &navigate_visual::CameraPose) -> Pose {
    from_camera(camera)
}
