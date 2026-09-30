use super::*;
use nalgebra::{Matrix3, Vector3};
use navigate_contract::{ClockDomainId, MonotonicNanos};

const HOST: ClockDomainId = ClockDomainId::new(1);
const MCU: ClockDomainId = ClockDomainId::new(2);
fn time(ms: u64, error_ms: u64) -> TimedCapture {
    TimedCapture {
        at: MonotonicNanos::from_nanos(ms * 1_000_000),
        clock: HOST,
        error_bound: DurationNanos::from_millis(error_ms),
    }
}
fn sample(ms: u64, yaw: f64) -> AttitudeSample {
    let q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw);
    let q = q.quaternion();
    AttitudeSample {
        time: time(ms, 1),
        body_to_ned: AttitudeQuaternion::new(q.w, q.i, q.j, q.k),
        angular_rms_rad: Some(0.02),
    }
}
fn alignment() -> CameraAlignment {
    CameraAlignment {
        eye_to_body: UnitQuaternion::from_matrix(&Matrix3::new(
            0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
        )),
        mounting_rms_rad: Some(0.01),
        interpolation_rms_rad: Some(0.005),
        maximum_gap: DurationNanos::from_millis(100),
        angular_speed_bound_rad_s: 10.0,
    }
}

#[test]
fn capture_uses_measurement_time_and_mounting_for_nadir_and_forward_cameras() {
    let before = sample(1000, 0.0);
    let after = sample(1040, 0.0);
    let down = camera_capture(time(1020, 1), before, after, alignment()).expect("capture");
    assert!(down.orientation.angle() < 1e-12);
    assert_eq!(down.at, time(1020, 0).at);
    let mut forward = alignment();
    forward.eye_to_body =
        UnitQuaternion::from_matrix(&Matrix3::new(0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0));
    let forward = camera_capture(time(1020, 1), before, after, forward).expect("forward capture");
    let direction = forward.orientation * -Vector3::z();
    assert!((direction - Vector3::y()).norm() < 1e-12);
}

#[test]
fn sparse_visual_updates_use_fresh_attitude_brackets_and_keep_error_allowances() {
    for ms in [1000, 5000, 11000] {
        let capture = camera_capture(
            time(ms, 2),
            sample(ms - 20, 0.0),
            sample(ms + 20, 0.2),
            alignment(),
        )
        .expect("capture");
        let expected = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), -0.1);
        assert!(capture.orientation.angle_to(&expected) < 1e-12);
        assert!((capture.attitude_sigma_rad - (0.02 + 0.01 + 0.005 + 10.0 * 0.003)).abs() < 1e-12);
    }
}

#[test]
fn quaternion_sign_change_does_not_create_a_rotation() {
    let before = sample(1000, 0.3);
    let mut after = sample(1040, 0.3);
    let q = after.body_to_ned;
    after.body_to_ned = AttitudeQuaternion::new(-q.w, -q.x, -q.y, -q.z);
    let capture = camera_capture(time(1020, 1), before, after, alignment()).expect("capture");
    assert!(
        capture
            .orientation
            .angle_to(&UnitQuaternion::from_axis_angle(&Vector3::z_axis(), -0.3))
            < 1e-12
    );
}

#[test]
fn arrival_gaps_time_error_and_foreign_clocks_never_trigger_extrapolation() {
    let before = sample(1000, 0.0);
    let after = sample(1040, 0.0);
    for t in [time(990, 0), time(1050, 0), time(1020, 21)] {
        assert!(matches!(
            camera_capture(t, before, after, alignment()),
            Err(CaptureError::NotBracketed { .. })
        ));
    }
    assert!(camera_capture(time(1100, 0), before, sample(1200, 0.0), alignment()).is_err());
    let mut rebooted = after;
    rebooted.time.clock = MCU;
    assert!(matches!(
        camera_capture(time(1020, 0), before, rebooted, alignment()),
        Err(CaptureError::ClockDomain)
    ));
    assert!(camera_capture(time(1020, 0), after, before, alignment()).is_err());
}

#[test]
fn absent_uncertainty_and_invalid_quaternions_are_not_exact_attitudes() {
    let before = sample(1000, 0.0);
    let after = sample(1040, 0.0);
    let mut unknown = before;
    unknown.angular_rms_rad = None;
    assert!(matches!(
        camera_capture(time(1020, 0), unknown, after, alignment()),
        Err(CaptureError::UnknownError { .. })
    ));
    let mut mount = alignment();
    mount.mounting_rms_rad = None;
    assert!(matches!(
        camera_capture(time(1020, 0), before, after, mount),
        Err(CaptureError::UnknownError { .. })
    ));
    let mut bad = before;
    for q in [
        AttitudeQuaternion::new(0.0, 0.0, 0.0, 0.0),
        AttitudeQuaternion::new(f64::NAN, 0.0, 0.0, 1.0),
    ] {
        bad.body_to_ned = q;
        assert!(camera_capture(time(1020, 0), bad, after, alignment()).is_err());
    }
    let mut speed = alignment();
    speed.angular_speed_bound_rad_s = 0.01;
    assert!(camera_capture(time(1020, 0), before, sample(1040, 0.3), speed).is_err());
}

#[test]
fn clock_mapping_handles_large_extended_boot_times_and_rejects_new_boots() {
    let source = u64::from(u32::MAX) * 1_000_000 + 5_000_000;
    let mapping = ClockAlignment {
        source_clock: MCU,
        source_anchor: MonotonicNanos::from_nanos(source),
        target_anchor: time(1000, 2),
        maximum_span: DurationNanos::from_millis(50),
    };
    for offset in [-20_i64, 20] {
        let at = MonotonicNanos::from_nanos(
            (i128::from(source) + i128::from(offset) * 1_000_000) as u64,
        );
        let mapped = mapping.map(at, MCU).expect("mapped time");
        assert_eq!(mapped.at.as_nanos(), ((1000 + offset) * 1_000_000) as u64);
        assert_eq!(mapped.clock, HOST);
        assert_eq!(mapped.error_bound, DurationNanos::from_millis(2));
    }
    assert!(matches!(
        mapping.map(MonotonicNanos::from_nanos(source + 51_000_000), MCU),
        Err(CaptureError::ClockWindow { .. })
    ));
    assert!(matches!(
        mapping.map(MonotonicNanos::from_nanos(source), ClockDomainId::new(3)),
        Err(CaptureError::ClockDomain)
    ));
}

#[test]
fn clock_mapping_refuses_overflow_and_negative_host_time() {
    let mut mapping = ClockAlignment {
        source_clock: MCU,
        source_anchor: MonotonicNanos::from_nanos(10),
        target_anchor: TimedCapture {
            at: MonotonicNanos::from_nanos(1),
            ..time(0, 0)
        },
        maximum_span: DurationNanos::from_nanos(20),
    };
    assert!(mapping.map(MonotonicNanos::from_nanos(0), MCU).is_err());
    mapping.target_anchor.at = MonotonicNanos::from_nanos(u64::MAX - 1);
    assert!(mapping.map(MonotonicNanos::from_nanos(20), MCU).is_err());
}
