use super::*;

#[test]
fn serialized_camera_preserves_observation_identity() {
    let camera = CameraModel {
        width: 640,
        height: 360,
        fx: 459.33717475790996,
        fy: 459.33717475790996,
        cx: 319.5,
        cy: 179.5,
    };
    let mut frame = Frame {
        stamp: FrameStamp {
            sequence: 570,
            capture_time_ns: 19_019_000_000,
        },
        camera,
        image: image::GrayImage::from_fn(640, 360, |x, y| image::Luma([(x ^ y) as u8])),
    };
    let identity = frame.evidence_sha256();
    let bytes = serde_json::to_vec(&CameraRecord::from(camera)).expect("serialize calibration");
    let restored: CameraRecord = serde_json::from_slice(&bytes).expect("read calibration");
    frame.camera = restored.model();
    assert_eq!(frame.evidence_sha256(), identity);
    assert_eq!(frame.camera.fx.to_bits(), camera.fx.to_bits());
    frame.camera.fx = f64::from_bits(camera.fx.to_bits().wrapping_add(1));
    assert_ne!(
        frame.evidence_sha256(),
        identity,
        "a calibration change changes the evidence identity"
    );
}
