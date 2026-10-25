use super::*;

#[test]
fn dji_xmp_gimbal_values_are_read_from_attributes() {
    let xmp = br#"<x:xmpmeta><rdf:Description drone-dji:RelativeAltitude="+99.20" drone-dji:GimbalYawDegree="-85.30" drone-dji:GimbalPitchDegree="-39.90"/></x:xmpmeta>"#;
    let metadata = PhotoMetadata::read(xmp, [4000, 3000]);
    assert_eq!(metadata.gimbal_yaw_deg, Some(-85.3));
    assert_eq!(metadata.gimbal_pitch_deg, Some(-39.9));
    assert_eq!(metadata.relative_altitude_m, Some(99.2));
    assert_eq!(metadata.focal_px, None);
}

#[test]
fn xmp_elements_are_read_too() {
    let xmp = b"<drone-dji:GimbalYawDegree>275.5</drone-dji:GimbalYawDegree>";
    assert_eq!(xmp_number(xmp, "drone-dji:GimbalYawDegree"), Some(275.5));
}

#[test]
fn a_photo_without_metadata_has_no_hints() {
    assert_eq!(
        PhotoMetadata::read(b"\xff\xd8\xff\xd9", [10, 10]),
        PhotoMetadata::default()
    );
}
