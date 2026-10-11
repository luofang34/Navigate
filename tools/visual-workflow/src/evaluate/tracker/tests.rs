use super::*;

#[test]
fn tracker_selection_is_explicit_and_rejects_unknown_backends() {
    assert!(matches!(
        serde_json::from_str::<TrackerBackend>("\"cpu\"").expect("CPU choice"),
        TrackerBackend::Cpu
    ));
    assert!(matches!(
        serde_json::from_str::<TrackerBackend>("\"wgpu\"").expect("GPU choice"),
        TrackerBackend::Wgpu
    ));
    assert!(serde_json::from_str::<TrackerBackend>("\"automatic\"").is_err());
}

#[cfg(not(feature = "gpu"))]
#[test]
fn requesting_an_unbuilt_gpu_backend_does_not_fall_back() {
    assert!(Tracker::new_blocking(TrackerBackend::Wgpu).is_err());
}
