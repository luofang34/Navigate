use super::*;
use serde_json::json;

#[test]
fn file_anchor_cache_depends_on_its_camera_not_the_query_camera() {
    let camera = json!({"size":[640,480],"focal":[500,500],"principal":[319.5,239.5]});
    let pose = json!({"position":[0,0,100],"rotation":[0,0,0,1]});
    let mut job: Job = serde_json::from_value(json!({
        "query":{"path":"query.png","sha256":"a".repeat(64)},
        "camera":camera,"prior":pose,"position_radius_m":100,"attitude_radius_rad":0.3,
        "anchor_lat_lon":[41,-75],"map_release":"test","map_sha256":"b".repeat(64),
        "references":[{"id":"anchor","pose":pose,"camera":camera,
            "image":{"path":"/bank/image.png","sha256":"c".repeat(64)},
            "depth":"/bank/image.depth","depth_sha256":"d".repeat(64)}]
    }))
    .unwrap();
    let initial = reference_key(&job, Path::new("/first"), &job.references[0]).unwrap();
    job.camera.focal[0] = 600.0;
    let changed_query = reference_key(&job, Path::new("/second"), &job.references[0]).unwrap();
    assert_eq!(initial, changed_query);
    job.references[0].camera.as_mut().unwrap().focal[0] = 550.0;
    assert_ne!(
        initial,
        reference_key(&job, Path::new("/first"), &job.references[0]).unwrap()
    );
    job.references[0].camera = None;
    let inherited = reference_key(&job, Path::new("/first"), &job.references[0]).unwrap();
    job.camera.focal[0] = 700.0;
    assert_ne!(
        inherited,
        reference_key(&job, Path::new("/first"), &job.references[0]).unwrap()
    );
}
