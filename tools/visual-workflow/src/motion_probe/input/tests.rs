use super::*;
use sha2::{Digest, Sha256};
#[test]
fn verified_result_binds_pixels_calibration_and_timestamp() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run.json");
    let camera = Camera {
        size: [64, 64],
        focal: [50.0, 51.0],
        principal: [31.5, 31.5],
    };
    let image = image::GrayImage::new(64, 64);
    let spec = Manifest {
        job: Image {
            path: "job.json".into(),
            sha256: "unused".into(),
        },
        runs: Vec::new(),
        lag: 1,
        frames: vec![Sample {
            image: Image {
                path: "frame.png".into(),
                sha256: "unused".into(),
            },
            capture_time_ns: 12,
        }],
    };
    let data = json!({"query_camera":camera,"query_transform":null,"query_focal_scale":1.0,"frames":[{"index":0,"capture_time_ns":12,"observation_sha256":digest(&image,camera.model(),0,12),"accepted":false,"pose":null,"evidence_kind":"rejected","map_refreshed":false}]});
    assets::write_json_blocking(&path, &data).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let asset = Image {
        path: path.clone(),
        sha256: format!("{:x}", Sha256::digest(&bytes)),
    };
    let run = Run::load_blocking(&asset, directory.path(), 1).unwrap();
    assert!(run.bind(&spec, 0, &image).is_ok());
    let mut changed = image;
    changed.put_pixel(20, 20, image::Luma([1]));
    assert!(run.bind(&spec, 0, &changed).is_err());
    let mut wrong_time = spec;
    wrong_time.frames[0].capture_time_ns = 13;
    assert!(
        run.bind(&wrong_time, 0, &image::GrayImage::new(64, 64))
            .is_err()
    );
    std::fs::write(&path, b"{}").unwrap();
    assert!(Run::load_blocking(&asset, directory.path(), 1).is_err());
}

#[test]
fn frame_pair_cannot_bridge_a_rejected_pose() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("gap.json");
    let frames: Vec<_>=(0..3).map(|index|json!({"index":index,"capture_time_ns":index*10,"observation_sha256":"unused",
        "accepted":index!=1,"pose":if index==1 {Value::Null} else {json!({"position":[index,0,10],"rotation":[0,0,0,1]})},
        "evidence_kind":"map_match","map_refreshed":true})).collect();
    assets::write_json_blocking(&path,&json!({"query_camera":{"size":[64,64],"focal":[50,50],"principal":[31.5,31.5]},"query_transform":null,"query_focal_scale":1.0,"frames":frames})).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let asset = Image {
        path: path.clone(),
        sha256: format!("{:x}", Sha256::digest(bytes)),
    };
    let run = Run::load_blocking(&asset, directory.path(), 3).unwrap();
    assert_eq!(run.score(0, 2, &[]).unwrap()["available"], false);
}
