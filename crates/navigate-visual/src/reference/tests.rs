#![allow(clippy::expect_used)]
use super::*;
use crate::{LocalFrame, MapRevision, VisualError};
use image::GrayImage;
use nalgebra::{UnitQuaternion, Vector3};

struct Double {
    calls: Vec<f64>,
    fail_above_m: f64,
}

impl ReferenceRenderer for Double {
    type Error = VisualError;

    fn identity(&self) -> RendererIdentity {
        RendererIdentity {
            revision: "test".into(),
            style_sha256: "b".repeat(64),
        }
    }

    fn render_blocking(&mut self, pose: CameraPose) -> Result<ReferenceView, VisualError> {
        self.calls.push(pose.position.x);
        if pose.position.x > self.fail_above_m {
            return Err(VisualError::Invalid { field: "test pose" });
        }
        Ok(ReferenceView {
            map: MapRevision {
                release_id: "r".into(),
                manifest_sha256: "a".repeat(64),
            },
            frame: LocalFrame::anchor_mercator(40.0, -74.0).expect("anchor"),
            pose,
            image: GrayImage::new(32, 32),
            depth_m: vec![0.0; 32 * 32],
        })
    }
}

fn pose(x: f64) -> CameraPose {
    CameraPose {
        position: Vector3::new(x, 0.0, 500.0),
        orientation: UnitQuaternion::identity(),
    }
}

#[test]
fn the_default_batch_keeps_order_and_stops_at_the_first_failure() {
    let mut renderer = Double {
        calls: vec![],
        fail_above_m: 10.0,
    };
    let views = renderer
        .render_batch_blocking(&[pose(1.0), pose(2.0), pose(3.0)])
        .expect("batch");
    let xs: Vec<f64> = views.iter().map(|v| v.pose.position.x).collect();
    assert_eq!(xs, vec![1.0, 2.0, 3.0]);
    renderer.calls.clear();
    assert!(
        renderer
            .render_batch_blocking(&[pose(1.0), pose(20.0), pose(3.0)])
            .is_err()
    );
    assert_eq!(renderer.calls, vec![1.0, 20.0]);
}
