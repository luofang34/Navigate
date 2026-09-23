//! Fixed pose and image changes for synthetic regression checks.

use crate::{BenchError, renderer::ReferenceRenderer};
use nalgebra::{UnitQuaternion, Vector3};
use navigate_visual::{CameraModel, CameraPose, Frame, FrameStamp, PosePrior};

pub(super) struct Case {
    pitch_deg: f64,
    offset: u8,
    appearance: &'static str,
}

pub(super) fn suite() -> Vec<Case> {
    let mut cases = Vec::new();
    for pitch_deg in [0.0, 45.0, 78.0] {
        for offset in 0..2 {
            for appearance in ["clean", "dim-blur"] {
                cases.push(Case {
                    pitch_deg,
                    offset,
                    appearance,
                });
            }
        }
    }
    cases.push(Case {
        pitch_deg: 0.0,
        offset: 0,
        appearance: "blank",
    });
    cases.push(Case {
        pitch_deg: 0.0,
        offset: 0,
        appearance: "outside-map",
    });
    cases.push(Case {
        pitch_deg: 45.0,
        offset: 0,
        appearance: "recovered",
    });
    cases
}

impl Case {
    pub fn name(&self) -> String {
        format!(
            "pitch-{}-offset-{}-{}",
            self.pitch_deg, self.offset, self.appearance
        )
    }

    pub fn expects_observation(&self) -> bool {
        matches!(self.appearance, "clean" | "dim-blur" | "recovered")
    }

    pub fn frame_blocking(
        &self,
        renderer: &mut ReferenceRenderer,
        camera: CameraModel,
        sequence: u64,
    ) -> Result<(Frame, PosePrior, CameraPose), BenchError> {
        let offset = f64::from(self.offset);
        let truth = CameraPose {
            position: Vector3::new(71.0 + 400.0 * offset, -123.0, 1650.0 + 400.0 * offset),
            orientation: UnitQuaternion::from_euler_angles(
                self.pitch_deg.to_radians(),
                0.0,
                0.12 + 0.2 * offset,
            ),
        };
        let mut image = renderer.render_blocking(truth)?.image;
        if self.appearance == "dim-blur" {
            image = image::imageops::blur(&image, 0.65);
            for (i, pixel) in image.pixels_mut().enumerate() {
                let noise = ((i.wrapping_mul(2654435761) >> 16) % 9) as f64 - 4.0;
                pixel[0] = (f64::from(pixel[0]) * 0.7 + 15.0 + noise).clamp(0.0, 255.0) as u8;
            }
        } else if self.appearance == "blank" {
            image.fill(40);
        }
        let delta = if self.offset == 0 {
            Vector3::new(55.0, -40.0, 30.0)
        } else {
            Vector3::new(-45.0, 35.0, -25.0)
        };
        let mut prior = PosePrior {
            pose: CameraPose {
                position: truth.position + delta,
                orientation: truth.orientation
                    * UnitQuaternion::from_euler_angles(0.012, -0.009, 0.015),
            },
            position_radius_m: 300.0,
            attitude_radius_rad: 0.15,
        };
        if self.appearance == "outside-map" {
            prior.pose.position.x += 50000.0;
        }
        Ok((
            Frame {
                stamp: FrameStamp {
                    sequence,
                    capture_time_ns: sequence * 1_000_000_000,
                },
                camera,
                image,
            },
            prior,
            truth,
        ))
    }
}
