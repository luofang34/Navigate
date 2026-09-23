//! Explicit fixed-reference comparison inputs.
use nalgebra::{Quaternion, UnitQuaternion};
use navigate_visual::{CameraModel, CameraPose, PosePrior};
use navigate_visual_onnx::{ExecutionConfig, MatcherFiles, Provider};
use serde::Deserialize;
use std::path::{Path, PathBuf};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Suite {
    pub library: PathBuf,
    pub models: Vec<Model>,
    pub cases: Vec<Case>,
    #[serde(default = "default_keypoints")]
    pub keypoints: usize,
    #[serde(default)]
    pub provider: RunProvider,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Model {
    Xfeat {
        path: PathBuf,
    },
    XfeatLighterglue {
        detector: PathBuf,
        matcher: PathBuf,
    },
    XfeatDense {
        detector: PathBuf,
        matcher: Option<PathBuf>,
    },
    Superglue {
        detector: PathBuf,
        matcher: PathBuf,
        image_size: [usize; 2],
    },
}
impl Model {
    pub fn files(&self, root: &Path) -> MatcherFiles {
        match self {
            Self::Xfeat { path } => MatcherFiles::XFeat {
                model: root.join(path),
            },
            Self::XfeatLighterglue { detector, matcher } => MatcherFiles::XFeatLighterGlue {
                detector: root.join(detector),
                matcher: root.join(matcher),
            },
            Self::XfeatDense { detector, matcher } => MatcherFiles::XFeatDense {
                detector: root.join(detector),
                matcher: matcher.as_ref().map(|p| root.join(p)),
            },
            Self::Superglue {
                detector,
                matcher,
                image_size,
            } => MatcherFiles::SuperGlue {
                detector: root.join(detector),
                matcher: root.join(matcher),
                image_size: *image_size,
            },
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Case {
    pub id: String,
    pub query: PathBuf,
    pub reference: PathBuf,
    pub depth: PathBuf,
    pub camera: Camera,
    pub reference_pose: Pose,
    pub prior: Prior,
    pub anchor_lat_lon: [f64; 2],
    pub map_release_id: String,
    pub map_manifest_sha256: String,
    pub reference_image_sha256: String,
    pub reference_depth_sha256: String,
    pub query_image_sha256: String,
}
#[derive(Deserialize, Clone, Copy)]
#[serde(deny_unknown_fields)]
pub(super) struct Camera {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}
impl Camera {
    pub fn model(self) -> CameraModel {
        CameraModel {
            width: self.width,
            height: self.height,
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
        }
    }
}
#[derive(Deserialize, Clone, Copy)]
#[serde(deny_unknown_fields)]
pub(super) struct Pose {
    pub position_enu_m: [f64; 3],
    pub eye_to_enu_xyzw: [f64; 4],
}
impl Pose {
    pub fn model(self) -> Result<CameraPose, Box<dyn std::error::Error>> {
        let [x, y, z, w] = self.eye_to_enu_xyzw;
        let q = Quaternion::new(w, x, y, z);
        if !q.coords.iter().all(|v| v.is_finite()) || (q.norm() - 1.0).abs() > 1e-5 {
            return Err("invalid unit quaternion".into());
        }
        let pose = CameraPose {
            position: self.position_enu_m.into(),
            orientation: UnitQuaternion::new_normalize(q),
        };
        pose.validate()?;
        Ok(pose)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Prior {
    pub pose: Pose,
    pub position_radius_m: f64,
    pub attitude_radius_rad: f64,
}
impl Prior {
    pub fn model(&self) -> Result<PosePrior, Box<dyn std::error::Error>> {
        Ok(PosePrior {
            pose: self.pose.model()?,
            position_radius_m: self.position_radius_m,
            attitude_radius_rad: self.attitude_radius_rad,
        })
    }
}

fn default_keypoints() -> usize {
    2048
}
#[derive(Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub(super) enum RunProvider {
    #[default]
    Cpu,
    CoremlAne,
    CoremlGpu,
    Cuda,
    TensorRt,
}
impl RunProvider {
    pub fn execution(&self) -> ExecutionConfig {
        ExecutionConfig {
            provider: match self {
                Self::Cpu => Provider::Cpu,
                Self::CoremlAne => Provider::CoreMlAne,
                Self::CoremlGpu => Provider::CoreMlGpu,
                Self::Cuda => Provider::Cuda,
                Self::TensorRt => Provider::TensorRt,
            },
            threads: 4,
            ..ExecutionConfig::default()
        }
    }
}
