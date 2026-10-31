//! Spatial-holdout PnP verification, pose scoring, and `.npz` candidate archives (`photo_pose`).

mod archive;

pub use archive::{LoadedPose, load_candidate_pose, save_candidate_npz};
pub use navigate_visual::holdout_pnp::holdout;
pub use navigate_visual::holdout_pnp::{
    MaskAwareSpatialHoldout, PoseCandidate, PoseScore, QueryRayDomain, candidate_priority,
    fit_pose_hybrid, fit_pose_l1_gnc, fit_pose_ransac, point_jacobian, refine_seed_eye_2pt,
    reproj_error, spatial_holdout,
};
