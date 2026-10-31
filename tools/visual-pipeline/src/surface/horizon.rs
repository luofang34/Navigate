//! Robust horizon line fitting, airframe and propeller exclusion, and attitude estimation.

pub use navigate_segment::horizon::{
    ADE20K_AIRPLANE_LABEL, ADE20K_SKY_LABEL, AIRFRAME_LABEL, GROUND_LABEL, PROPELLER_LABEL,
    RansacHorizonEstimator, SKY_LABEL, horizon_attitude, ransac_fit_line, theil_sen_fit_line,
};
