//! The float functions of this crate. The `std` or `libm` feature of
//! `navigate-geodesy` selects the implementation, so one feature list
//! selects it for every crate of the workspace.

pub(crate) use navigate_geodesy::float_math::{
    atan2, ceil, exp, hypot, ln, powi, round, sin_cos, sqrt,
};
