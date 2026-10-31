//! Re-exports of sans-IO corridor session and dual-path dead reckoning from `navigate-locate`.

pub use navigate_locate::corridor::session::{
    AnchorWorkerState, DeadReckonEstimate, DeadReckonPath, DualPathDeadReckoner,
    VisualCorridorSession, can_update_vertical_velocity,
};
