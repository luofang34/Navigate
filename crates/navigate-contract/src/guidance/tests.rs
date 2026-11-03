//! The space setpoints are plain `Copy` values that a command carries
//! without heap memory.

use crate::frame::FrameId;
use crate::stamp::{SolutionStamp, SourceEpoch, WrappingSequence};
use crate::time::{ClockDomainId, MonotonicNanos};

use super::{GuidanceCommand, GuidanceSetpoint};

fn stamp() -> SolutionStamp {
    SolutionStamp::new(
        SourceEpoch::new(1),
        WrappingSequence::new(7),
        MonotonicNanos::from_nanos(10),
        ClockDomainId::new(0),
    )
}

fn assert_copy<T: Copy>(_: &T) {}

#[test]
fn space_setpoints_are_copy_and_compare_by_value() {
    let setpoints = [
        GuidanceSetpoint::ThrustVector {
            frame: FrameId::EARTH_J2000,
            direction_unit: [0.0, -1.0, 0.0],
            specific_force_mps2: 0.6,
            velocity_to_go_mps: 108.0,
            cutoff: false,
        },
        GuidanceSetpoint::EntryBank {
            bank_rad: -1.0,
            reference_drag_mps2: 8.0,
            drag_error_mps2: 0.25,
            heading_course_rad: 1.5,
        },
        GuidanceSetpoint::CartesianVelocity {
            frame: FrameId::MOON_J2000,
            velocity_mps: [1.0, 2.0, 3.0],
        },
    ];
    for setpoint in setpoints {
        let command = GuidanceCommand::new(MonotonicNanos::from_nanos(11), setpoint, stamp());
        let copy = command;
        assert_copy(&command);
        assert_eq!(copy, command);
        assert_eq!(copy.setpoint, setpoint);
    }
}

#[test]
fn a_thrust_vector_names_its_frame() {
    let inertial = GuidanceSetpoint::ThrustVector {
        frame: FrameId::EARTH_J2000,
        direction_unit: [1.0, 0.0, 0.0],
        specific_force_mps2: 0.0,
        velocity_to_go_mps: 0.0,
        cutoff: true,
    };
    let body_fixed = GuidanceSetpoint::ThrustVector {
        frame: FrameId::EARTH_ITRF93,
        direction_unit: [1.0, 0.0, 0.0],
        specific_force_mps2: 0.0,
        velocity_to_go_mps: 0.0,
        cutoff: true,
    };
    assert_ne!(inertial, body_fixed);
}
