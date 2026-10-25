use super::*;
use nalgebra::{SVector, Translation3, UnitQuaternion};

#[test]
fn the_transfer_jacobian_matches_central_differences() {
    let frame = Pose::from_parts(
        Translation3::new(5.0, -3.0, 110.0),
        UnitQuaternion::from_euler_angles(3.0, 0.1, -0.7),
    );
    let transfer = Pose::from_parts(
        Translation3::new(4.0, 11.0, -0.5),
        UnitQuaternion::from_euler_angles(0.02, -0.03, 0.2),
    );
    let keyframe = frame * transfer;
    let jacobian = transfer_jacobian(&frame, &transfer);
    let h = 1e-6;
    for k in 0..6 {
        let moved = |sign: f64| {
            let mut delta = SVector::<f64, 6>::zeros();
            delta[k] = sign * h;
            let rotation = frame.rotation
                * UnitQuaternion::from_scaled_axis(delta.fixed_rows::<3>(3).into_owned());
            let position = frame.translation.vector + delta.fixed_rows::<3>(0);
            let moved = Pose::from_parts(position.into(), rotation) * transfer;
            let mut error = SVector::<f64, 6>::zeros();
            error
                .fixed_rows_mut::<3>(0)
                .copy_from(&(moved.translation.vector - keyframe.translation.vector));
            error
                .fixed_rows_mut::<3>(3)
                .copy_from(&(keyframe.rotation.inverse() * moved.rotation).scaled_axis());
            error
        };
        let numeric = (moved(1.0) - moved(-1.0)) / (2.0 * h);
        let analytic = jacobian.column(k);
        assert!(
            (numeric - analytic).norm() < 1e-6,
            "column {k}: {numeric:?} vs {analytic:?}"
        );
    }
}
