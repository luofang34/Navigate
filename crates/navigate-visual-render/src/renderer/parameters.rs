use super::*;

pub(super) fn make(renderer: &GpuRenderer, pose: CameraPose) -> [f32; 64] {
    let mut p = [0.0; 64];
    let c = renderer.camera;
    p[..8].copy_from_slice(&[
        c.width as f32,
        c.height as f32,
        c.fx as f32,
        c.fy as f32,
        c.cx as f32,
        c.cy as f32,
        renderer.limits.far_m,
        renderer.limits.steps as f32,
    ]);
    p[8..11].copy_from_slice(&pose.position.map(|v| v as f32).as_slice()[..3]);
    p[11] = renderer.limits.refinements as f32;
    let rotation = pose.orientation.to_rotation_matrix();
    for col in 0..3 {
        for row in 0..3 {
            p[12 + col * 4 + row] = rotation[(row, col)] as f32;
        }
    }
    p[24..28].copy_from_slice(&renderer.imagery.projection(renderer.frame));
    p[28..32].copy_from_slice(&renderer.terrain.projection(renderer.frame));
    let (s, c) = renderer.frame.anchor()[0].to_radians().sin_cos();
    p[32] = s as f32;
    p[33] = c as f32;
    match renderer.frame {
        LocalFrame::Wgs84Enu { .. } => {
            let a = navigate_geodesy::wgs84::SEMI_MAJOR_AXIS_M;
            let e2 = navigate_geodesy::wgs84::FIRST_ECCENTRICITY_SQUARED;
            let d0_sqrt = (1.0 - e2 * s * s).sqrt();
            let n0 = a / d0_sqrt;
            p[34] = n0 as f32;
            p[35] = 2.0;
            p[36] = e2 as f32;
            p[37] = d0_sqrt as f32;
            p[38] = (a * (1.0 - e2)) as f32;
        }
        LocalFrame::SphericalEnu { .. } => {
            p[34] = navigate_visual::MERCATOR_SPHERE_RADIUS_M as f32;
            p[35] = 1.0;
        }
        _ => {
            p[34] = navigate_visual::MERCATOR_SPHERE_RADIUS_M as f32;
            p[35] = 0.0;
        }
    }
    p
}
