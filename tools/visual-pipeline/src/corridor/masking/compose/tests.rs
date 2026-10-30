use super::compose_amortized_u8;
use nalgebra::{Matrix3, Rotation3, Vector3};

#[test]
fn homography_parallel_warp_matches_ray_projection_at_800x600() {
    let (w, h) = (800u32, 600u32);
    let (width, height) = (w as usize, h as usize);
    let total = width * height;
    let air: Vec<u8> = (0..total)
        .map(|i| if (i % width) < 48 { 0 } else { 255 })
        .collect();
    let sky: Vec<u8> = (0..total)
        .map(|i| if (i / width) < 180 { 0 } else { 255 })
        .collect();
    let prop: Vec<bool> = (0..total).map(|i| (i % 97) == 0).collect();
    let k = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
    let r_key = Matrix3::identity();
    let r_cur = *Rotation3::from_euler_angles(0.03, -0.04, 0.05).matrix();

    let (fast, valid_fast) =
        compose_amortized_u8([w, h], &air, &sky, &prop, Some(&r_key), Some(&r_cur), &k);
    let delta = r_key * r_cur.transpose();
    let mut expected = vec![0u8; total];
    let mut valid_exp = 0usize;
    for y in 0..height {
        let yc = (y as f64 - 299.5) / 600.0;
        for x in 0..width {
            let idx = y * width + x;
            if air[idx] == 0 || prop[idx] {
                continue;
            }
            let ray = delta * Vector3::new((x as f64 - 399.5) / 600.0, yc, 1.0);
            if ray.z <= 1e-3 {
                continue;
            }
            let uk = (600.0 * (ray.x / ray.z) + 399.5).round().clamp(0.0, 799.0) as usize;
            let vk = (600.0 * (ray.y / ray.z) + 299.5).round().clamp(0.0, 599.0) as usize;
            let keep = sky[vk * width + uk];
            expected[idx] = keep;
            valid_exp += usize::from(keep != 0);
        }
    }
    assert_eq!(valid_fast, valid_exp);
    assert_eq!(fast, expected);
}
