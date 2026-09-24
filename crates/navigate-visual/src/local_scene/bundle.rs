//! Refine conditional camera poses and scene points.
use super::{SceneCoordinateGauge, pose::Pose};
use crate::CameraModel as Camera;
use nalgebra::{DMatrix, DVector, Vector2};
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use std::collections::BTreeSet;

type Cross = SMatrix<f64, 6, 3>;
type Hessian = SMatrix<f64, 6, 6>;
type Motion = SVector<f64, 6>;
#[derive(Clone)]
pub(super) struct Observation {
    pub frame: usize,
    pub pixel: Vector2<f64>,
}
#[derive(Clone)]
pub(super) struct Landmark {
    pub world: Vector3<f64>,
    pub observations: Vec<Observation>,
}
struct PointNormal {
    h: Matrix3<f64>,
    b: Vector3<f64>,
    cross: Vec<(usize, Cross)>,
}
struct Normal {
    cameras: Vec<Hessian>,
    rhs: Vec<Motion>,
    points: Vec<PointNormal>,
}
pub(super) struct Statistics {
    pub initial_cost: f64,
    pub final_cost: f64,
    pub steps: usize,
}
fn projection(camera: &Camera, p: Vector3<f64>) -> SMatrix<f64, 2, 3> {
    SMatrix::from_row_slice(&[
        camera.fx / p.z,
        0.0,
        -camera.fx * p.x / p.z.powi(2),
        0.0,
        camera.fy / p.z,
        -camera.fy * p.y / p.z.powi(2),
    ])
}
fn accumulate(
    camera: &Camera,
    poses: &[Pose],
    landmarks: &[Landmark],
    indices: &[Option<usize>],
    count: usize,
) -> Normal {
    let mut normal = Normal {
        cameras: vec![Hessian::zeros(); count],
        rhs: vec![Motion::zeros(); count],
        points: Vec::new(),
    };
    for landmark in landmarks {
        let mut point = PointNormal {
            h: Matrix3::zeros(),
            b: Vector3::zeros(),
            cross: Vec::new(),
        };
        for observation in &landmark.observations {
            let pose = poses[observation.frame];
            let Some(pixel) = pose.project(camera, landmark.world) else {
                continue;
            };
            let p = pose.r * landmark.world + pose.t;
            let error = observation.pixel - pixel;
            let weight = 9.0 / (9.0 + error.norm_squared());
            let image = projection(camera, p);
            let jp = image * pose.r;
            point.h += jp.transpose() * jp * weight;
            point.b += jp.transpose() * error * weight;
            if let Some(index) = indices[observation.frame] {
                let mut motion = SMatrix::<f64, 3, 6>::zeros();
                motion
                    .fixed_view_mut::<3, 3>(0, 0)
                    .copy_from(&Matrix3::identity());
                motion
                    .fixed_view_mut::<3, 3>(0, 3)
                    .copy_from(&(-p.cross_matrix() * 0.001));
                let jc = image * motion;
                normal.cameras[index] += jc.transpose() * jc * weight;
                normal.rhs[index] += jc.transpose() * error * weight;
                point.cross.push((index, jc.transpose() * jp * weight));
            }
        }
        normal.points.push(point);
    }
    normal
}
fn solve(normal: &Normal, damping: f64) -> Option<(Vec<Motion>, Vec<Vector3<f64>>)> {
    let count = normal.cameras.len();
    let mut reduced = DMatrix::zeros(count * 6, count * 6);
    let mut rhs = DVector::zeros(count * 6);
    for (i, (h, b)) in normal.cameras.iter().zip(&normal.rhs).enumerate() {
        let damped = h + Hessian::from_diagonal(&h.diagonal().map(|v| damping * v.max(1e-9)));
        reduced.view_mut((i * 6, i * 6), (6, 6)).copy_from(&damped);
        rhs.rows_mut(i * 6, 6).copy_from(b);
    }
    let mut inverses = Vec::new();
    for point in &normal.points {
        let damped =
            point.h + Matrix3::from_diagonal(&point.h.diagonal().map(|v| damping * v.max(1e-9)));
        let inverse = damped.try_inverse()?;
        for (i, left) in &point.cross {
            let projected = left * inverse;
            for r in 0..6 {
                rhs[i * 6 + r] -= (projected * point.b)[r]
            }
            for (j, right) in &point.cross {
                let block = projected * right.transpose();
                for r in 0..6 {
                    for c in 0..6 {
                        reduced[(i * 6 + r, j * 6 + c)] -= block[(r, c)]
                    }
                }
            }
        }
        inverses.push(inverse);
    }
    let camera_delta = if count == 0 {
        DVector::zeros(0)
    } else {
        reduced.cholesky()?.solve(&rhs)
    };
    let cameras: Vec<Motion> = camera_delta
        .as_slice()
        .as_chunks::<6>()
        .0
        .iter()
        .map(|values| Motion::from_row_slice(values))
        .collect();
    let points = normal
        .points
        .iter()
        .zip(inverses)
        .map(|(point, inverse)| {
            let mut b = point.b;
            for (i, cross) in &point.cross {
                b -= cross.transpose() * cameras[*i]
            }
            inverse * b
        })
        .collect::<Vec<_>>();
    cameras
        .iter()
        .flat_map(|v| v.iter())
        .chain(points.iter().flat_map(|v| v.iter()))
        .all(|v| v.is_finite())
        .then_some((cameras, points))
}
fn cost(camera: &Camera, poses: &[Pose], points: &[Landmark]) -> f64 {
    points
        .iter()
        .map(|point| {
            point
                .observations
                .iter()
                .map(|observation| {
                    let error = poses[observation.frame]
                        .project(camera, point.world)
                        .map_or(1000.0, |p| (p - observation.pixel).norm().min(1000.0));
                    4.5 * (error * error / 9.0).ln_1p()
                })
                .sum::<f64>()
        })
        .sum()
}
fn gauge_indices(
    poses: &[Pose],
    points: &[Landmark],
    fixed: &BTreeSet<usize>,
    coordinate_gauge: Option<SceneCoordinateGauge>,
) -> Option<(Vec<Option<usize>>, usize)> {
    let mut support_fixed = fixed.clone();
    if let Some(gauge) = coordinate_gauge {
        support_fixed.insert(gauge.scale_camera);
    }
    if points.is_empty()
        || points.len() > 16384
        || poses
            .iter()
            .any(|p| p.r.iter().chain(p.t.iter()).any(|x| !x.is_finite()))
    {
        return None;
    }
    for point in points {
        let mut frames = BTreeSet::new();
        if point.observations.len() < 2
            || !point.world.iter().all(|x| x.is_finite())
            || point
                .observations
                .iter()
                .any(|o| !frames.insert(o.frame) || !o.pixel.iter().all(|x| x.is_finite()))
        {
            return None;
        }
    }
    if support_fixed.len() < 2
        || support_fixed.iter().any(|&i| i >= poses.len())
        || points
            .iter()
            .any(|p| p.observations.iter().any(|o| o.frame >= poses.len()))
    {
        return None;
    }
    let baseline = support_fixed.iter().any(|&a| {
        support_fixed
            .iter()
            .any(|&b| (poses[a].center() - poses[b].center()).norm() > 1e-6)
    });
    if !baseline {
        return None;
    }
    let mut count = 0_usize;
    let indices: Vec<_> = (0..poses.len())
        .map(|i| {
            if fixed.contains(&i) {
                None
            } else {
                let id = count;
                count = count.wrapping_add(1);
                Some(id)
            }
        })
        .collect();
    (count <= if coordinate_gauge.is_some() { 128 } else { 96 }).then_some((indices, count))
}
pub(super) fn refine(
    camera: &Camera,
    poses: &mut [Pose],
    points: &mut [Landmark],
    fixed: &BTreeSet<usize>,
    iterations: usize,
    coordinate_gauge: Option<SceneCoordinateGauge>,
) -> Option<Statistics> {
    if !(1..=100).contains(&iterations) {
        return None;
    }
    if ![camera.fx, camera.fy, camera.cx, camera.cy]
        .iter()
        .all(|v| v.is_finite())
        || camera.fx <= 0.0
        || camera.fy <= 0.0
    {
        return None;
    }
    let (indices, count) = gauge_indices(poses, points, fixed, coordinate_gauge)?;
    let initial_cost = cost(camera, poses, points);
    let mut current = initial_cost;
    let mut damping = 0.001;
    let mut steps = 0_usize;
    for _ in 0..iterations {
        let normal = accumulate(camera, poses, points, &indices, count);
        let Some((camera_delta, point_delta)) = solve(&normal, damping) else {
            damping *= 10.0;
            if damping > 1e10 {
                break;
            }
            continue;
        };
        let mut next_poses = poses
            .iter()
            .zip(&indices)
            .map(|(&p, index)| index.map_or(p, |i| p.increment(camera_delta[i])))
            .collect::<Vec<_>>();
        let mut next_points = points.to_vec();
        for (point, delta) in next_points.iter_mut().zip(point_delta) {
            point.world += delta
        }
        if let Some(gauge) = coordinate_gauge {
            super::scale::normalize(poses, &mut next_poses, &mut next_points, gauge)?;
        }
        let next = cost(camera, &next_poses, &next_points);
        if next < current {
            poses.copy_from_slice(&next_poses);
            points.clone_from_slice(&next_points);
            let improvement = current - next;
            current = next;
            steps = steps.wrapping_add(1);
            damping = (damping * 0.3).max(1e-8);
            if improvement < 1e-9 {
                break;
            }
        } else {
            damping *= 10.0;
            if damping > 1e10 {
                break;
            }
        }
    }
    Some(Statistics {
        initial_cost,
        final_cost: current,
        steps,
    })
}
