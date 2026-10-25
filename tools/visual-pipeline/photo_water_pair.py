"""Joint water-outline, road-line and two-view camera proposals from frozen seeds."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from photo_line import query_line, reference_line
from photo_pair_seed import prepare
from photo_pair_depth import forest_pairs, forest_training_pairs, forest_residual, forest_check
from photo_pose import spatial_holdout
from photo_shoreline import photo_water, reference_water
from photo_surface import Surface, ContextSurface, basis
from photo_testset import digest, write_json


def contour(mask):
    outlines, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not outlines:
        return None
    xy = max(outlines, key=cv2.contourArea).reshape(-1, 2)
    h, w = mask.shape
    inside = ((xy[:, 0] > 5) & (xy[:, 0] < w - 6) & (xy[:, 1] > 5) & (xy[:, 1] < h - 6))
    return xy[inside][::2]


def shape_check(data, rendered, depth):
    query = photo_water(data['image'], data['labels'], data['valid'])
    reference = reference_water(rendered, depth)
    a, b = contour(query), contour(reference)
    boundary = (float(np.median(np.r_[cKDTree(a).query(b)[0], cKDTree(b).query(a)[0]]))
                if a is not None and b is not None and min(len(a), len(b)) else None)
    strip = reference_line(rendered, depth > 0)
    observed = query_line(data['image'], data['valid'], strip) if strip is not None else None
    road_error = None
    if observed is not None:
        line = cv2.fitLine(observed.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
        road_error = float(np.median(abs((strip - line[2:]) @ np.array([-line[1], line[0]]))))
    return {'water_iou': float(np.sum(query & reference) / max(1, np.sum(query | reference))),
            'water_boundary_median_px': boundary, 'road_strip_to_edge_median_px': road_error,
            'pose_refitted': False}


def evidence(case, root, seed, surface):
    data = prepare(case, root, surface)
    road = reference_line(seed['color'], seed['depth'] > 0)
    observed = query_line(data['image'], data['valid'], road) if road is not None else None
    if observed is None:
        raise ValueError(f"no supported road-line pair: {case['id']}")
    a = contour(reference_water(seed['color'], seed['depth']))
    b = contour(photo_water(data['image'], data['labels'], data['valid']))
    if a is None or b is None or min(len(a), len(b)) < 30:
        raise ValueError(f"no supported water outlines: {case['id']}")
    xy = road.round().astype(int)
    line = cv2.fitLine(observed.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
    data.update(shore_world=seed['world'][a[:, 1], a[:, 0]], shore_pixels=b,
        road_world=seed['world'][xy[:, 1], xy[:, 0]], road_pixels=observed,
        road_origin=line[2:], road_normal=np.array([-line[1], line[0]]),
        eye=seed['eye'].copy(), rotation=seed['rotation'].copy(), K=seed['K'].copy())
    data['vector'] = cv2.Rodrigues(np.diag([1, -1, -1]) @ data['rotation'].T)[0].ravel()
    data['up'] = surface.basis.T @ basis(data['latitude'], data['longitude'])[:, 2]
    position = case['prior']['position']
    data['sigma_m'] = max(30, position.get('reported_horizontal_error_m') or
                         (position.get('search_radius_m') or 300) / 3)
    return data


def camera(data, delta, focal):
    eye = data['eye'] + delta[:3]
    vector = data['vector'] + delta[3:6]
    K = data['K'].copy()
    K[0, 0] *= math.exp(focal)
    K[1, 1] *= math.exp(focal)
    return eye, vector, cv2.Rodrigues(vector)[0], K


def view_residual(data, delta, focal):
    eye, vector, rotation, K = camera(data, delta, focal)
    project = lambda points: cv2.projectPoints(points - eye, vector, np.zeros(3), K, None)[0].reshape(-1, 2)
    uv = project(data['shore_world'])
    n = len(uv)
    road_error = (project(data['road_world']) - data['road_origin']) @ data['road_normal']
    line = np.linalg.inv(K).T @ rotation @ data['up']
    roll = math.degrees(math.atan2(-line[0], line[1]))
    difference = (roll - data['horizon']['roll_deg'] + 90) % 180 - 90
    return np.r_[cKDTree(data['shore_pixels']).query(uv)[0], cKDTree(uv).query(data['shore_pixels'])[0],
        road_error * np.sqrt(n / len(road_error)), delta[:2] / data['sigma_m'] * np.sqrt(n),
        delta[2:3] / 100 * np.sqrt(n), difference / .5 * np.sqrt(n)]


def epipolar(cameras, pairs):
    e1, _, R1, K1 = cameras[0]
    e2, _, R2, K2 = cameras[1]
    t = R2 @ (e1 - e2)
    baseline = np.linalg.norm(t)
    if baseline < 1e-6:
        return np.full(len(pairs), 1e6)
    t /= baseline
    cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.linalg.inv(K2).T @ cross @ R2 @ R1.T @ np.linalg.inv(K1)
    a, b = np.c_[pairs[:, 0], np.ones(len(pairs))], np.c_[pairs[:, 1], np.ones(len(pairs))]
    Fa, Ftb = a @ F.T, b @ F
    return np.sum(b * Fa, axis=1) / np.sqrt(np.maximum(1e-15, np.sum(Fa[:, :2]**2 + Ftb[:, :2]**2, axis=1)))


def optimize(residual, initial, views, fixed_focal=False):
    steps = np.array([.1, .1, .1, 1e-4, 1e-4, 1e-4] * views + [1e-4])
    count = len(initial) - int(fixed_focal)
    expand = lambda x: np.r_[x, initial[-1]] if fixed_focal else x
    def jacobian(x):
        x = expand(x)
        columns = []
        for i, step in enumerate(steps[:count]):
            delta = np.zeros(len(x)); delta[i] = step
            columns.append((residual(x + delta) - residual(x - delta)) / (2 * step))
        return np.column_stack(columns)
    result = least_squares(lambda x: residual(expand(x)), initial[:count], jac=jacobian,
        x_scale=([100, 100, 100, .1, .1, .1] * views + [.1])[:count],
        bounds=((([-1000] * 3 + [-.5] * 3) * views + [math.log(.5)])[:count],
                (([1000] * 3 + [.5] * 3) * views + [math.log(2)])[:count]),
        loss='soft_l1', f_scale=2, max_nfev=400)
    result.x = expand(result.x)
    return result


def solve(data, pairs, surface=None, fixed_focal=False):
    held = spatial_holdout(pairs[:, 0], data[0]['size'])
    raw_train = pairs[~held]
    if len(raw_train) < 60 or held.sum() < 12:
        raise ValueError('insufficient independent photo-pair support')
    cv2.setRNGSeed(0)
    _, consensus = cv2.findFundamentalMat(raw_train[:, 0], raw_train[:, 1], cv2.USAC_MAGSAC, 1., .999, 10000)
    if consensus is None or consensus.sum() < 60:
        raise ValueError('photo-pair training geometry did not form a consensus')
    consensus_pairs = raw_train[consensus.ravel() > 0]
    train = consensus_pairs[::4]
    initial = np.zeros(13)
    for i, view in enumerate(data):
        fitted = optimize(lambda x: np.r_[view_residual(view, x[:6], x[6]), x[6] / .25 * 10],
                          np.zeros(7), 1, fixed_focal)
        initial[i * 6:i * 6 + 6] = fitted.x[:6]
        initial[12] += fitted.x[6] / 2
    cameras = lambda x: [camera(view, x[i * 6:i * 6 + 6], x[12]) for i, view in enumerate(data)]
    def residual(x):
        terms = [view_residual(view, x[i * 6:i * 6 + 6], x[12]) for i, view in enumerate(data)]
        return np.concatenate([*terms, epipolar(cameras(x), train) * 2, [x[12] / .15 * 10]])
    result = optimize(residual, initial, 2, fixed_focal)
    training_forest = forest_training_pairs(data, consensus_pairs)
    depth_report = {'status': 'not_evaluated', 'training_pairs': len(training_forest)}
    if surface is not None:
        held_forest = pairs[held][forest_pairs(data, pairs[held])]
        depth_report['before'] = forest_check(cameras(result.x), held_forest, surface)
        if len(training_forest) >= 30:
            weight = np.sqrt(sum(len(view['shore_world']) for view in data) / len(training_forest))
            result = optimize(lambda x: np.r_[residual(x), weight * forest_residual(
                cameras(x), training_forest, surface)], result.x, 2, fixed_focal)
        depth_report.update(forest_check(cameras(result.x), held_forest, surface))
        depth_report['constraint_applied'] = len(training_forest) >= 30
    error = np.abs(epipolar(cameras(result.x), pairs[held]))
    report = {'status': 'research_proposal' if result.success else 'optimization_incomplete',
        'geometry_accepted': False, 'requires_map_verification': True,
        'optimizer_converged': bool(result.success), 'focal_factor': math.exp(result.x[12]),
        'focal_model': 'fixed_seed' if fixed_focal else 'shared_correction',
        'forest_clearance': depth_report,
        'photo_pairs': len(pairs), 'training_pairs': len(train), 'held_pairs': int(held.sum()),
        'held_epipolar_median_px': float(np.median(error)), 'held_epipolar_inlier_fraction': float(np.mean(error < 2)),
        'holdout_filtered_by_fitted_geometry': False, 'x': result.x.tolist()}
    return cameras(result.x), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path); parser.add_argument('seeds', type=Path)
    parser.add_argument('output', type=Path); parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--fix-focal', action='store_true',
                        help='preserve seed camera intrinsics while fitting the two poses')
    parser.add_argument('--context-catalog', type=Path, help='wider imagery and terrain for uncovered rays')
    parser.add_argument('--distant-water', action='store_true',
                        help='fit one paired reflective lake and hold out a second map landmark')
    args = parser.parse_args(); cv2.setNumThreads(4)
    if args.distant_water and args.context_catalog is None:
        parser.error('--distant-water requires --context-catalog')
    config = json.loads((args.seeds / 'run.json').read_text())
    if config['profile_sha256'] != digest(args.profile) or config['catalog_sha256'] != digest(args.catalog):
        parser.error('pair seeds must use the exact profile and catalog')
    profile = json.loads(args.profile.read_text()); root = (args.profile.parent / profile['root']).resolve()
    cases = [next(c for c in profile['cases'] if c['id'] == q) for q in config['cases']]
    calibration = lambda c: (c['prior'].get('camera_model'), c['prior'].get('lens_model'), c['prior'].get('focal_35mm_mm'))
    if calibration(cases[0]) != calibration(cases[1]):
        parser.error('a shared focal correction requires matching camera and lens claims')
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {'inputs': {p.name: digest(p) for p in args.seeds.glob('*.npz')},
        'profile_sha256': digest(args.profile), 'catalog_sha256': digest(args.catalog),
        'cases': config['cases'],
        'fixed_focal': args.fix_focal,
        'distant_water_enabled': args.distant_water,
        'context_catalog_sha256': digest(args.context_catalog) if args.context_catalog else None,
        'seed_run_sha256': digest(args.seeds / 'run.json'),
        'code': {p.name: digest(p) for p in Path(__file__).parent.glob('photo_*.py')},
        'opencv': cv2.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__}
    write_json(args.output / 'run.json', provenance)
    surface = Surface(args.catalog); data = []
    if args.context_catalog:
        surface = ContextSurface(surface, Surface(args.context_catalog))
    for i, case in enumerate(cases):
        with np.load(args.seeds / f'seed-{i}.npz') as seed:
            data.append(evidence(case, root, seed, surface))
    with np.load(args.seeds / 'pairs.npz') as matched:
        if args.distant_water:
            from photo_water_context import solve_landmarks
            cameras, report = solve_landmarks(data, matched['pairs'], matched['homography'],
                                              surface, args.output, args.fix_focal)
        else:
            cameras, report = solve(data, matched['pairs'], surface, args.fix_focal)
    if any(digest(args.seeds / name) != value for name, value in provenance['inputs'].items()):
        raise ValueError('pair seeds changed during fitting')
    if args.context_catalog and digest(args.context_catalog) != provenance['context_catalog_sha256']:
        raise ValueError('context catalog changed during fitting')
    report.update(provenance)
    report['positions'] = []
    report['shape_checks'] = []
    for i, (view, (eye, _, rotation, K)) in enumerate(zip(data, cameras)):
        pose = rotation.T @ np.diag([1, -1, -1])
        color, depth, _ = surface.render(eye, pose, K, view['size'])
        np.savez_compressed(args.output / f'camera-{i}.npz', eye=eye, rotation=pose, K=K, size=view['size'])
        cv2.imwrite(str(args.output / f'comparison-{i}.jpg'), np.concatenate(
            [view['image'], color, cv2.addWeighted(view['image'], .5, color, .5, 0)]))
        report['positions'].append(list(map(float, surface.geodetic(eye))))
        report['shape_checks'].append(shape_check(view, color, depth))
    write_json(args.output / 'report.json', report)
    for i, case in enumerate(cases):
        write_json(args.output / (case['id'] + '.json'), {'id': case['id'],
            'status': report['status'], 'geometry_accepted': False,
            'requires_map_verification': True, 'camera': f'camera-{i}.npz',
            'camera_sha256': digest(args.output / f'camera-{i}.npz'),
            'report_sha256': digest(args.output / 'report.json')})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
