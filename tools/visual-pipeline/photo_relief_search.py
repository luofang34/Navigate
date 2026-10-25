"""Propose and independently check local surface relief from a terrain pose seed."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import scipy

from photo_pose import fit, ground_pairs
from photo_relief import fit_relief
from photo_relief_surface import ReliefSurface, region_mask, save_correction
from photo_surface import Surface, basis
from photo_testset import digest, write_json
from photo_tracking import matching_pixels, prepared


def dominant_region(state, labels):
    """Select the dominant observed class, without assuming a building label."""
    pixels = state['pixels']
    xy = pixels.round().astype(int)
    inside = ((xy >= 0).all(axis=1) & (xy[:, 0] < labels.shape[1])
              & (xy[:, 1] < labels.shape[0]))
    supported = np.flatnonzero(inside & state['inliers'])
    if len(supported) < 60:
        return None
    values = labels[xy[supported, 1], xy[supported, 0]]
    label = int(np.bincount(values).argmax())
    selected = supported[values == label]
    if len(selected) < 60 or len(selected) < .8 * len(supported):
        return None
    hull = np.zeros(labels.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(pixels[selected].astype(np.float32)).round().astype(int), 255)
    return selected, hull, label


def exclude_region(valid, pixels):
    hull = np.zeros(valid.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(pixels.astype(np.float32)).round().astype(int), 255)
    # The margin separates roof edges from curb evidence at a 640-pixel long edge.
    hull = cv2.dilate(hull, np.ones((41, 41), np.uint8))
    mask = valid.copy()
    mask[hull > 0] = 0
    return mask


def ground_patch_pairs(image, color, mask, reference_mask, matcher, directory, height):
    """Give small ground regions matching support independent of the masked roof."""
    h, w = mask.shape
    pairs = []
    for top, bottom in [(0, h // 2), (h // 2, h)]:
        for left, right in [(0, w // 2), (w // 2, w)]:
            area = np.s_[top:bottom, left:right]
            if min(np.mean(mask[area] > 0), np.mean(reference_mask[area] > 0)) < .2:
                continue
            query = directory / f'ground-query-{height}-{left}-{top}.png'
            reference = directory / f'ground-reference-{height}-{left}-{top}.png'
            cv2.imwrite(str(query), matching_pixels(image[area], mask[area], matcher))
            cv2.imwrite(str(reference), matching_pixels(color[area], reference_mask[area], matcher))
            response = matcher.match(reference, query)
            patch = np.asarray(response['pairs'], float).reshape(-1, 2, 2)
            inside = (np.isfinite(patch).all(axis=(1, 2)) & (patch >= 0).all(axis=(1, 2))
                      & (patch[:, :, 0] < right - left).all(axis=1)
                      & (patch[:, :, 1] < bottom - top).all(axis=1))
            pairs.extend((patch[inside] + [left, top]).tolist())
    return pairs


def ground_candidate(state, selected, image, valid, surface, matcher, directory):
    mask = exclude_region(valid, state['pixels'][selected])
    query = directory / 'ground-query.png'
    cv2.imwrite(str(query), matching_pixels(image, mask, matcher))
    cv2.imwrite(str(directory / 'ground-mask.png'), mask)
    lat, lon, _ = surface.geodetic(state['eye'])
    up = surface.basis.T @ basis(lat, lon)[:, 2]
    vector = cv2.Rodrigues(np.diag([1, -1, -1]) @ state['rotation'].T)[0]
    records = []
    for height in [0, 10, 20]:
        eye = state['eye'] + height * up
        color, depth, world = surface.render(eye, state['rotation'], state['K'], tuple(state['size']))
        projected = cv2.projectPoints(state['points'][selected] - eye, vector,
                                      np.zeros(3), state['K'], None)[0].reshape(-1, 2)
        reference_mask = exclude_region((depth > 0).astype(np.uint8), projected)
        reference = directory / f'ground-reference-{height}.png'
        cv2.imwrite(str(reference), matching_pixels(color, reference_mask, matcher))
        for scope in ['full', 'quadrants']:
            pairs = (matcher.match(reference, query)['pairs'] if scope == 'full' else
                     ground_patch_pairs(image, color, mask, reference_mask, matcher, directory, height))
            # Mask both endpoints; a blacked-out roof must never provide ground evidence.
            points, pixels = ground_pairs(pairs, depth * (reference_mask > 0), world, mask)
            result = fit(points, pixels, state['K'], tuple(state['size']))
            record = {'camera_height_offset_m': height, 'scope': scope, 'pairs': len(points),
                      'score': result[2] if result else None}
            records.append(record)
            write_json(directory / 'ground-search.json', records)
            print(json.dumps({'phase': 'ground', **record}), flush=True)
            if result and result[2]['geometry_accepted']:
                eye, rotation, _, inliers = result
                np.savez_compressed(directory / 'ground.npz', eye=eye, rotation=rotation,
                                    points=points, pixels=pixels, inliers=inliers)
                return eye, rotation, points[inliers], pixels[inliers]
    return None


def fixed_verification(pairs, valid, region):
    """Check elevated and ground pixels using the rendered surface footprint."""
    pairs = np.asarray(pairs, float).reshape(-1, 2, 2)
    xy = pairs.round().astype(int)
    h, w = valid.shape
    inside = ((xy >= 0).all(axis=(1, 2)) & (xy[:, :, 0] < w).all(axis=1)
              & (xy[:, :, 1] < h).all(axis=1))
    pairs, xy = pairs[inside], xy[inside]
    keep = valid[xy[:, 1, 1], xy[:, 1, 0]] > 0
    pairs, xy = pairs[keep], xy[keep]
    error = np.linalg.norm(pairs[:, 0] - pairs[:, 1], axis=1)
    elevated = region[xy[:, 0, 1], xy[:, 0, 0]] > 0
    groups = []
    for selected in [elevated, ~elevated]:
        groups.append({'pairs': int(selected.sum()),
            'median_px': float(np.median(error[selected])) if selected.any() else None,
            'inlier_fraction': float(np.mean(error[selected] < 4)) if selected.any() else 0})
    inliers = error < 4
    coverage = (float(cv2.contourArea(cv2.convexHull(pairs[inliers, 1].astype(np.float32))) / (w * h))
                if inliers.sum() >= 3 else 0)
    passed = (coverage >= .05 and all(group['pairs'] >= 30 and group['inlier_fraction'] >= .6
                                      and group['median_px'] <= 4 for group in groups))
    return {'pairs': len(pairs), 'inliers': int(inliers.sum()), 'image_support_fraction': coverage,
            'regions': groups, 'passed': bool(passed), 'pose_refitted': False}, pairs, error


def load_seed(run, case_id, allow_unverified=False):
    """Keep failed flat-terrain fits available only as explicit research seeds."""
    path = run / case_id / 'candidate.npz'
    score = None
    if not path.exists():
        if not allow_unverified:
            raise ValueError('no accepted seed; --unverified-seed permits research from failed terrain fits')
        result = json.loads((run / f'{case_id}.json').read_text())
        attempts = [a for a in result['attempts'] if a.get('prior_gate')
                    and a.get('inliers', 0) >= 60 and a.get('positive_depth_fraction') == 1
                    and a.get('image_support_fraction', 0) >= .03
                    and np.isfinite(a.get('holdout_median_px', float('inf')))]
        if not attempts:
            raise ValueError('no plausible unverified relief seed')
        score = max(attempts, key=lambda a: a['inliers'] * a['image_support_fraction'] /
                    (1 + a['holdout_median_px']))
        path = run / case_id / f"evidence-{score['seed']:02}-{score['step']:02}.npz"
    with np.load(path) as saved:
        state = dict(saved)
    if score is not None:
        state['K'] = state['input_K'].copy()
        state['K'][0, 0] = state['K'][1, 1] = state['focal_px']
        state['size'] = np.asarray(score['resolution'])
    return state, {'seed_sha256': digest(path), 'seed_file': path.name,
                   'seed_is_unverified': score is not None, 'seed_score': score}


def propose(case, root, state, surface, matcher, directory, catalog_hash):
    image, valid, size = prepared(case, root)
    if tuple(state['size']) != size:
        raise ValueError('relief seed must use the matching image dimensions')
    labels_path = root / case['semantic_labels']
    if digest(labels_path) != case['semantic_sha256']:
        raise ValueError('semantic labels changed')
    labels = cv2.resize(cv2.imread(str(labels_path), 0), size, interpolation=cv2.INTER_NEAREST)
    region = dominant_region(state, labels)
    if region is None:
        return {'status': 'no_dominant_region'}
    selected, _, label = region
    ground = ground_candidate(state, selected, image, valid, surface, matcher, directory)
    if ground is None:
        return {'status': 'no_independent_ground'}
    eye, rotation, points, pixels = ground
    lat, lon, _ = surface.geodetic(eye)
    up = surface.basis.T @ basis(lat, lon)[:, 2]
    result = fit_relief([(state['points'][selected], state['pixels'][selected]), (points, pixels)],
                        eye, rotation, state['K'], size, up)
    if result is None:
        return {'status': 'insufficient_joint_geometry'}
    report = {key: value for key, value in result.items() if key not in ('eye', 'rotation')}
    report.update(status='rejected_joint_geometry', dominant_class=label)
    if not result['optimizer_converged'] or any(score['held_median_px'] > 4
            or score['held_inlier_fraction'] < .6 or score['positive_depth_fraction'] != 1
            for score in result['scores']):
        return report
    segmented = region_mask(surface, state['points'][selected])
    if segmented is None:
        return {**report, 'status': 'no_bounded_surface_mask'}
    correction = save_correction(directory, *segmented, result['height_offset_m'], catalog_hash)
    corrected = ReliefSurface(surface, correction, catalog_hash)
    color, depth, world = corrected.render(result['eye'], result['rotation'], state['K'], size)
    reference, query = directory / 'reference.png', directory / 'query.png'
    cv2.imwrite(str(reference), matching_pixels(color, (depth > 0).astype(np.uint8), matcher))
    cv2.imwrite(str(query), matching_pixels(image, valid, matcher))
    response = matcher.match(reference, query)
    latitude, longitude, _ = corrected.geodetic(world)
    footprint = corrected.region(latitude, longitude) & (depth > 0)
    verification, pairs, errors = fixed_verification(response['pairs'], valid, footprint)
    np.savez_compressed(directory / 'fixed-matches.npz', pairs=pairs, errors=errors)
    np.savez_compressed(directory / 'proposal.npz', eye=result['eye'], rotation=result['rotation'],
                        K=state['K'], size=size)
    cv2.imwrite(str(directory / 'comparison.jpg'), np.concatenate(
        [image, color, cv2.addWeighted(image, .5, color, .5, 0)], axis=1))
    return {**report, 'status': 'research_proposal' if verification['passed'] else 'rejected_fixed_render',
        'fixed_verification': verification, 'surface_correction_sha256': digest(correction),
        'position': list(map(float, surface.geodetic(result['eye']))),
        'requires_render_verification': not verification['passed'], 'requires_visual_review': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--unverified-seed', action='store_true',
                        help='allow a plausible failed terrain fit as a research seed')
    parser.add_argument('--roma-model', type=Path, required=True)
    parser.add_argument('--roma-encoder', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    args = parser.parse_args()
    cv2.setNumThreads(4)
    from photo_roma import RomaMatcher, configuration
    profile = json.loads(args.profile.read_text())
    case = next(case for case in profile['cases'] if case['id'] == args.case)
    root = (args.profile.parent / profile['root']).resolve()
    catalog_hash = digest(args.catalog)
    if json.loads((args.run / 'run.json').read_text())['catalog_sha256'] != catalog_hash:
        parser.error('relief seed and surface must use the same catalog frame')
    state, seed_info = load_seed(args.run, args.case, args.unverified_seed)
    config = {'catalog_sha256': catalog_hash, 'profile_sha256': digest(args.profile),
              **seed_info, 'case': args.case,
              'opencv': cv2.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
              'code': {p.name: digest(p) for p in Path(__file__).parent.glob('photo_*.py')},
              **configuration(args.roma_model, args.roma_encoder, args.device, True)}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'run.json', config)
    surface = Surface(args.catalog)
    matcher = RomaMatcher(args.roma_model, args.roma_encoder, args.device, True)
    try:
        result = propose(case, root, state, surface, matcher, args.output, catalog_hash)
        write_json(args.output / 'report.json', result)
        print(json.dumps(result), flush=True)
    finally:
        matcher.close()


if __name__ == '__main__':
    main()
