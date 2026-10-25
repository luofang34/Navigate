"""Generate terrain-view seeds from photo priors and an overlapping photo pair."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from photo_direction import orientation_proposal
from photo_match import Matcher, match_pixels
from photo_surface import Surface, basis, horizon_attitude
from photo_testset import digest, write_json
from photo_tracking import match_pair, prepared


def prepare(case, root, surface):
    image, valid, size = prepared(case, root)
    prior = case['prior']
    position = prior.get('position')
    if position is None:
        raise ValueError('pair seeding requires a position prior for each photo')
    path = root / case['semantic_labels']
    if digest(path) != case['semantic_sha256']:
        raise ValueError('pair semantic labels changed')
    labels = cv2.resize(cv2.imread(str(path), 0), size, interpolation=cv2.INTER_NEAREST)
    focal = (prior.get('focal_35mm_mm') or 26) * math.hypot(*size) / math.hypot(36, 24)
    K = np.array([[focal, 0, (size[0] - 1) / 2], [0, focal, (size[1] - 1) / 2], [0, 0, 1.]])
    horizon = horizon_attitude(labels, K)
    if horizon is None:
        raise ValueError('pair seeding requires a supported horizon estimate')
    lat, lon = position['latitude'], position['longitude']
    ground, covered = surface.sample('dem', lat, lon)
    if not covered:
        raise ValueError('pair position is outside the terrain catalog')
    altitude = position.get('altitude_m')
    eye = surface.local(lat, lon, altitude if altitude is not None else float(ground) + 1000)
    return {'image': image, 'valid': valid, 'labels': labels, 'size': size, 'K': K,
            'eye': eye, 'horizon': horizon, 'latitude': lat, 'longitude': lon}


def generate(cases, root, surface, matcher, output):
    data = [prepare(case, root, surface) for case in cases]
    direction = cases[0]['prior']
    if direction.get('direction_reference') != 'T' or direction.get('direction_deg') is None:
        raise ValueError('the first pair photo requires a true-north heading prior')
    first = data[0]
    rotation = surface.camera_rotation(first['latitude'], first['longitude'], direction['direction_deg'],
        first['horizon']['down_deg'], first['horizon']['roll_deg'])
    paths = []
    for i, item in enumerate(data):
        path = output / f'query-{i}.png'
        cv2.imwrite(str(path), match_pixels(item['image'], item['valid'], True))
        paths.append(path)
    if data[0]['size'] == data[1]['size']:
        pairs = matcher.match(*paths, refinement=True)['pairs']
    else:
        pairs = match_pair(matcher, *paths, output)
    pairs = np.asarray(pairs, float).reshape(-1, 2, 2)
    pairs = pairs[np.isfinite(pairs).all(axis=(1, 2))]
    good = np.ones(len(pairs), bool)
    for i, item in enumerate(data):
        xy = pairs[:, i].round().astype(int)
        w, h = item['size']
        inside = ((xy >= 0).all(axis=1) & (xy[:, 0] < w) & (xy[:, 1] < h))
        indices = np.flatnonzero(inside)
        inside[indices] &= item['valid'][xy[indices, 1], xy[indices, 0]] > 0
        good &= inside
    pairs = pairs[good]
    result = orientation_proposal(pairs, data[0]['K'], data[1]['K'], rotation,
                                  data[0]['size'], data[1]['size'])
    if result is None:
        return {'status': 'no_orientation_consensus', 'pairs': len(pairs)}
    np.savez_compressed(output / 'pairs.npz', pairs=pairs, inliers=result['inliers'],
                        homography=result['transform'])
    report = {'status': 'orientation_proposal', 'requires_map_verification': True,
              'geometry_accepted': False, 'score': result['score'], 'views': []}
    for i, (case, item) in enumerate(zip(cases, data)):
        lat, lon = item['latitude'], item['longitude']
        forward = basis(lat, lon).T @ surface.basis @ (-(rotation if i == 0 else result['rotation'])[:, 2])
        heading = math.degrees(math.atan2(forward[0], forward[1])) % 360
        # A planar warp absorbs translation into pitch. Retain the independent
        # horizon pitch and roll while transferring only its heading proposal.
        pose = surface.camera_rotation(lat, lon, heading, item['horizon']['down_deg'],
                                       item['horizon']['roll_deg'])
        color, depth, world = surface.render(item['eye'], pose, item['K'], item['size'])
        np.savez_compressed(output / f'seed-{i}.npz', eye=item['eye'], rotation=pose,
                            K=item['K'], size=item['size'], color=color, depth=depth, world=world)
        cv2.imwrite(str(output / f'reference-{i}.png'), color)
        report['views'].append({'id': case['id'], 'heading_deg': heading,
            'heading_source': 'source_EXIF' if i == 0 else 'photo_pair_homography_rotation',
            'original_heading_deg': case['prior'].get('direction_deg'), 'horizon': item['horizon']})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--case', action='append', required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--worker', type=Path, default=Path(__file__).resolve().parents[2] / 'target/release/navigate-recording-worker')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    args = parser.parse_args()
    if len(args.case) != 2 or len(set(args.case)) != 2:
        parser.error('supply two different photos in source-to-target order')
    profile = json.loads(args.profile.read_text())
    if profile['profile'] != 'with-prior':
        parser.error('pair seeding requires an explicit prior profile')
    cases = [next(case for case in profile['cases'] if case['id'] == q) for q in args.case]
    root = (args.profile.parent / profile['root']).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'run.json', {'profile_sha256': digest(args.profile),
        'catalog_sha256': digest(args.catalog), 'cases': args.case,
        'worker_sha256': digest(args.worker), 'model_sha256': digest(args.model),
        'runtime_sha256': digest(args.runtime), 'opencv': cv2.__version__, 'numpy': np.__version__,
        'code': {p.name: digest(p) for p in Path(__file__).parent.glob('photo_*.py')}})
    cv2.setNumThreads(4)
    matcher = Matcher(args.worker, args.model, args.runtime)
    try:
        result = generate(cases, root, Surface(args.catalog), matcher, args.output)
        write_json(args.output / 'report.json', result)
        print(json.dumps(result), flush=True)
    finally:
        matcher.close()


if __name__ == '__main__':
    main()
