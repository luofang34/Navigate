"""Recover camera hypotheses from retrieved photographs and verify against terrain."""
import math

import cv2
import numpy as np

from photo_pose import fit, ground_pairs
from photo_render_check import verify_pose, merit
from photo_testset import digest, write_json
from photo_tracking import prepared, matching_pixels, match_pair


def terrain_gate(surface, eye):
    lat, lon, altitude = surface.geodetic(eye)
    ground, covered = surface.sample('dem', lat, lon)
    return bool(covered and 30 < altitude - ground < 6000), {
        'latitude': float(lat), 'longitude': float(lon), 'altitude_m': float(altitude)}


def photo_hypotheses(case, mask, size, query, directory, references, reference_root,
                     anchors, ranked, surface, source_surface, matcher, focal_grid, progress=None):
    attempts, seeds = [], []
    for index, retrieved in enumerate(ranked):
        identifier = retrieved['anchor']
        anchor = anchors[identifier]
        reference, valid, ref_size = prepared(references[identifier], reference_root)
        source = source_surface(anchor['catalog_sha256'])
        with np.load(anchor['path']) as state:
            _, depth, world = source.render(state['eye'], state['rotation'], state['K'], ref_size)
        depth[valid == 0] = 0
        path = directory / f'anchor-{index:02}.png'
        if not cv2.imwrite(str(path), matching_pixels(reference, depth > 0, matcher)):
            raise OSError(f'cannot write masked reference: {path}')
        pairs = match_pair(matcher, path, query, directory)
        points, pixels = ground_pairs(pairs, depth, world, mask)
        if source is not surface:
            points = (points @ source.basis.T + source.origin - surface.origin) @ surface.basis
        np.savez_compressed(directory / f'anchor-{index:02}.npz', pairs=pairs, points=points, pixels=pixels)
        for equivalent in focal_grid:
            focal = equivalent * math.hypot(*size) / math.hypot(36, 24)
            intrinsics = np.array([[focal, 0, (size[0] - 1) / 2],
                                   [0, focal, (size[1] - 1) / 2], [0, 0, 1.]])
            result = fit(points, pixels, intrinsics, size)
            record = {**retrieved, 'focal_35mm_mm': equivalent, 'pairs': len(points)}
            if result is None:
                attempts.append({**record, 'status': 'insufficient_geometry'})
                continue
            eye, rotation, score, inliers = result
            gate, location = terrain_gate(surface, eye)
            record.update(score=score, terrain_gate=gate, **location)
            attempts.append(record)
            if not gate or not score['geometry_accepted']:
                continue
            filename = f'seed-{index:02}-{equivalent:g}.npz'
            np.savez_compressed(directory / filename, eye=eye, rotation=rotation, K=intrinsics,
                                size=size, points=points, pixels=pixels, inliers=inliers)
            seeds.append({**record, 'file': filename,
                          'anchor_pose_sha256': digest(anchor['path']),
                          'anchor_report_sha256': digest(anchor['report'])})
        if progress is not None:
            progress({'phase': 'photo-anchor', 'anchor': identifier, 'pairs': len(points), 'seeds': len(seeds)})
    write_json(directory / 'acquisition.json', {'id': case['id'], 'attempts': attempts, 'seeds': seeds})
    return seeds


def focal_beam(seeds):
    """Keep a representative of every focal hypothesis until terrain can distinguish it."""
    chosen = {}
    for seed in seeds:
        score = seed['score']
        rank = score['inliers'] * score['image_support_fraction'] / (1 + score['holdout_median_px'])
        focal = seed['focal_35mm_mm']
        if focal not in chosen or rank > chosen[focal][0]:
            chosen[focal] = (rank, seed)
    return [chosen[focal][1] for focal in sorted(chosen)]


def refine_hypothesis(seed, directory, surface, matcher, query, mask, steps, prefix, progress=None):
    with np.load(directory / seed['file']) as state:
        eye, rotation, intrinsics = state['eye'].copy(), state['rotation'].copy(), state['K'].copy()
        size = tuple(int(x) for x in state['size'])
    focal = intrinsics[0, 0]
    records, candidates = [], []
    for step in range(steps):
        destination = directory / f'{prefix}-{step:02}'
        destination.mkdir(exist_ok=True)
        color, depth, world = surface.render(eye, rotation, intrinsics, size)
        reference = destination / 'input.png'
        if not cv2.imwrite(str(reference), matching_pixels(color, depth > 0, matcher)):
            raise OSError(f'cannot write terrain reference: {reference}')
        response = matcher.match(reference, query)
        points, pixels = ground_pairs(response['pairs'], depth, world, mask)
        np.savez_compressed(destination / 'input-matches.npz', points=points, pixels=pixels,
                            eye=eye, rotation=rotation, K=intrinsics, size=size)
        result = fit(points, pixels, intrinsics, size, (focal * .85, focal * 1.15))
        if result is None:
            records.append({'step': step, 'status': 'insufficient_geometry'})
            break
        eye, rotation, score, inliers = result
        intrinsics[0, 0] = intrinsics[1, 1] = score['focal_px']
        gate, location = terrain_gate(surface, eye)
        record = {'step': step, 'score': score, 'terrain_gate': gate, **location}
        records.append(record)
        if not gate:
            break
        if not score['geometry_accepted']:
            continue
        check = verify_pose(surface, eye, rotation, intrinsics, size, matcher, query, mask, destination)
        record['fixed_render'] = check
        if progress is not None:
            progress({'phase': prefix, **record})
        path = destination / 'pose.npz'
        np.savez_compressed(path, eye=eye, rotation=rotation, K=intrinsics, size=size,
                            points=points, pixels=pixels, inliers=inliers)
        rank = merit(check)
        if rank is not None:
            candidates.append({**record, 'merit': rank, 'file': str(path.relative_to(directory)),
                               'anchor': seed['anchor'], 'seed_file': seed.get('seed_file', seed['file'])})
    return records, candidates


def acquire(case, root, directory, references, reference_root, anchors, ranked,
            surface, source_surface, matcher, focal_grid, coarse_steps, refine_steps, progress=None):
    directory.mkdir(parents=True, exist_ok=True)
    image, mask, size = prepared(case, root)
    query = directory / 'query.png'
    if not cv2.imwrite(str(query), matching_pixels(image, mask, matcher)):
        raise OSError(f'cannot write masked query: {query}')
    seeds = photo_hypotheses(case, mask, size, query, directory, references, reference_root,
                             anchors, ranked, surface, source_surface, matcher, focal_grid, progress)
    records, candidates = [], []
    for index, seed in enumerate(focal_beam(seeds)):
        attempts, accepted = refine_hypothesis(seed, directory, surface, matcher, query, mask,
                                               coarse_steps, f'hypothesis-{index:02}', progress)
        records.append({'seed': seed, 'attempts': attempts})
        candidates.extend(accepted)
    if candidates and refine_steps:
        best = max(candidates, key=lambda item: item['merit'])
        attempts, accepted = refine_hypothesis(best, directory, surface, matcher, query, mask,
                                               refine_steps, 'polish', progress)
        records.append({'seed': best, 'attempts': attempts})
        candidates.extend(accepted)
    candidate = None
    if candidates:
        best = max(candidates, key=lambda item: item['merit'])
        state = directory / 'candidate.npz'
        state.write_bytes((directory / best['file']).read_bytes())
        with np.load(state) as saved:
            color, _, _ = surface.render(saved['eye'], saved['rotation'], saved['K'], size)
        if not cv2.imwrite(str(directory / 'comparison.jpg'),
                           np.concatenate([image, color, cv2.addWeighted(image, .5, color, .5, 0)], axis=1)):
            raise OSError(f'cannot write acquisition comparison: {directory}')
        anchor = anchors[best['anchor']]
        lineage = {'id': best['anchor'], 'pose_sha256': digest(anchor['path']),
                   'report_sha256': digest(anchor['report'])}
        candidate = {**best, 'evidence_kind': 'terrain_refinement', 'state_sha256': digest(state),
                     'anchor_state_sha256': lineage['pose_sha256'],
                     'anchor_report_sha256': lineage['report_sha256'],
                     'anchor_lineage': [lineage, *anchor.get('lineage', [])]}
    return {'id': case['id'], 'status': 'candidate_requires_review' if candidate else 'no_candidate',
            'candidate': candidate, 'query_metadata_used': False,
            'retrieval': ranked, 'attempts': records}
