"""Joint camera proposals using a training lake and a separate map landmark check."""
import cv2
import numpy as np

from photo_pair_depth import clearance, forest_check, forest_pairs, forest_training_pairs
from photo_pose import spatial_holdout
from photo_water_landmarks import landmark_residual, map_regions, pair_regions, reflective_regions


def canopy_residual(cameras, pairs, surface):
    heights, valid = clearance(cameras, pairs, surface)
    # The validation tolerance allows DEM error; it is not a target below ground.
    return np.r_[np.where(valid, np.minimum(heights, 0) / 5, 1000),
                 np.where(valid, np.maximum(heights - 120, 0) / 10, 1000)]


def solve_landmarks(data, pairs, homography, surface, output, fixed_focal=False):
    from photo_water_pair import camera, epipolar, optimize, solve, view_residual
    from photo_testset import digest, write_json

    _, initial = solve(data, pairs, surface, fixed_focal)
    calibrated, _ = solve(data, pairs, surface, fixed_focal=True)
    observations, pairing = pair_regions([reflective_regions(d) for d in data], homography)
    targets = np.array([item['reference_center'] for item in pairing])
    polygons, associations = map_regions(surface, calibrated[0], data[0]['size'], targets,
                                         [len(region) for region in observations[0]])
    artifact = output / 'landmarks.npz'
    np.savez_compressed(artifact, **{f'world_{i}': p for i, p in enumerate(polygons)},
        **{f'pixels_{i}_{j}': p for i, view in enumerate(observations) for j, p in enumerate(view)})
    write_json(output / 'landmarks.json', {'photo_pairs': pairing, 'map_associations': associations,
        'training_component': 0, 'held_component': 1,
        'selection_uses_held_fit_error': False, 'artifact_sha256': digest(artifact)})
    held = spatial_holdout(pairs[:, 0], data[0]['size'])
    raw = pairs[~held]; cv2.setRNGSeed(0)
    _, consensus = cv2.findFundamentalMat(raw[:, 0], raw[:, 1], cv2.USAC_MAGSAC, 1., .999, 10000)
    if consensus is None or consensus.sum() < 60:
        raise ValueError('photo-pair training geometry did not form a consensus')
    supported = raw[consensus.ravel() > 0]
    train, forest = supported[::4], forest_training_pairs(data, supported)
    if len(forest) < 30:
        raise ValueError('distant water fitting requires independent forest depth support')
    cameras = lambda x: [camera(d, x[i*6:i*6+6], x[12]) for i, d in enumerate(data)]
    x0 = np.array(initial['x'])
    outline_count = sum(len(d['shore_world']) for d in data)
    depth_weight = 4 * np.sqrt(outline_count / len(forest))
    far_weight = np.sqrt(outline_count / len(landmark_residual(cameras(x0), observations, polygons)))
    def residual(x):
        poses = cameras(x)
        return np.concatenate([*[view_residual(d, x[i*6:i*6+6], x[12]) for i, d in enumerate(data)],
            epipolar(poses, train) * 2, [x[12] / .15 * 10],
            depth_weight * canopy_residual(poses, forest, surface),
            far_weight * landmark_residual(poses, observations, polygons)])
    result = optimize(residual, x0, 2, fixed_focal)
    poses = cameras(result.x)
    distant = landmark_residual(poses, observations, polygons, (1,))
    error = abs(epipolar(poses, pairs[held]))
    report = {**initial, 'status': 'research_proposal' if result.success else 'optimization_incomplete',
        'optimizer_converged': bool(result.success), 'x': result.x.tolist(),
        'focal_factor': float(np.exp(result.x[12])), 'geometry_accepted': False,
        'forest_clearance': {**forest_check(poses, pairs[held & forest_pairs(data, pairs)], surface),
            'training_pairs': len(forest), 'constraint_applied': True,
            'optimization_lower_bound_m': 0, 'weight_multiplier': 4},
        'held_epipolar_median_px': float(np.median(error)),
        'held_epipolar_inlier_fraction': float(np.mean(error < 2)),
        'distant_water': {'photo_pairs': pairing, 'map_associations': associations,
            'training_component': 0, 'held_component': 1, 'held_pixels': len(distant),
            'held_median_px': float(np.median(distant)), 'held_p90_px': float(np.quantile(distant, .9)),
            'held_inlier_fraction': float(np.mean(distant <= 4)),
            'footprint': 'concave shoreline, 1.5 orthophoto-pixel simplification',
            'artifact_sha256': digest(artifact), 'absolute_pose_verified': False}}
    return poses, report
