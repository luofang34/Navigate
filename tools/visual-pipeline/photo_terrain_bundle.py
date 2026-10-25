"""Recompute a research terrain bundle from same-camera photos and saved map anchors."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import scipy

from photo_bundle_fit import solve
from photo_bundle_pairs import match_edges
from photo_match import Matcher
from photo_surface import ContextSurface, Surface, frame, reframe_camera
from photo_testset import digest, write_json
from photo_tracking import load_anchors, prepared


def inputs(args):
    profile = json.loads(args.profile.read_text())
    if profile['profile'] != 'with-prior':
        raise ValueError('terrain bundles require an explicit same-camera prior profile')
    root = (args.profile.parent / profile['root']).resolve()
    cases = {case['id']: case for case in profile['cases']}
    if len(args.case) < 2 or len(set(args.case)) != len(args.case):
        raise ValueError('terrain bundles require at least two distinct ordered cases')
    selected = [cases[identifier] for identifier in args.case]
    calibration = lambda c: tuple(c['prior'].get(k) for k in ['camera_model', 'lens_model', 'focal_35mm_mm'])
    focal = calibration(selected[0])[-1]
    if (focal is None or not math.isfinite(focal) or focal <= 0 or
            any(calibration(c) != calibration(selected[0]) for c in selected)):
        raise ValueError('shared bundle calibration requires matching camera, lens and focal claims')
    manifest_path = args.profile.parent / profile['manifest'] if profile.get('manifest') else root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if not any(set(args.case).issubset(flight['photos']) for flight in manifest['flights']):
        raise ValueError('bundle cases must belong to one declared flight')
    anchors = load_anchors(args.seed_run, digest(args.catalog), max_depth=4,
                           catalog_paths=args.source_catalog)
    destination = frame(json.loads(args.catalog.read_text()))
    views = []; hashes = {str(args.profile): digest(args.profile), str(manifest_path): digest(manifest_path)}
    for case in selected:
        anchor = anchors[case['id']]
        with np.load(anchor['path']) as state:
            view = {key: state[key].copy() for key in state.files}
        if anchor['source_frame'] is not None:
            source_basis, source_origin = anchor['source_frame']
            destination_basis, destination_origin = destination
            view['eye'], view['rotation'] = reframe_camera(
                view['eye'], view['rotation'], anchor['source_frame'], destination)
            view['points'] = (view['points'] @ source_basis.T + source_origin -
                              destination_origin) @ destination_basis
        _, _, size = prepared(case, root)
        if tuple(view['size']) != size:
            raise ValueError(f"seed resolution differs from matching photo: {case['id']}")
        view.update(prior=case['prior'], direct_map_anchor=anchor['depth'] == 0)
        views.append(view)
        for path in [anchor['path'], anchor['report'], root / case['image'], root / case['valid_mask']]:
            hashes[str(path)] = digest(path)
    for run in args.seed_run:
        hashes[str(run / 'run.json')] = digest(run / 'run.json')
    for catalog in args.source_catalog:
        hashes[str(catalog)] = digest(catalog)
    return root, selected, views, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--context-catalog', type=Path)
    parser.add_argument('--seed-run', type=Path, action='append', required=True)
    parser.add_argument('--source-catalog', type=Path, action='append', default=[],
                        help='explicit coordinate frame for seeds from another catalog; repeatable')
    parser.add_argument('--case', action='append', required=True)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--exclude-bright-clouds', action='store_true',
                        help='research heuristic: exclude bright low-chroma cores; unsuitable for snow')
    args = parser.parse_args(); cv2.setNumThreads(4)
    root, cases, views, hashes = inputs(args)
    for path in [args.catalog, args.worker, args.model, args.runtime, *([args.context_catalog] if args.context_catalog else [])]:
        hashes[str(path)] = digest(path)
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {'inputs': hashes, 'cases': args.case, 'profile_sha256': digest(args.profile),
        'catalog_sha256': digest(args.catalog),
        'context_catalog_sha256': digest(args.context_catalog) if args.context_catalog else None,
        'code': {p.name: digest(p) for p in Path(__file__).parent.glob('photo_*.py')},
        'opencv': cv2.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
        'exclude_bright_clouds': args.exclude_bright_clouds,
        'flow': {'long_edge': 1600, 'cycle_max_native_px': .5, 'change_max_solver_px': 3,
                 'patch_error_max': 20, 'pixel_center_mapping': '(pixel + 0.5) * scale - 0.5'}}
    write_json(args.output / 'run.json', provenance)
    matcher = Matcher(args.worker, args.model, args.runtime)
    try:
        edges, matched = match_edges(cases, root, args.output, matcher, args.exclude_bright_clouds)
    finally:
        matcher.close()
    surface = Surface(args.catalog)
    if args.context_catalog:
        surface = ContextSurface(surface, Surface(args.context_catalog))
    cameras, report = solve(views, edges, surface)
    if any(digest(Path(path)) != value for path, value in hashes.items()):
        raise ValueError('bundle inputs changed during reconstruction')
    report.update(provenance); report['matching'] = matched
    for i, (eye, _, rotation, K) in enumerate(cameras):
        np.savez_compressed(args.output / f'camera-{i}.npz', eye=eye,
                            rotation=rotation.T @ np.diag([1, -1, -1]), K=K, size=views[i]['size'])
    write_json(args.output / 'report.json', report)
    for i, case in enumerate(cases):
        camera = args.output / f'camera-{i}.npz'
        write_json(args.output / (case['id'] + '.json'), {'id': case['id'], 'status': report['status'],
            'geometry_accepted': False, 'requires_map_verification': True, 'camera': camera.name,
            'camera_sha256': digest(camera), 'report_sha256': digest(args.output / 'report.json')})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
