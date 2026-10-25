"""Locate metadata-free photos using a frozen photo bank and regional terrain."""
import argparse
from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path

import cv2
import numpy as np

from photo_acquisition import acquire
from photo_bank import read_profile, flight_partition, rank_references, DinoFeatures, cached_features
from photo_runs import resolved_inputs
from photo_surface import Surface
from photo_testset import digest, write_json
from photo_tracking import load_anchors


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference-profile', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--anchor-run', type=Path, action='append', required=True)
    parser.add_argument('--anchor-catalog', type=Path, action='append', default=[])
    parser.add_argument('--exclude-flight-manifest', type=Path,
                        help='benchmark partition covering all query and reference IDs')
    parser.add_argument('--roma-model', type=Path, required=True)
    parser.add_argument('--roma-encoder', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--case', action='append')
    parser.add_argument('--anchors-per-case', type=int, default=6)
    parser.add_argument('--coarse-steps', type=int, default=2)
    parser.add_argument('--refine-steps', type=int, default=3)
    parser.add_argument('--focal-grid', nargs='+', type=float,
                        default=[14, 20, 26, 35, 52, 65, 75, 85, 100, 120, 139])
    args = parser.parse_args()
    if args.anchors_per_case < 1 or args.coarse_steps < 1 or args.refine_steps < 0:
        parser.error('anchor and coarse-step counts must be positive; refine steps cannot be negative')
    if (not np.isfinite(args.focal_grid).all() or min(args.focal_grid) <= 0
            or len(set(args.focal_grid)) != len(args.focal_grid)):
        parser.error('focal hypotheses must be distinct positive finite 35mm-equivalent values')
    return args


def configuration(args, catalog_hash, backend):
    _, parents = resolved_inputs(args.anchor_run)
    return {'profile_sha256': digest(args.profile), 'catalog_sha256': catalog_hash,
            'reference_profile_sha256': digest(args.reference_profile), 'anchor_runs': parents,
            'anchor_catalogs': {str(p.resolve()): digest(p) for p in args.anchor_catalog},
            'flight_partition_sha256': digest(args.exclude_flight_manifest) if args.exclude_flight_manifest else None,
            'query_metadata_used': False, 'reference_scope': 'supplied regional terrain and photo bank',
            'backend': backend, 'opencv': cv2.__version__, 'numpy': np.__version__,
            'coarse_steps': args.coarse_steps, 'refine_steps': args.refine_steps,
            'anchors_per_case': args.anchors_per_case, 'focal_grid': args.focal_grid,
            'cases': sorted(args.case) if args.case else None,
            'code': {p.name: digest(p) for p in Path(__file__).parent.glob('photo_*.py')}}


def retrieve_cases(args, cases, root, references, reference_root, anchors, flights, output, backend):
    features = {}
    rankings = {}
    cache = output / 'features'
    cache.mkdir(exist_ok=True)
    identity = {'encoder_sha256': backend['encoder_sha256'], 'device': args.device,
                'torch': backend['torch'], 'romatch': importlib.metadata.version('romatch'),
                'code': digest(Path(__file__).with_name('photo_bank.py')),
                'opencv': cv2.__version__, 'numpy': np.__version__}
    extractor = DinoFeatures(args.roma_encoder, args.device)
    try:
        for identifier in sorted(anchors):
            features[identifier] = cached_features(references[identifier], reference_root, extractor, cache, identity)
        for identifier, case in cases.items():
            query = cached_features(case, root, extractor, cache, identity)
            rankings[identifier] = rank_references(case, query, references, features, flights,
                                                   anchors)[:args.anchors_per_case]
    finally:
        extractor.close()
    write_json(output / 'retrieval.json', rankings)
    return rankings


def main():
    from photo_roma import RomaMatcher, configuration as roma_configuration
    args = arguments()
    cv2.setNumThreads(4)
    cases, root = read_profile(args.profile, blind=True)
    references, reference_root = read_profile(args.reference_profile)
    if args.case:
        if set(args.case) - cases.keys():
            raise ValueError('requested query IDs are absent from the no-prior profile')
        cases = {q: c for q, c in cases.items() if q in args.case}
    flights = flight_partition(args.exclude_flight_manifest)
    catalog_hash = digest(args.catalog)
    catalog_paths = {digest(p): p for p in [args.catalog, *args.anchor_catalog]}
    anchors = load_anchors(args.anchor_run, catalog_hash, catalog_paths=catalog_paths.values())
    anchors = {q: a for q, a in anchors.items() if q in references}
    if not anchors:
        raise ValueError('photo bank has no direct terrain candidates in the reference profile')
    if flights is not None and any(q not in flights for q in [*cases, *anchors]):
        raise ValueError('flight exclusion requires a group for every query and reference')
    backend = roma_configuration(args.roma_model, args.roma_encoder, args.device)
    config = configuration(args, catalog_hash, backend)
    output = args.output.resolve()
    if output in {root, reference_root}:
        raise ValueError('acquisition output must differ from the input datasets')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run.json').exists() and json.loads((output / 'run.json').read_text()) != config:
        raise ValueError('acquisition inputs changed; choose a new output directory')
    write_json(output / 'run.json', config)
    rankings = retrieve_cases(args, cases, root, references, reference_root, anchors, flights, output, backend)
    surface = Surface(args.catalog)

    @lru_cache(maxsize=1)
    def source_surface(key):
        return surface if key == catalog_hash else Surface(catalog_paths[key])

    matcher = RomaMatcher(args.roma_model, args.roma_encoder, args.device)
    try:
        for identifier, case in cases.items():
            path = output / (identifier + '.json')
            if path.exists():
                saved = json.loads(path.read_text())
                if saved.get('candidate') and digest(output / identifier / 'candidate.npz') != saved['candidate']['state_sha256']:
                    raise ValueError(f'acquisition camera changed: {path}')
                continue
            result = acquire(case, root, output / identifier, references, reference_root, anchors,
                             rankings[identifier], surface, source_surface, matcher, args.focal_grid,
                             args.coarse_steps, args.refine_steps,
                             lambda event: print(json.dumps({'id': identifier, **event}), flush=True))
            result['excluded_flight'] = flights[identifier] if flights is not None else None
            write_json(path, result)
            print(json.dumps({'id': identifier, 'status': result['status'],
                              'fixed_render': result['candidate']['fixed_render'] if result['candidate'] else None}), flush=True)
    finally:
        matcher.close()


if __name__ == '__main__':
    main()
