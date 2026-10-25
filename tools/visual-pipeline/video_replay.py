"""Export video test-set poses as Sokoly replays that play the video over the terrain.

Each clip becomes one replay with an H.264 display copy of the video and a pose
track. Only geometrically accepted candidates become samples. Poses from map
matching are `measured`; poses from photo-to-photo tracking passes are `tracked`.
The viewer interpolates between samples and releases the camera across gaps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from photo_pose import candidate_priority
from photo_preview import map_corners, reconstruction, view, write_image
from photo_surface import Surface
from photo_testset import digest, write_json
from airframe_labels import decode_video
from video_track import (REGISTER_POLICY, FeatureTracker, consistent_anchors, corrected, dense_rotations,
                         residual_shift, smooth_shifts)

SIDC = '10040100001201000003'
# The viewer copies each frame through the CPU; it accepts at most 1280 pixels per side.
MAX_DISPLAY_SIDE = 1280


def accepted(runs):
    """Best accepted candidate per frame, with whether its pass was map matching or tracking."""
    best = {}
    for run in runs:
        configuration = json.loads((run / 'run.json').read_text())
        kind = 'tracked' if 'association' in configuration else 'measured'
        for path in sorted(run.glob('q-*.json')):
            result = json.loads(path.read_text())
            if result['status'] != 'candidate_requires_review' or not result['candidate']['score']['geometry_accepted']:
                continue
            merit = candidate_priority(result['candidate'])
            previous = best.get(result['id'])
            # A map-matched pose outranks a tracked pose of the same frame.
            rank = (kind == 'measured', merit)
            if previous is None or rank > previous[0]:
                best[result['id']] = (rank, run, result, path, configuration['catalog_sha256'], kind)
    return best


def display_copy(source, output, width):
    """Transcode to H.264 without audio; browsers do not play HEVC portably."""
    command = ['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-map', '0:v:0', '-an',
               '-vf', f'scale={width}:-2', '-c:v', 'libx264', '-preset', 'slow', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    subprocess.run(command, check=True)
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                            'stream=width,height,r_frame_rate,nb_frames', '-of', 'json', str(output)],
                           capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)['streams'][0]
    numerator, denominator = map(int, stream['r_frame_rate'].split('/'))
    return [stream['width'], stream['height']], numerator / denominator, int(stream['nb_frames'])


def sample(entry, state, surface, shot, kind, result):
    eye, rotation, K, size = state['eye'], state['rotation'], state['K'], state['size']
    lat, lon, alt = surface.geodetic(eye)
    terrain, covered = surface.sample('dem', lat, lon)
    if not covered:
        raise ValueError(f'candidate camera has no terrain sample: {entry["id"]}')
    width, height = entry['size']
    focal = float(K[0, 0] * width / size[0])
    return {'time_s': entry['video_frame']['time_s'], 'kind': kind, 'shot': shot,
            'view': view(eye, rotation, surface), 'camera_agl_m': float(alt - terrain),
            'check_rms_px': float(result['candidate']['score']['train_rms_px']),
            'camera': {'size': [width, height], 'focal': [focal, focal],
                       'principal': [(width - 1) / 2, (height - 1) / 2]}}


def terrain_shifts(frames, rotations, eyes, K, surface, masks):
    """Per-frame image shift of the terrain rendered at each pose against the video frame."""
    width = REGISTER_POLICY['width']
    scale = width / frames.shape[2]
    small_K = np.array([[K[0, 0] * scale, 0, (width - 1) / 2],
                        [0, K[1, 1] * scale, (round(frames.shape[1] * scale) - 1) / 2], [0, 0, 1.]])
    size = (width, round(frames.shape[1] * scale))
    shifts = []
    for i, frame in enumerate(frames):
        if rotations[i] is None:
            shifts.append(None)
            continue
        render, depth, _ = surface.render(eyes[i], rotations[i], small_K, size)
        ground = cv2.resize(cv2.imread(str(masks / f'{i:06d}-ground.png'), 0), size) > 0
        video = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        shift = residual_shift(video, render, ground & (depth > 0))
        shifts.append(None if shift is None else (shift[0] / scale, shift[1] / scale))
    return shifts


def register_to_terrain(frames, rotations, eyes, K, surface, masks):
    """Turn each frame's camera by its smoothed residual against the rendered terrain.

    The anchors share one focal length and a straight path, which leaves a
    slowly varying offset of a few pixels; per-frame registration removes it.
    """
    smooth = smooth_shifts(terrain_shifts(frames, rotations, eyes, K, surface, masks))
    return [None if r is None else corrected(r, shift, K[0, 0]) for r, shift in zip(rotations, smooth)]


def tracking_mask(masks, index, clouds):
    valid = cv2.imread(str(masks / f'{index:06d}-ground.png'), 0) > 0
    if clouds:
        # Clouds kilometres away barely move in angle over a few seconds, so they constrain rotation
        # where hazy ground has too little texture; they never enter map registration.
        valid |= cv2.imread(str(masks / f'{index:06d}-sky.png'), 0) > 0
    return valid


def dense_samples(clip, anchors, surface, source, masks, shot, size, clouds=False):
    """One sample per video frame: anchors keep their pose, other frames are tracked between them."""
    frames = decode_video(source / clip['video'], 960)
    count, height, width = frames.shape[:3]
    ratio, path, fitted = consistent_anchors({f: a[0] for f, a in anchors.items()}, clip['fps'])
    K = np.array([[ratio * width, 0, (width - 1) / 2], [0, ratio * width, (height - 1) / 2], [0, 0, 1.]])
    tracker = FeatureTracker(K)
    features = [tracker.features(frame, tracking_mask(masks, i, clouds)) for i, frame in enumerate(frames)]
    steps, inliers = [], []
    for i in range(count - 1):
        step, support = tracker.rotation(features[i], features[i + 1])
        steps.append(step)
        inliers.append(support)
    rotations = dense_rotations({f: fitted[f][0] for f in fitted}, steps, count)
    times = np.arange(count) / clip['fps']
    eyes = np.column_stack([np.ones(count), times]) @ path
    rotations = register_to_terrain(frames, rotations, eyes, K, surface, masks)
    samples = []
    for i in range(count):
        if rotations[i] is None:
            continue
        lat, lon, alt = surface.geodetic(eyes[i])
        terrain, covered = surface.sample('dem', lat, lon)
        if not covered:
            continue
        focal = float(ratio * size[0])
        record = {'time_s': round(i / clip['fps'], 4), 'kind': anchors[i][1] if i in anchors else 'tracked',
                  'shot': shot, 'view': view(eyes[i], rotations[i], surface), 'camera_agl_m': float(alt - terrain),
                  'camera': {'size': list(size), 'focal': [focal, focal],
                             'principal': [(size[0] - 1) / 2, (size[1] - 1) / 2]}}
        if i in fitted:
            # Reprojection error of the anchor's map correspondences under the shared focal and path.
            record['check_rms_px'] = round(fitted[i][2], 2)
        samples.append(record)
    return samples, {'frames': count, 'failed_steps': sum(s is None for s in steps),
                     'median_inliers': float(np.median(inliers)) if inliers else 0.,
                     'focal_ratio': round(ratio, 4), 'speed_m_s': round(float(np.linalg.norm(path[1])), 1),
                     'anchor_rms_px_max': round(max(a[2] for a in fitted.values()), 2)}


def export_clip(clip, flight, entries, candidates, surfaces, root, source, output, args, start_s):
    """Return the debrief media entry and sample count, or None when no frame of the clip is accepted."""
    if not any(identifier in candidates for identifier in flight['photos']):
        return None, 0
    directory = output / Path(clip['video']).stem
    directory.mkdir(parents=True, exist_ok=True)
    size, fps, frames = display_copy(source / clip['video'], directory / 'video.mp4',
                                     min(args.width, MAX_DISPLAY_SIDE))
    shot = Path(clip['video']).stem
    samples, first, anchors = [], None, {}
    for identifier in flight['photos']:
        if identifier not in candidates:
            continue
        _, run, result, path, catalog, kind = candidates[identifier]
        with np.load(run / identifier / 'candidate.npz') as state:
            state = {key: state[key] for key in state.files}
        samples.append(sample(entries[identifier], state, surfaces[catalog], shot, kind, result))
        anchors[entries[identifier]['video_frame']['index']] = (state, kind, result, catalog)
        if first is None:
            first = (identifier, state, catalog, path, run)
    measured = sum(kind == 'measured' for _, kind, _, _ in anchors.values())
    note = (f'{measured} of {len(flight["photos"])} sampled frames matched to the map; '
            f'{len(anchors) - measured} tracked from them; poses between samples are interpolated')
    if args.masks.get(shot):
        if len({a[3] for a in anchors.values()}) != 1:
            raise ValueError(f'{shot}: dense tracking needs anchors from one reference catalog')
        accepted_count = len(samples)
        samples, stats = dense_samples(clip, anchors, surfaces[first[2]], source, args.masks[shot], shot,
                                       entries[first[0]]['size'], shot in args.track_clouds)
        measured = sum(s['kind'] == 'measured' for s in samples)
        note = (f'{measured} of {stats["frames"]} frames matched to the map; the other '
                f'{len(samples) - measured} tracked from them frame to frame; every frame is then aligned '
                f'to the rendered terrain')
        print(json.dumps({'clip': shot, 'anchors': accepted_count, **stats}), flush=True)
    doc = {'version': 3, 'name': shot, 'duration_s': frames / fps, 'fps': fps, 'agl_m': None,
           'origin': None, 'mosaic': None, 'frames': [],
           'video': {'file': 'video.mp4', 'size': size},
           'registration': f'{note}. Imagery: Esri. Terrain: Mapzen / AWS.'}
    identifier, state, catalog, path, run = first
    entry = entries[identifier]
    evidence = hashlib.sha256(''.join([digest(path), digest(run / identifier / 'candidate.npz'),
                                       entry['image_sha256'], entry['mask_sha256'], catalog]).encode()).hexdigest()
    review = reconstruction(surfaces[catalog], state, entry, root, directory, evidence)
    grid = surfaces[catalog].grids['imagery'][0]
    scale = min(1, args.basemap_side / max(grid.shape[:2]))
    write_image(directory / 'basemap.png', cv2.resize(grid, (round(grid.shape[1] * scale), round(grid.shape[0] * scale))))
    doc['reconstruction'] = {
        key: review[key] for key in ['view', 'overview', 'camera', 'review', 'reference_image']}
    doc['reconstruction']['basemap'] = {'image': 'basemap.png', 'corners': map_corners(
        surfaces[catalog], 0, 0, grid.shape[1], grid.shape[0])}
    doc['reconstruction']['pose_track'] = {'max_gap_s': args.max_gap_s, 'samples': samples}
    if args.context.get(shot):
        # Oblique views see far beyond the localization catalog; a coarse mosaic fills the view to the horizon.
        context = Surface(args.context[shot])
        coarse = context.grids['imagery'][0]
        shrink = min(1, args.context_side / max(coarse.shape[:2]))
        write_image(directory / 'context.png', cv2.resize(coarse, (round(coarse.shape[1] * shrink),
                                                                  round(coarse.shape[0] * shrink))))
        doc['reconstruction']['context_basemap'] = {'image': 'context.png', 'corners': map_corners(
            context, 0, 0, coarse.shape[1], coarse.shape[0])}
    # The pose track replaces the single-photo ground projection.
    (directory / 'projection.png').unlink(missing_ok=True)
    write_json(directory / 'replay.json', doc)
    return {'id': shot, 'label': shot, 'replay': f'{shot}/replay.json', 'start_s': start_s,
            'duration_s': doc['duration_s']}, len(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, help='video test set from video_testset.py')
    parser.add_argument('source', type=Path, help='directory that contains the source videos')
    parser.add_argument('output', type=Path)
    parser.add_argument('--catalog', type=Path, action='append', required=True)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--max-gap-s', type=float, default=1.)
    # The base map is context outside the video; a public page pays for every byte of it.
    parser.add_argument('--basemap-side', type=int, default=1536)
    parser.add_argument('--name', default='Cockpit video poses')
    parser.add_argument('--context', action='append', default=[], metavar='CLIP=CATALOG',
                        help='coarse imagery catalog drawn beneath the detailed basemap out to the horizon')
    parser.add_argument('--context-side', type=int, default=1024)
    parser.add_argument('--track-clouds', action='append', default=[], metavar='CLIP',
                        help='also track rotation on clouds for a clip whose ground is too hazy to track')
    parser.add_argument('--masks', action='append', default=[], metavar='CLIP=DIR',
                        help='video_masks.py mask directory; tracks rotation for every frame of that clip')
    args = parser.parse_args()
    args.masks = {key: Path(value) for key, _, value in (item.partition('=') for item in args.masks)}
    args.context = {key: Path(value) for key, _, value in (item.partition('=') for item in args.context)}
    root, output = args.dataset.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    entries = {e['id']: e for e in manifest['photos']}
    surfaces = {digest(path): Surface(path) for path in args.catalog}
    candidates = accepted(args.run)
    unknown = {c[4] for c in candidates.values()} - set(surfaces)
    if unknown:
        parser.error(f'runs use catalogs that were not given: {sorted(unknown)}')
    units, counts, start_s = [], {}, 0.
    for clip, flight in zip(manifest['videos'], manifest['flights']):
        media, count = export_clip(clip, flight, entries, candidates, surfaces, root, args.source.resolve(),
                                   output, args, start_s)
        counts[clip['video']] = count
        if media is None:
            continue
        # Clips play one after another; their recording times are days apart and do not share a clock.
        start_s = round(start_s + media.pop('duration_s') + 1, 3)
        # One unit per clip: each video is its own camera path and timeline.
        units.append({'id': media['id'], 'sidc': SIDC, 'media': [media]})
    write_json(output / 'debrief.json', {'format': 'sokoly-debrief', 'version': 1, 'name': args.name, 'units': units})
    print(json.dumps({'accepted_samples': counts, 'output': str(output)}))


if __name__ == '__main__':
    main()
