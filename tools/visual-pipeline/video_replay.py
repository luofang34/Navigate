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
from video_track import FeatureTracker, dense_rotations, interpolate_eyes

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


def dense_samples(clip, anchors, surface, source, masks, shot, size):
    """One sample per video frame: anchors keep their pose, other frames are tracked between them."""
    frames = decode_video(source / clip['video'], 960)
    count, height, width = frames.shape[:3]
    state = next(iter(anchors.values()))[0]
    scale = width / float(state['size'][0])
    K = np.array([[state['K'][0, 0] * scale, 0, (width - 1) / 2], [0, state['K'][1, 1] * scale, (height - 1) / 2], [0, 0, 1.]])
    tracker = FeatureTracker(K)
    features = [tracker.features(frame, cv2.imread(str(masks / f'{i:06d}-ground.png'), 0) > 0)
                for i, frame in enumerate(frames)]
    steps, inliers = [], []
    for i in range(count - 1):
        step, support = tracker.rotation(features[i], features[i + 1])
        steps.append(step)
        inliers.append(support)
    rotations = dense_rotations({f: a[0]['rotation'] for f, a in anchors.items()}, steps, count)
    eyes = interpolate_eyes({f: a[0]['eye'] for f, a in anchors.items()}, count)
    order = sorted(anchors)
    focal_ratio = np.interp(np.arange(count), order,
                            [anchors[f][0]['K'][0, 0] / float(anchors[f][0]['size'][0]) for f in order])
    samples = []
    for i in range(count):
        if rotations[i] is None:
            continue
        lat, lon, alt = surface.geodetic(eyes[i])
        terrain, covered = surface.sample('dem', lat, lon)
        if not covered:
            continue
        focal = float(focal_ratio[i] * size[0])
        record = {'time_s': round(i / clip['fps'], 4), 'kind': anchors[i][1] if i in anchors else 'tracked',
                  'shot': shot, 'view': view(eyes[i], rotations[i], surface), 'camera_agl_m': float(alt - terrain),
                  'camera': {'size': list(size), 'focal': [focal, focal],
                             'principal': [(size[0] - 1) / 2, (size[1] - 1) / 2]}}
        if i in anchors:
            record['check_rms_px'] = float(anchors[i][2]['candidate']['score']['train_rms_px'])
        samples.append(record)
    return samples, {'frames': count, 'failed_steps': sum(s is None for s in steps),
                     'median_inliers': float(np.median(inliers)) if inliers else 0.}


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
    note = 'poses between samples are interpolated'
    if args.masks.get(shot):
        if len({a[3] for a in anchors.values()}) != 1:
            raise ValueError(f'{shot}: dense tracking needs anchors from one reference catalog')
        accepted_count = len(samples)
        samples, stats = dense_samples(clip, anchors, surfaces[first[2]], source, args.masks[shot], shot,
                                       entries[first[0]]['size'])
        note = (f'every frame has a pose; {stats["failed_steps"]} of {stats["frames"] - 1} frame-to-frame '
                f'rotations failed (median {stats["median_inliers"]:.0f} inliers)')
        print(json.dumps({'clip': shot, 'anchors': accepted_count, **stats}), flush=True)
    doc = {'version': 3, 'name': shot, 'duration_s': frames / fps, 'fps': fps, 'agl_m': None,
           'origin': None, 'mosaic': None, 'frames': [],
           'video': {'file': 'video.mp4', 'size': size},
           'registration': f'{len(anchors)} map-matched poses of {len(flight["photos"])} sampled frames; '
                           f'{note}. Imagery: Esri. Terrain: Mapzen / AWS.'}
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
    parser.add_argument('--masks', action='append', default=[], metavar='CLIP=DIR',
                        help='video_masks.py mask directory; tracks rotation for every frame of that clip')
    args = parser.parse_args()
    args.masks = {key: Path(value) for key, _, value in (item.partition('=') for item in args.masks)}
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
