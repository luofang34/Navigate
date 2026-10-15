"""Seed every video frame's map search with the pose that video_replay.py tracked for it.

The tracked pose becomes the frame's position and true-heading prior, so a
following photo_match.py pass registers each frame against the terrain
independently. The prior only seeds the search; acceptance still requires the
map geometry checks.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from photo_preview import TO_GLOBE
from photo_surface import RADIUS, basis
from photo_testset import cases, write_json


def matrix(quaternion):
    w, x, y, z = quaternion
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def geodetic_pose(globe_pose):
    """Camera position and true heading of the optical axis from a viewer globe pose."""
    # The stored quaternion rotates the globe frame into the camera; its transpose holds the camera axes.
    axes = matrix(globe_pose['orientation']).T
    target = np.array(globe_pose['target']) * (RADIUS + globe_pose['target_elevation_meters'])
    eye = TO_GLOBE.T @ (target + axes[:, 2] * globe_pose['distance_meters'])
    lat = float(np.degrees(np.arctan2(eye[2], np.hypot(eye[0], eye[1]))))
    lon = float(np.degrees(np.arctan2(eye[1], eye[0])))
    east, north, _ = basis(lat, lon).T @ (TO_GLOBE.T @ -axes[:, 2])
    return lat, lon, float(np.linalg.norm(eye) - RADIUS), float(np.degrees(np.arctan2(east, north)) % 360)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, help='full-rate video test set from video_testset.py')
    parser.add_argument('replay', type=Path, help='video_replay.py output with one pose sample per frame')
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    tracks = {}
    for path in args.replay.glob('*/replay.json'):
        doc = json.loads(path.read_text())
        tracks[path.parent.name] = {round(s['time_s'] * doc['fps']): s
                                    for s in doc['reconstruction']['pose_track']['samples']}
    seeded = 0
    for entry in manifest['photos']:
        clip = Path(entry['source'].split('#')[0]).stem
        sample = tracks.get(clip, {}).get(entry['video_frame']['index'])
        if sample is None:
            continue
        lat, lon, alt, heading = geodetic_pose(sample['view']['globe_pose'])
        claims = entry['claims']
        claims['position'] = {**(claims['position'] or {}), 'latitude': lat, 'longitude': lon,
                              'altitude_m': alt, 'unsigned_altitude_m': abs(alt)}
        claims.update(direction_deg=heading, direction_reference='T',
                      position_scope=f'tracked video pose ({sample["kind"]}); a search seed, not evidence')
        write_json(root / 'metadata' / (entry['id'] + '.json'), entry)
        seeded += 1
    write_json(root / 'manifest.json', manifest)
    for name, prior in [('with-prior', True), ('no-prior', False)]:
        path = root / 'profiles' / (name + '.json')
        profile = json.loads(path.read_text())
        profile['cases'] = cases(manifest['photos'], prior)
        write_json(path, profile)
    print(json.dumps({'frames': len(manifest['photos']), 'seeded': seeded}))


if __name__ == '__main__':
    main()
