"""Sample cockpit video frames into a photo-terrain test set.

Each sampled frame becomes one case with the clip's QuickTime location and the
recording start plus the frame time. The location is one claim for the clip,
not a per-frame position. The camera direction is unknown.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np

from photo_testset import cases, contact_sheets, digest, write_json

ISO6709 = re.compile(r'^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?/?$')
# iPhone wide-camera horizontal field of view for 4:3 frames; 16:9 video keeps the frame width.
WIDE_HORIZONTAL_FOV_DEG = 67.3


def quicktime_claims(tags):
    """Read the keyed QuickTime metadata that iPhone writes; legacy ©xyz atoms are absent."""
    position = None
    match = ISO6709.match(tags.get('com.apple.quicktime.location.ISO6709', ''))
    if match:
        latitude, longitude = float(match[1]), float(match[2])
        altitude = float(match[3]) if match[3] else None
        accuracy = tags.get('com.apple.quicktime.location.accuracy.horizontal')
        position = {'latitude': latitude, 'longitude': longitude,
                    'reported_horizontal_error_m': float(accuracy) if accuracy else None,
                    'altitude_m': altitude, 'unsigned_altitude_m': None if altitude is None else abs(altitude),
                    'altitude_reference_text': 'ISO6709', 'altitude_datum': 'unverified'}
    created = tags.get('com.apple.quicktime.creationdate')
    return {'capture_time': datetime.fromisoformat(created).isoformat() if created else None,
            'clock_source': 'QuickTime creationdate plus decoded frame time', 'clock_accuracy_s': None,
            'position': position, 'position_scope': 'one location claim for the whole clip',
            'direction_deg': None, 'direction_reference': None, 'focal_mm': None,
            'camera_model': tags.get('com.apple.quicktime.model'), 'lens_model': None}


def equivalent_focal(width, height):
    """Express the assumed horizontal field of view as the 35 mm diagonal focal the matcher expects."""
    focal_px = width / 2 / math.tan(math.radians(WIDE_HORIZONTAL_FOV_DEG / 2))
    return round(focal_px * math.hypot(36, 24) / math.hypot(width, height), 2)


def probe(video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-of', 'json', str(video)],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)['format'].get('tags', {})


def sample_video(video, root, fps, edge, source_root):
    tags = probe(video)
    base = quicktime_claims(tags)
    video_sha = digest(video)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f'cannot open video: {video}')
    rate = capture.get(cv2.CAP_PROP_FPS)
    step = max(1, round(rate / fps))
    entries = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            entries.append(frame_entry(frame, index, rate, video, video_sha, base, root, edge, source_root))
        index += 1
    capture.release()
    if not entries:
        raise ValueError(f'no frames decoded: {video}')
    return entries, {'video': str(video.relative_to(source_root)), 'sha256': video_sha, 'frames': index,
                     'fps': rate, 'sample_step': step, 'tags': tags}


def frame_entry(frame, index, rate, video, video_sha, base, root, edge, source_root):
    identifier = 'q-' + hashlib.sha256(f'{video_sha}:{index}'.encode()).hexdigest()[:20]
    height, width = frame.shape[:2]
    scale = min(1., edge / max(width, height))
    picture = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    size = [picture.shape[1], picture.shape[0]]
    image, mask = f'images/{identifier}.png', f'masks/{identifier}.png'
    for name, pixels in [(image, picture), (mask, np.full(picture.shape[:2], 255, np.uint8)),
                         (f'thumbnails/{identifier}.jpg',
                          cv2.resize(picture, (320, round(size[1] * 320 / size[0])), interpolation=cv2.INTER_AREA))]:
        if not cv2.imwrite(str(root / name), pixels):
            raise OSError(f'cannot write {name}')
    claims = dict(base, focal_35mm_mm=equivalent_focal(width, height),
                  focal_basis=f'assumed {WIDE_HORIZONTAL_FOV_DEG} degree horizontal field of view')
    time_s = index / rate
    if claims['capture_time']:
        claims['capture_time'] = (datetime.fromisoformat(claims['capture_time']) + timedelta(seconds=time_s)).isoformat()
    entry = {'id': identifier, 'source': f'{video.relative_to(source_root)}#frame={index}',
             'source_sha256': video_sha, 'image': image, 'image_sha256': digest(root / image),
             'size': size, 'valid_mask': mask, 'mask_status': 'unreviewed_all_pixels',
             'mask_sha256': digest(root / mask), 'companions': [],
             'video_frame': {'index': index, 'time_s': round(time_s, 4)},
             'preprocess': {'version': 1, 'long_edge': edge, 'orientation': 'decoder', 'color': 'sRGB'},
             'decoded_size': [width, height], 'metadata_version': 2, 'exif': {}, 'claims': claims}
    write_json(root / 'metadata' / (identifier + '.json'), entry)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='directory that contains the videos')
    parser.add_argument('output', type=Path)
    parser.add_argument('videos', type=Path, nargs='+', help='video paths inside the source directory')
    parser.add_argument('--fps', type=float, default=3.)
    parser.add_argument('--long-edge', type=int, default=1600)
    args = parser.parse_args()
    source, root = args.source.resolve(), args.output.resolve()
    if root == source or root.is_relative_to(source):
        parser.error('output must be outside the source tree')
    if args.fps <= 0 or args.long_edge < 64:
        parser.error('invalid sample rate or image size')
    for folder in ['images', 'masks', 'metadata', 'thumbnails', 'profiles']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    entries, flights, clips = [], [], []
    for number, video in enumerate(args.videos, 1):
        video = video.resolve()
        if not video.is_relative_to(source):
            parser.error(f'{video} is outside the source directory')
        sampled, clip = sample_video(video, root, args.fps, args.long_edge, source)
        entries += sampled
        clips.append(clip)
        # One clip is one continuous camera path; the flight group carries its frame order.
        flights.append({'id': f'flight-{number:03}', 'status': 'video clip', 'basis': clip['video'],
                        'photos': [e['id'] for e in sampled]})
    version = subprocess.run(['ffprobe', '-version'], capture_output=True, text=True, check=True).stdout
    manifest = {'format': 'photo-terrain-testset', 'version': 1, 'source_root': str(source),
                'tools': {'ffprobe': version.splitlines()[0], 'opencv': cv2.__version__},
                'sample_fps': args.fps, 'videos': clips, 'flights': flights, 'photos': entries,
                # Each video frame is its own exposure; tracking must not treat a clip as one capture.
                'capture_groups': [{'photos': [e['id']], 'basis': 'one video frame'} for e in entries]}
    write_json(root / 'manifest.json', manifest)
    for name, prior in [('with-prior', True), ('no-prior', False)]:
        write_json(root / 'profiles' / (name + '.json'), {
            'format': 'photo-terrain-cases', 'version': 1, 'root': '..',
            'profile': name, 'reference_catalog': None, 'cases': cases(entries, prior)})
    contact_sheets(root, entries)
    print(json.dumps({'frames': len(entries), 'clips': len(clips), 'output': str(root)}))


if __name__ == '__main__':
    main()
