#!/usr/bin/env python3
"""Import playable camera media without assuming a location or camera calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from capture_clock import capture_anchor


def import_video(video, output, fps=5.0, label=None):
    video = video.resolve()
    output = output.resolve()
    probe = json.loads(subprocess.run([
        'ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(video)
    ], check=True, capture_output=True, text=True).stdout)
    stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    numerator, denominator = map(int, stream['avg_frame_rate'].split('/'))
    duration = float(stream['duration'])
    if not 0.1 <= fps <= numerator / denominator or duration <= 0:
        raise ValueError('invalid duration or sample rate')
    if (output / 'replay.json').exists() or (output / 'frames').exists():
        raise ValueError('prepared media already exists; use a different output directory')
    output.mkdir(parents=True, exist_ok=True)
    try:
        relative = video.relative_to(output)
    except ValueError:
        relative = Path('originals') / video.name
        target = output / relative
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            raise ValueError('source destination already exists')
        shutil.copyfile(video, target)
    with video.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    with (output / relative).open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != digest:
            raise ValueError('source copy checksum mismatch')
    frames_dir = output / 'frames'
    frames_dir.mkdir()
    subprocess.run([
        'ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '4', '-i', str(output / relative),
        '-map', '0:v:0', '-vf', f'fps={fps}:start_time=0,scale=960:-2',
        '-compression_level', '2', '-start_number', '0', str(frames_dir / '%06d.png')
    ], check=True)
    frames = [{'time_s': i / fps, 'image': str(p.relative_to(output)),
               'corners': None, 'position': None, 'inliers': 0, 'rms_px': None}
              for i, p in enumerate(sorted(frames_dir.glob('*.png'))) if i / fps < duration]
    if not frames:
        raise ValueError('no frames decoded')
    name = label or video.name
    asset = {'id': 'camera-original', 'path': str(relative), 'media_type': 'video/mp4',
             'source': str(video), 'sha256': digest, 'bytes': video.stat().st_size}
    document = {'version': 3, 'name': name, 'duration_s': duration, 'fps': fps,
                'agl_m': None, 'origin': None, 'mosaic': None, 'frames': frames,
                'source_sha256': digest, 'source': asset, 'time_anchor': capture_anchor(video),
                'registration': 'Camera media only. Position, calibration and ground projection are unavailable.'}
    (output / 'replay.json').write_text(json.dumps(document, indent=2) + '\n')
    archive = {'format': 'sokoly-debrief', 'version': 1, 'name': name,
               'activities': [{'id': 'flight-1', 'title': 'Flight 1', 'entity': 'UAV-1', 'kind': 'flight',
                               'recorded': {'start': 0, 'end': round(duration * 1000)}}],
               'units': [{'id': 'UAV-1', 'sidc': '130401000012030000080100000000',
                          'media': [{'id': 'camera-1', 'activity': 'flight-1', 'label': name,
                                     'replay': 'replay.json', 'start_s': 0}]}]}
    (output / 'debrief.json').write_text(json.dumps(archive, indent=2) + '\n')
    print(f'Imported {len(frames)} samples, {duration:.3f} s: {output / "debrief.json"}', flush=True)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--fps', type=float, default=5)
    parser.add_argument('--label')
    args = parser.parse_args()
    import_video(args.video, args.output, args.fps, args.label)


if __name__ == '__main__':
    main()
