"""Prepare a compact, portable photo replay without changing solver input pixels."""
import argparse
import json
from pathlib import Path
import shutil

import cv2

from photo_testset import digest, write_json


def export(source, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    archive = json.loads(source.read_text())
    archive['name'] = 'Delaware · aerial photo reconstruction'
    shared = {}
    for unit in archive['units']:
        for clip in unit['media']:
            original = source.parent / clip['replay']
            replay = json.loads(original.read_text())
            identifier = clip['id']

            def asset(name, kind):
                path = original.parent / name
                signature = digest(path)
                key = (signature, kind)
                if key in shared:
                    return shared[key]
                # Camera PNGs are the exact normalized benchmark inputs. Only display
                # derivatives are resized; camera calibration remains in original pixels.
                suffix = '.png'
                target = output / (signature + ('' if kind == 'photo' else '-' + kind) + suffix)
                if kind == 'photo':
                    shutil.copyfile(path, target)
                else:
                    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    if image is None:
                        raise ValueError(f'cannot decode {path}')
                    edge = 1024 if kind in {'basemap', 'projection'} else 640
                    scale = min(1, edge / max(image.shape[:2]))
                    if scale < 1:
                        image = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)),
                                           interpolation=cv2.INTER_AREA)
                    options = [cv2.IMWRITE_PNG_COMPRESSION, 6]
                    if not cv2.imwrite(str(target), image, options):
                        raise OSError(f'cannot encode {target}')
                shared[key] = target.name
                return target.name

            for frame in replay['frames']:
                frame['image'] = asset(frame['image'], 'photo')
            review = replay.get('reconstruction')
            if review:
                review['reference_image'] = asset(review['reference_image'], 'reference')
                for key in ['basemap', 'detail_basemap', 'projection']:
                    if review.get(key):
                        review[key]['image'] = asset(review[key]['image'], 'projection' if key == 'projection' else 'basemap')
            replay.pop('source', None)
            clip['replay'] = identifier + '.json'
            write_json(output / clip['replay'], replay)
    write_json(output / 'debrief.json', archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    archive = export(args.source, args.output)
    print(json.dumps({'flights': len(archive['units']), 'photos': sum(len(u['media']) for u in archive['units']),
                      'bytes': sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())}))


if __name__ == '__main__':
    main()
