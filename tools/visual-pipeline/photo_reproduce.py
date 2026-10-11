"""Download, verify and run a published photo reconstruction locally."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

BASE = 'https://demo.sokoly.app/Delaware/reproduce/'


def contained(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or not name or any(p in {'', '.', '..'} for p in name.split('/')) or '\\' in name:
        raise ValueError(f'unsafe package path: {name}')
    target = root / path
    target.resolve().relative_to(root.resolve())
    return target


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(item, root):
    target = contained(root, item['path'])
    if target.exists():
        if sha256(target) != item['sha256']:
            raise ValueError(f'local file differs: {target}; use a fresh directory')
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        staged = Path(temporary) / 'download'
        request = urllib.request.Request(item['url'], headers={'User-Agent': 'Sokoly-photo-replay/1'})
        with urllib.request.urlopen(request, timeout=120) as response, staged.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        if staged.stat().st_size != item['bytes'] or sha256(staged) != item['sha256']:
            raise ValueError(f'remote content changed: {item["url"]}; expected {item["sha256"]}')
        staged.replace(target)
    if item.get('executable'):
        target.chmod(0o755)
    return target


def fetch(root, base, groups):
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'publication.json'
    if not manifest_path.exists():
        with urllib.request.urlopen(base.rstrip('/') + '/publication.json', timeout=60) as response:
            manifest = json.load(response)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    manifest = json.loads(manifest_path.read_text())
    selected = [item for item in manifest['files'] if item['group'] in groups]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, path in enumerate(pool.map(lambda item: download(item, root), selected), 1):
            if index % 40 == 0 or index == len(selected):
                print(f'Verified {index}/{len(selected)} files', flush=True)
    for item in selected:
        if item.get('contents'):
            with zipfile.ZipFile(contained(root, item['path'])) as archive:
                if set(archive.namelist()) != set(item['contents']):
                    raise ValueError('archive inventory differs')
                for name, expected in item['contents'].items():
                    target = contained(root, name)
                    data = archive.read(name)
                    if hashlib.sha256(data).hexdigest() != expected:
                        raise ValueError(f'archive content differs: {name}')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() and sha256(target) != expected:
                        raise ValueError(f'local executable changed: {target}')
                    if not target.exists():
                        target.write_bytes(data)
                    target.chmod(0o755)
    return manifest


def native_host():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('The downloadable native binaries support Apple Silicon macOS. Use the web viewer on other platforms.')


def run(root, name, *args):
    subprocess.run([sys.executable, str(root / 'scripts' / (name + '.py')), *map(str, args)], check=True)


def restore_images(root):
    manifest = json.loads((root / 'dataset/manifest.json').read_text())
    for entry in manifest['photos']:
        target = contained(root / 'dataset', entry['image'])
        source = root / 'operation/media' / (entry['image_sha256'] + '.png')
        if sha256(source) != entry['image_sha256']:
            raise ValueError(f'benchmark image differs: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                target.hardlink_to(source)
            except OSError:
                shutil.copyfile(source, target)
        elif sha256(target) != entry['image_sha256']:
            raise ValueError(f'benchmark input changed: {target}')


def references(root, manifest):
    tasks = []
    for name in manifest['catalogs']:
        path = contained(root, name)
        catalog = json.loads(path.read_text())
        tasks.extend({**tile, 'path': str((path.parent / tile['path']).relative_to(root))}
                     for tile in catalog['tiles'])
    with ThreadPoolExecutor(max_workers=6) as pool:
        for index, _ in enumerate(pool.map(lambda item: download(item, root), tasks), 1):
            if index % 200 == 0 or index == len(tasks):
                print(f'Verified reference tiles {index}/{len(tasks)}', flush=True)


def new_photos(args, root):
    native_host()
    fetch(root, args.base_url, {'code', 'models', 'native'})
    if not args.source or not args.bbox:
        raise ValueError('new requires --source and --bbox WEST SOUTH EAST NORTH')
    dataset = root / 'new-dataset'
    run(root, 'photo_testset', args.source.resolve(), dataset)
    run(root, 'photo_masks', dataset, '--model', root / 'models/segmentation', '--policy', 'airframe-occlusions')
    run(root, 'photo_reference', root / 'new-reference', '--bbox', *args.bbox, '--imagery-zoom', 16, '--dem-zoom', 14)
    if args.prior:
        lat, lon, radius = args.prior
        if not (-90 <= lat <= 90 and -180 <= lon <= 180 and radius > 0):
            raise ValueError('prior must be latitude longitude positive-radius-meters')
        path = dataset / 'profiles/with-prior.json'
        profile = json.loads(path.read_text())
        for case in profile['cases']:
            if case['prior'].get('position') is None:
                case['prior']['position'] = {'latitude': lat, 'longitude': lon,
                    'reported_horizontal_error_m': radius, 'altitude_m': None,
                    'unsigned_altitude_m': None, 'altitude_datum': 'unknown'}
        path.write_text(json.dumps(profile, indent=2) + '\n')
    recipe = json.loads((root / 'new-template.json').read_text())
    recipe['source'] = str(args.source.resolve())
    path = root / 'new-photos.json'
    path.write_text(json.dumps(recipe, indent=2) + '\n')
    run(root, 'photo_pipeline', path, '--run-id', args.run_id, '--stage', 'match', '--mode', args.mode)
    run(root, 'photo_pipeline', path, '--run-id', args.run_id, '--stage', 'preview', '--mode', args.mode)
    print('Preview:', dataset / 'runs' / args.run_id / args.mode / 'preview/debrief.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'fetch', 'replay', 'references', 'rebuild', 'match', 'new'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--base-url', default=BASE)
    parser.add_argument('--run-id', default='local-01')
    parser.add_argument('--mode', choices=['with-prior', 'no-prior'], default='with-prior')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--bbox', nargs=4, type=float)
    parser.add_argument('--prior', nargs=3, type=float, metavar=('LAT', 'LON', 'RADIUS_M'))
    args = parser.parse_args()
    root = args.directory.resolve()
    if args.action == 'new':
        new_photos(args, root)
        return
    groups = {'native', 'code'} if args.action == 'setup' else {'replay', 'native', 'code'}
    if args.action not in {'setup', 'fetch', 'replay'}:
        groups.add('benchmark')
    if args.action == 'match':
        native_host()
        groups.add('models')
    manifest = fetch(root, args.base_url, groups)
    if args.action == 'replay':
        native_host()
        subprocess.run([str(root / 'native/sokoly-app'), '--operation', str(root / 'operation'), '--no-save', '--paused', '--recorded-path'], check=True)
    elif args.action in {'references', 'rebuild', 'match'}:
        restore_images(root)
        references(root, manifest)
        if args.action == 'rebuild':
            command = [root / 'dataset', root / 'rebuilt-preview']
            for catalog in manifest['catalogs']:
                command += ['--catalog', root / catalog]
            for directory in manifest['baseline_runs']:
                command += ['--run', root / directory]
            run(root, 'photo_preview', *command)
        elif args.action == 'match':
            for stage in ['match', 'preview']:
                run(root, 'photo_pipeline', root / 'pipeline.json', '--run-id', args.run_id, '--stage', stage, '--mode', args.mode)


if __name__ == '__main__':
    main()
