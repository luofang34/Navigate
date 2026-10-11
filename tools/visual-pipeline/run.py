"""Run a Navigate photo or video processing command."""
import argparse
from pathlib import Path
import subprocess
import sys

COMMANDS = {
    'prepare': 'photo_testset',
    'mask': 'photo_masks',
    'reference': 'photo_reference',
    'pipeline': 'photo_pipeline',
    'preview': 'photo_preview',
    'freeze': 'photo_freeze',
    'reproduce': 'photo_reproduce',
    'publish': 'photo_publication',
    'import-video': 'import-video',
    'prepare-video-overlay': 'prepare-video-overlay',
    'prepare-navigate-debrief': 'prepare-navigate-debrief',
    'compose-survey': 'compose-survey',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=sorted(COMMANDS))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script = Path(__file__).with_name(COMMANDS[args.command] + '.py')
    return subprocess.run([sys.executable, str(script), *args.arguments]).returncode


if __name__ == '__main__':
    raise SystemExit(main())
