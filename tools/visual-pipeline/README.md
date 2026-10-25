# Visual recording pipeline

This component prepares photos and videos for visual positioning. It owns image
masks, reference rendering, image matching, camera pose fitting, and checks of
pose quality. It uses the Navigate visual libraries through the recording worker.
It does not depend on Sokoly source code.

## Build and run

Run these commands from the Navigate root. Install Rust, Python 3.12 or later,
uv, ImageMagick, and FFmpeg. ImageMagick decodes source photos. FFmpeg reads videos.

```sh
cargo build --release -p navigate-recording-worker
uv sync --locked --project tools/visual-pipeline --extra photo-matching --extra photo-roma
uv run --locked --project tools/visual-pipeline python tools/visual-pipeline/run.py --help
uv run --locked --project tools/visual-pipeline --extra photo-matching --extra photo-roma \
  python tools/visual-pipeline/run.py pipeline /path/to/recipe.json --run-id trial-01
```

A recipe gives paths to the dataset, masks, reference catalogs, model files, and
worker. Set `worker` to `target/release/navigate-recording-worker`. Recipe paths
are relative to the recipe file. Keep model and reference hashes with the recipe.
Use `--dry-run` to inspect the commands. Use `--stage preview` to produce views
from saved results. Use `--mode no-prior` for regional retrieval without photo
location claims. This mode does not perform a global search.

The `photo_*.py` commands remain available for individual processing stages.
Use each command's `--help` to inspect its inputs. A new run ID keeps an independent
run. Reuse a run ID only to resume that run.

For video, use `import-video`, then `prepare-navigate-debrief` with a map package,
model, runtime library, and acquisition prior. The worker accepts JSON Lines on
standard input. It writes one result per input line. The `navigate-worker` command
processes recording evidence. The `match-worker` command computes image matches.

## Results and viewer boundary

`navigate-recording` owns the recording input and result types. A result keeps
camera pose, source identity, support limits, and unlocated intervals. It does
not supply vehicle-body telemetry.

`photo_preview.py` exports the `sokoly-debrief` version 1 catalog and version 3
camera replays. The export adapter maps camera coordinates into the viewer's globe
coordinates. It retains intrinsics, terrain projections, masks, flight groups,
and evidence identities. The viewer loads these results; it does not fit poses.
`photo_publication.py` makes a portable display package without changing camera
pixels or calibration. Sokoly owns playback and operation-package publication.

Saved reproduction bundles contain frozen source and model hashes. They do not
change when this component changes. Use their frozen entry points to reproduce
their results. Do not replace their scripts with newer source files.

## Checks

```sh
cargo test -p navigate-recording -p navigate-recording-worker
uv run --locked --project tools/visual-pipeline \
  python -m unittest discover -s tools/visual-pipeline -p 'test_*.py'
```

Set `SOKOLY_APP_BINARY` to test the export with a built Sokoly viewer. The backend
tests also run without a viewer. The navigation core and browser matching use
`navigate-visual` and `navigate-visual-session`. This Python tool supplies the
recorded-photo workflow. It does not replace their browser or video interfaces.
