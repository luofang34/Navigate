# Visual recording pipeline (`navigate-visual-pipeline`)

`navigate-visual-pipeline` is a Rust command-line tool and library.
It prepares aerial photos and videos for visual camera positioning.
It builds occlusion masks, renders terrain reference views, matches images, checks camera poses, and exports viewer archives.
It does not use Python.
It does not depend on Sokoly source code.

## Technical names

- A **dataset** is a directory that holds `manifest.json`, normalized images, valid-pixel masks, and case profiles.
- A **case profile** is a JSON file (`profiles/with-prior.json` or `profiles/no-prior.json`) that lists image cases and optional search priors.
- A **search prior** is an unverified position or heading claim that bounds the pose search.
- A **reference catalog** (`catalog.json`) is an immutable manifest of Web Mercator orthoimagery tiles and Mapzen Terrarium elevation tiles with SHA-256 digests.
- A **valid-pixel mask** is a grayscale PNG where `255` marks matchable terrain and `0` marks sky, cabin, airframe, or propeller pixels.
- A **pose candidate** (`candidate.npz`) is a 6-DoF camera pose and 3D-2D correspondence set that passed spatial holdout checks.
- A **reference bank** (`NVKBANK1`) is a binary file of precomputed reference views, 3D terrain points, 160-byte `ViewSketch` summaries, and 28-byte `KashinDescriptor` codes with a SHA-256 checksum.
- A **debrief archive** (`debrief.json`) is a `sokoly-debrief` version 1 output file for display and review.

## Build and run

Run all commands from the Navigate repository root.
Install the Rust toolchain.
Install `magick` or macOS `sips` only when you decode HEIC or RAW source photos.
Install `ffmpeg` only when you extract frames from video files.
Install `curl` only when you download remote tiles or reproduction bundles.

Build the pipeline binary:

```sh
cargo build --release -p navigate-visual-pipeline
```

Show all subcommands and options:

```sh
cargo run --release -p navigate-visual-pipeline -- --help
```

## Photo workflow

1. Run `prepare` to decode source photos into a new dataset directory.
   The tool does not change the source directory.
2. Run `priors` to interpolate missing search priors along each flight on the unit sphere.
3. Run `mask` with a SegFormer-B0 checkpoint directory to write valid-pixel masks and semantic class maps.
4. Run `reference` to download tiles and record their SHA-256 digests in a reference catalog.
5. Run `match`, `track`, `refine`, or `terrain-bundle` (or run `pipeline --recipe pipeline.json`) to fit and check pose candidates.
   Select `--backend xfeat`, `--backend loftr`, or `--backend pyramidal`.
   Supply `.safetensors` weights for in-process `burn` execution.
   Supply `.onnx` weights with `--onnx-runtime` for worker execution.
6. Run `preview` to write `debrief.json` (`sokoly-debrief` version 1) and `replay.json` (version 3).
7. Run `freeze` to store accepted pose candidates.
8. Run `publish` to write `publication.json`, and run `reproduce` to check SHA-256 digests.

## Precomputed reference bank and real-time matching

1. Run `bank-build` to render terrain seed views with adaptive ray-marching, extract `XFeat` keypoints, ray-cast 3D terrain points at keypoint pixels, and save a reference bank (`NVKBANK1`).
2. Run `bank-match` to extract masked `XFeat` keypoints from each query photo, shortlist reference views by `ViewSketch` similarity, match 3D-2D correspondences from the reference bank, and solve 6-DoF camera poses with deterministic L1-GNC and spatial holdout checks.
3. Supply `--lighterglue` when you want adaptive early-exit `LighterGlue` cross-attention on shortlisted reference bank entries.
4. Supply `--polish` when you want one adaptive render and sparse keypoint ray-cast refinement step on the winning pose candidate.

```sh
cargo run --release -p navigate-visual-pipeline -- bank-build \
  --dataset /path/to/dataset \
  --profile /path/to/dataset/profiles/with-prior.json \
  --catalog /path/to/catalog.json \
  --model /path/to/xfeat-f32.safetensors \
  --output /path/to/corridor.kashin

cargo run --release -p navigate-visual-pipeline -- bank-match \
  --dataset /path/to/dataset \
  --profile /path/to/dataset/profiles/with-prior.json \
  --catalog /path/to/catalog.json \
  --bank /path/to/corridor.kashin \
  --model /path/to/xfeat-f32.safetensors \
  --lighterglue /path/to/lighterglue-f32.safetensors \
  --polish \
  --output /path/to/run-out
```

## Semantic segmentation and Rust training

`navigate-segment` implements SegFormer-B0 inference, propeller detection, and 4-class training in Rust on `burn`.
The four classes are `sky` (`0`), `ground` (`1`), `airframe` (`2`), and `propeller` (`3`).

1. Run `airframe-labels` to combine binary class masks into a 4-class label PNG and an optional review contact sheet.
2. Put training and validation pairs in `images/train`, `labels/train`, `images/val`, and `labels/val`.
3. Run `segment-finetune` to train the 4-class model with differential AdamW learning rates.
   The command writes `config.json`, `preprocessor_config.json`, `model.safetensors`, and `training.json`.

```sh
cargo run --release -p navigate-visual-pipeline -- segment-finetune \
  --dataset /path/to/seg-dataset \
  --base-model /path/to/segformer-b0 \
  --output /path/to/model-out \
  --size 512 --batch-size 8 --epochs 40
```

## Video and Navigate session workflow

1. Run `import-video` to record the video SHA-256 digest and QuickTime `moov/mvhd` clock anchor.
   The tool ignores filesystem modification dates.
2. Run `live-prior` to correlate a Sokoly version 1 live replay with the video clock and write `acquisition.json` (`navigate-prior-track` version 1).
3. Run `video-prepare`, `video-priors`, and `video-masks` to prepare masked video frames.
4. Run `navigate-debrief` to process `configuration.json` and `observations.json` through `navigate_recording::Processor` and write `results.json` and `debrief.json`.
5. Run `compose-survey` to align multiple flight debrief archives on one UTC timeline.

## Quality checks

Run the unit tests, linter, and documentation checks:

```sh
cargo test -p navigate-segment -p navigate-visual-pipeline
cargo clippy -p navigate-segment -p navigate-visual-pipeline --all-targets -- -D warnings
RUSTDOCFLAGS="-D missing_docs -D rustdoc::broken_intra_doc_links" \
  cargo doc -p navigate-segment -p navigate-visual-pipeline --no-deps
```
