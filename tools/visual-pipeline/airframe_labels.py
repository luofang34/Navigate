"""Make offline airframe labels with SAM2 point and box prompts.

The labels train a segmentation model; they are not used at run time.
`photos` labels each still image in a photo test set. `video` propagates one
keyframe prompt forward and backward through a video.
"""
import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np

SAM_MODEL = 'facebook/sam2.1-hiera-tiny'


def parse_object(obj):
    """Split one object prompt into an optional box and its [x, y, label] points."""
    box = obj[0][1:] if obj and obj[0][0] == 'box' else None
    points = [p for p in obj if p[0] != 'box']
    return box, points


def decode_video(path, width):
    """Decode at the same width as video_masks.py so frame indices and mask files agree."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f'cannot open video: {path}')
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (width, round(frame.shape[0] * width / frame.shape[1])),
                                 interpolation=cv2.INTER_AREA))
    capture.release()
    if not frames:
        raise ValueError(f'no frames decoded: {path}')
    return np.stack(frames)


def contact_sheet(cells, columns=4, edge=400):
    tiles = []
    for image, caption in cells:
        scale = edge / max(image.shape[:2])
        tile = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA)
        cell = np.full((edge + 10, edge + 10, 3), 30, np.uint8)
        cell[:tile.shape[0], :tile.shape[1]] = tile
        cv2.putText(cell, caption, (5, edge), 0, .7, (0, 255, 255), 2)
        tiles.append(cell)
    while len(tiles) % columns:
        tiles.append(np.full_like(tiles[0], 30))
    return np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])


def label_photos(args):
    import torch
    from transformers import Sam2Model, Sam2Processor
    model = Sam2Model.from_pretrained(args.sam_model).to(args.device).eval()
    processor = Sam2Processor.from_pretrained(args.sam_model)
    photos = json.loads((args.testset / 'manifest.json').read_text())['photos']
    # {"<photo id>": [object, ...]}; an object is [["box", x0, y0, x1, y1]?, [x, y, label], ...]
    prompts = json.loads(args.prompts.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    cells = []
    for photo in photos:
        if photo['id'] not in prompts:
            continue
        bgr = cv2.imread(str(args.testset / photo['image']))
        mask = np.zeros(bgr.shape[:2], bool)
        for obj in prompts[photo['id']]:
            box, points = parse_object(obj)
            kwargs = {'input_boxes': [[box]]} if box else {}
            if points:
                kwargs.update(input_points=[[[p[:2] for p in points]]], input_labels=[[[p[2] for p in points]]])
            inputs = processor(images=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), return_tensors='pt',
                               **kwargs).to(args.device)
            with torch.inference_mode():
                out = model(**inputs, multimask_output=False)
            mask |= processor.post_process_masks(out.pred_masks.cpu(), inputs['original_sizes'].cpu())[0][0, 0].numpy()
        np.savez_compressed(args.output / f'{photo["id"]}.npz', mask=mask)
        if args.sheet:
            view = bgr.copy()
            view[mask] = (view[mask] * .4 + np.array([255, 0, 255]) * .6).astype(np.uint8)
            cells.append((view, f'{photo["id"][-12:]} {mask.mean():.2f}'))
    if args.sheet and cells:
        cv2.imwrite(str(args.sheet), contact_sheet(cells), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(json.dumps({'photos': len(prompts), 'output': str(args.output)}))


def label_video(args):
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    # {"frame": keyframe, "objects": [[[x, y, label], ...], ...], "valid_from_frame": first frame where the airframe is in view}
    prompt = json.loads(args.prompts.read_text())
    frames = decode_video(args.video, args.width)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    model = Sam2VideoModel.from_pretrained(args.sam_model).to(args.device, dtype=torch.float32)
    processor = Sam2VideoProcessor.from_pretrained(args.sam_model)
    masks = np.zeros(frames.shape[:3], bool)
    objects = prompt['objects']
    for reverse in [False, True]:
        session = processor.init_video_session(video=rgb, inference_device=args.device, dtype=torch.float32)
        processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=prompt['frame'], obj_ids=list(range(1, len(objects) + 1)),
            input_points=[[[p[:2] for p in obj] for obj in objects]],
            input_labels=[[[p[2] for p in obj] for obj in objects]])
        model(inference_session=session, frame_idx=prompt['frame'])
        for output in model.propagate_in_video_iterator(session, start_frame_idx=prompt['frame'], reverse=reverse):
            result = processor.post_process_masks([output.pred_masks], binarize=True,
                                                  original_sizes=[[session.video_height, session.video_width]])[0]
            masks[output.frame_idx] = result.any(dim=0)[0].cpu().numpy()
        # A held session keeps every frame's features; two sessions do not fit together on a laptop GPU.
        del session
        gc.collect()
        if args.device == 'mps':
            torch.mps.empty_cache()
    # Before the airframe enters the view, propagation can attach the prompt to unrelated objects.
    masks[:prompt.get('valid_from_frame', 0)] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, masks=masks)
    print(json.dumps({'frames': len(frames), 'mean_fraction': round(float(masks.mean()), 4),
                      'output': str(args.output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sam-model', default=SAM_MODEL)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    commands = parser.add_subparsers(dest='command', required=True)
    photos = commands.add_parser('photos')
    photos.add_argument('testset', type=Path)
    photos.add_argument('prompts', type=Path)
    photos.add_argument('output', type=Path)
    photos.add_argument('--sheet', type=Path)
    video = commands.add_parser('video')
    video.add_argument('video', type=Path)
    video.add_argument('prompts', type=Path)
    video.add_argument('output', type=Path)
    video.add_argument('--width', type=int, default=960)
    args = parser.parse_args()
    {'photos': label_photos, 'video': label_video}[args.command](args)


if __name__ == '__main__':
    main()
