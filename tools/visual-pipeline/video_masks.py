"""Stream per-frame sky, propeller and ground masks from cockpit video."""
import argparse
from collections import deque
import json
from pathlib import Path
import time

import cv2
import numpy as np

from photo_masks import surface_mask

SKY_LABEL = 2
AIRFRAME_CLASSES = ('sky', 'ground', 'airframe', 'propeller')
CONFIDENCE_MIN = .35
PROPELLER_POLICY = {'width_pixels': 480, 'history_frames': 8, 'darker_by': 35,
                    'dark_max': 110, 'minimum_area_fraction': .002, 'padding_pixels': 3}


class PropellerDetector:
    """Find transient dark blades by comparing each frame with past frames only.

    The blade passes faster than the camera rate, so a blade pixel is usually
    bright in most recent frames. The second-brightest past value is the
    background; one earlier blade pass at the same pixel does not hide the blade.
    """
    def __init__(self, policy=PROPELLER_POLICY):
        self.policy = policy
        self.history = deque(maxlen=policy['history_frames'])

    def update(self, frame):
        p = self.policy
        height, width = frame.shape[:2]
        size = (p['width_pixels'], round(height * p['width_pixels'] / width))
        gray = cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY).astype(np.int16)
        blade = np.zeros(gray.shape, np.uint8)
        if len(self.history) >= 3:
            background = np.partition(np.stack(self.history), -2, axis=0)[-2]
            blade = ((gray < background - p['darker_by']) & (gray < p['dark_max'])).astype(np.uint8)
            blade = cv2.morphologyEx(blade, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            count, components, stats, _ = cv2.connectedComponentsWithStats(blade)
            large = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= gray.size * p['minimum_area_fraction'])
            blade = np.isin(components, large[large != 0]).astype(np.uint8)
            pad = 2 * p['padding_pixels'] + 1
            blade = cv2.dilate(blade, np.ones((pad, pad), np.uint8))
        self.history.append(gray)
        return cv2.resize(blade * 255, (width, height), interpolation=cv2.INTER_NEAREST)


class Segmenter:
    def __init__(self, model_path, device):
        import torch
        from transformers import SegformerForSemanticSegmentation
        self.torch = torch
        self.device = device
        processor = json.loads((model_path / 'preprocessor_config.json').read_text())
        config = json.loads((model_path / 'config.json').read_text())
        names = config['id2label']
        if [names.get(str(i)) for i in range(len(names))] == list(AIRFRAME_CLASSES):
            self.scheme = 'airframe'
        elif names.get(str(SKY_LABEL)) == 'sky':
            self.scheme = 'ade'
        else:
            raise ValueError('the sky mask requires ADE20K labels or the segment_finetune.py classes')
        edge = processor['size']
        self.size = (edge, edge) if isinstance(edge, int) else (edge['width'], edge['height'])
        self.mean = np.array(processor['image_mean'], np.float32)
        self.std = np.array(processor['image_std'], np.float32)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_path, local_files_only=True, use_safetensors=True).eval().to(device)

    def labels(self, frame):
        pixels = cv2.cvtColor(cv2.resize(frame, self.size, interpolation=cv2.INTER_LINEAR),
                              cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        pixels = ((pixels - self.mean) / self.std).transpose(2, 0, 1).copy()[None]
        with self.torch.inference_mode():
            logits = self.model(pixel_values=self.torch.from_numpy(pixels).to(self.device)).logits
            confidence, labels = logits.softmax(dim=1)[0].max(dim=0)
        size = (frame.shape[1], frame.shape[0])
        return (cv2.resize(labels.cpu().numpy().astype(np.uint8), size, interpolation=cv2.INTER_NEAREST),
                cv2.resize(confidence.cpu().numpy(), size, interpolation=cv2.INTER_LINEAR))


def frame_masks(frame, labels, confidence, detector):
    propeller = detector.update(frame) > 0
    # Semantic sky often includes a blade that crosses the sky.
    sky = (labels == SKY_LABEL) & (confidence >= CONFIDENCE_MIN) & ~propeller
    ground = (surface_mask(labels, confidence) > 0) & ~sky & ~propeller
    return {name: mask.astype(np.uint8) * 255
            for name, mask in [('sky', sky), ('propeller', propeller), ('ground', ground)]}


def class_masks(frame, labels, detector):
    """Masks from a segment_finetune.py model: only the ground class is usable; airframe is excluded."""
    propeller = (detector.update(frame) > 0) | (labels == AIRFRAME_CLASSES.index('propeller'))
    sky = (labels == AIRFRAME_CLASSES.index('sky')) & ~propeller
    ground = (labels == AIRFRAME_CLASSES.index('ground')) & ~propeller
    return {name: mask.astype(np.uint8) * 255
            for name, mask in [('sky', sky), ('propeller', propeller), ('ground', ground)]}


def preview(frame, masks):
    image = frame.copy()
    for name, color in [('sky', (255, 120, 0)), ('propeller', (0, 220, 255))]:
        selected = masks[name] > 0
        image[selected] = (image[selected] * .4 + np.array(color) * .6).astype(np.uint8)
    excluded = (masks['sky'] | masks['propeller'] | masks['ground']) == 0
    image[excluded] = (image[excluded] * .4 + np.array([40, 0, 200]) * .6).astype(np.uint8)
    return image


def process(video, output, model, device, width):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f'cannot open video: {video}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    segmenter = Segmenter(model, device)
    detector = PropellerDetector()
    masks_dir = output / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=False)
    writer = None
    records = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        start = time.perf_counter()
        frame = cv2.resize(frame, (width, round(frame.shape[0] * width / frame.shape[1])),
                           interpolation=cv2.INTER_AREA)
        labels, confidence = segmenter.labels(frame)
        segmented = time.perf_counter()
        masks = (frame_masks(frame, labels, confidence, detector) if segmenter.scheme == 'ade'
                 else class_masks(frame, labels, detector))
        finished = time.perf_counter()
        index = len(records)
        for name, mask in masks.items():
            path = masks_dir / f'{index:06d}-{name}.png'
            if not cv2.imwrite(str(path), mask):
                raise OSError(f'cannot write {path}')
        if writer is None:
            writer = cv2.VideoWriter(str(output / 'preview.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (frame.shape[1], frame.shape[0]))
        writer.write(preview(frame, masks))
        records.append({'frame': index, 'time_s': index / fps,
                        'segment_ms': round((segmented - start) * 1000, 2),
                        'masks_ms': round((finished - segmented) * 1000, 2),
                        **{name + '_fraction': round(float(np.mean(mask > 0)), 4)
                           for name, mask in masks.items()}})
    capture.release()
    if writer is None:
        raise ValueError(f'no frames decoded: {video}')
    writer.release()
    # The first segmentation includes model warm-up and is not a frame cost.
    steady = records[1:] or records
    summary = {'video': str(video), 'frames': len(records), 'fps': fps, 'width': width,
               'device': device, 'model': str(model), 'label_scheme': segmenter.scheme, 'sky_label': SKY_LABEL,
               'confidence_min': CONFIDENCE_MIN, 'propeller': PROPELLER_POLICY,
               'segment_ms_median': float(np.median([r['segment_ms'] for r in steady])),
               'masks_ms_median': float(np.median([r['masks_ms'] for r in steady]))}
    (output / 'masks.json').write_text(json.dumps({**summary, 'frames_detail': records}, indent=1) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--width', type=int, default=960)
    args = parser.parse_args()
    print(json.dumps(process(args.video.resolve(), args.output.resolve(), args.model,
                             args.device, args.width)), flush=True)


if __name__ == '__main__':
    main()
