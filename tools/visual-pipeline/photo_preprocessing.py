"""Select masked-window and grayscale matching without changing camera geometry."""
from pathlib import Path

import cv2
import numpy as np

from photo_bundle_pairs import matching_window
from photo_render_check import alignment
from photo_testset import write_json


class WindowMatcher:
    """Restore full-image coordinates before terrain fitting or verification."""
    def __init__(self, matcher, variant, bounds=None):
        self.matcher = matcher
        self.variant = variant
        self.bounds = bounds
        self.color = getattr(matcher, 'color', False)
        self.patch_refinement = getattr(matcher, 'patch_refinement', True)

    def match(self, reference, query, refinement=False):
        if self.variant == 'full-color':
            return self.matcher.match(reference, query, refinement=refinement)
        directory = Path(reference).parent / self.variant
        directory.mkdir(exist_ok=True)
        paths = []
        for name, path in [('reference', reference), ('query', query)]:
            mode = cv2.IMREAD_GRAYSCALE if self.variant.endswith('gray') else cv2.IMREAD_COLOR
            pixels = cv2.imread(str(path), mode)
            if pixels is None:
                raise ValueError(f'cannot decode matching image: {path}')
            if self.bounds is not None:
                left, top, right, bottom = self.bounds
                pixels = pixels[top:bottom, left:right]
            destination = directory / (name + '.png')
            if not cv2.imwrite(str(destination), pixels):
                raise OSError(f'cannot save matching image: {destination}')
            paths.append(destination)
        response = self.matcher.match(*paths, refinement=refinement)
        pairs = np.asarray(response['pairs'], float).reshape(-1, 2, 2)
        if self.bounds is not None:
            pairs += self.bounds[:2]
        return {**response, 'pairs': pairs.tolist(),
                'backend': response['backend'] + '/' + self.variant}


def select_preprocessing(matcher, reference, query, mask, reference_mask, directory):
    """Rank bounded preprocessing trials at a frozen seed, before fitting any pose."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    intersection = (mask > 0) & (reference_mask > 0)
    choices = [WindowMatcher(matcher, 'full-color')]
    reports = []
    if intersection.any():
        bounds = list(matching_window(intersection))
        if bounds != [0, 0, mask.shape[1], mask.shape[0]]:
            choices += [WindowMatcher(matcher, 'crop-color', bounds),
                        WindowMatcher(matcher, 'crop-gray', bounds)]
        choices.append(WindowMatcher(matcher, 'full-gray'))
        for choice in choices:
            response = choice.match(reference, query, refinement=choice.patch_refinement)
            report, pairs, errors = alignment(response['pairs'], mask, reference_mask)
            report.update(variant=choice.variant, crop_xyxy=choice.bounds,
                          backend=response['backend'])
            np.savez_compressed(directory / (choice.variant + '.npz'),
                                pairs=pairs, errors=errors)
            reports.append(report)
    # Seed ranking is not acceptance. Every fitted camera must still pass the
    # spatial holdout and a fresh render using the complete image's support area.
    supported = [i for i, r in enumerate(reports) if r['pairs'] >= 100
                 and r['median_px'] <= 30
                 and (r['image_support_fraction'] >= .05 or r['pairs'] >= 1000)]
    selected = max(supported, key=lambda i: reports[i]['pairs'] / (1 + reports[i]['median_px'])) if supported else 0
    result = {'selected': choices[selected].variant, 'crop_xyxy': choices[selected].bounds,
              'attempts': reports, 'pose_refitted': False}
    write_json(directory / 'selection.json', result)
    return choices[selected], result
