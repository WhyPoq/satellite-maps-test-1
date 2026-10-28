"""Feature inference followed by optional seeded road segmentation."""
from time import perf_counter
from pathlib import Path

import cv2
import numpy as np

from gabor import gabor_bank
from tensor_voting import infer
from .config import PipelineConfig


def read_image(path):
    # imdecode supports Windows filenames containing non-ASCII characters.
    image = cv2.imdecode(np.fromfile(Path(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {Path(path).name}")
    return image


def process_image(path, config: PipelineConfig, progress=None, annotation=None):
    """Return raw arrays + metadata; no rendering, disk output, or HTTP concerns.

    Resizing precedes filtering. Frequencies and sigma refer to processed pixels.
    Gabor responses are normalized together, preserving relative orientation strength.
    """
    report = progress or (lambda stage: None)
    start = perf_counter()
    report("preprocessing")
    image = read_image(path)
    original_h, original_w = image.shape[:2]
    scale = min(1.0, config.max_dimension / max(original_h, original_w))
    if scale < 1:
        image = cv2.resize(image, (max(1, round(original_w * scale)),
                                   max(1, round(original_h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    prepared = perf_counter()
    report("gabor")
    responses = gabor_bank(gray, config.n_orient, config.n_scale, config.ul, config.uh)
    filtered = perf_counter()
    report("tensor_voting")
    voted = infer(responses, sigma=config.sigma, n=config.n_angular)
    voted_at = perf_counter()
    arrays = {"original": image, "gray": gray, "gabor_responses": responses,
              "gabor_combined": responses.sum(axis=0), **voted}
    segmentation_metadata = None
    if config.segmentation_enabled:
        from .segmentation_stage import run_segmentation
        report('segmentation')
        segmented, segmentation_metadata = run_segmentation(arrays, config, annotation, report)
        arrays.update(segmented)
    segmented_at = perf_counter()
    extraction_metadata = None
    if config.extraction_enabled:
        from .extraction_stage import run_extraction
        extracted, extraction_metadata = run_extraction(arrays, config, report)
        arrays.update(extracted)
    finished = perf_counter()
    metadata = {
        "original_size": [original_w, original_h],
        "processed_size": [image.shape[1], image.shape[0]], "resize_scale": scale,
        "timings": {"preprocessing": prepared - start, "gabor": filtered - prepared,
                    "tensor_voting": voted_at - filtered, "total": finished - start},
        "curve_fraction": float(voted["is_curve"].mean()),
        "curve_points": int(voted["curve_points"].sum()),
    }
    if segmentation_metadata is not None:
        metadata['segmentation'] = segmentation_metadata
        metadata['timings']['segmentation'] = segmented_at - voted_at
    if extraction_metadata is not None:
        metadata['network_extraction'] = extraction_metadata
        metadata['timings']['network_extraction'] = finished - segmented_at
    return arrays, metadata


def extract_saved(arrays, metadata, config, progress=None):
    """Run only extraction on cached Stage 1/2 arrays; upstream maps stay unchanged."""
    from copy import deepcopy
    from .extraction_stage import run_extraction
    report = progress or (lambda stage: None)
    start = perf_counter()
    # A saved source may itself have extraction outputs. Drop those before rerunning.
    upstream = {key: value for key, value in arrays.items() if not key.startswith(
        ('centreline_', 'network_', 'extraction_'))}
    extracted, extraction_metadata = run_extraction(upstream, config, report)
    upstream.update(extracted)
    result = deepcopy(metadata)
    for key in ('views', 'settings', 'raw_arrays', 'downloads', 'name', 'folder', 'network_extraction'):
        result.pop(key, None)
    elapsed = perf_counter() - start
    result['source_timings'] = result.get('source_timings', result.get('timings', {}))
    result['timings'] = {'network_extraction': elapsed, 'total': elapsed}
    result['network_extraction'] = extraction_metadata
    return upstream, result
