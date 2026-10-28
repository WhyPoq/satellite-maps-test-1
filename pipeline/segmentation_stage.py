"""Adapter between tensor-voting outputs and the seeded segmentation prototype."""
import cv2

from segmentation import segment
from .seeds import seed_masks


def run_segmentation(arrays, config, annotation, report):
    road, background = seed_masks(annotation, arrays['gray'].shape)
    # Only classified curve ridges carry orientation evidence, as in paper section 5.
    # Saliency is a relative confidence, not a calibrated probability.
    confidence = arrays['curve_map']
    data = dict(image=cv2.cvtColor(arrays['original'], cv2.COLOR_BGR2RGB),
                orientation_rad=arrays['theta'], orientation_confidence=confidence,
                road_seeds=road, background_seeds=background)
    result = segment(data, config.segmentation_config(), config.segmentation_orientation,
                     progress=lambda cycle, label, count: report(
                         f'segmentation · cycle {cycle}/{config.seg_max_cycles} · label {label}/{count}'))
    output = dict(road_seeds=road, background_seeds=background,
                  segmentation_confidence=confidence, road_mask=result['road_mask'],
                  segment_labels=result['label_map'], segment_orientation_rad=result['orientation_rad'],
                  segment_orientation_bin=result['orientation_bin'])
    metadata = {key: result[key] for key in ('method', 'config', 'energy_trace', 'cycles', 'converged')}
    metadata.update(road_fraction=float(result['road_mask'].mean()),
                    seed_pixels={'road': int(road.sum()), 'background': int(background.sum())},
                    confidence_source='normalized curve saliency on classified curve ridges')
    return output, metadata
