"""Resolution-independent brush annotations, shared by HTTP validation and processing."""
import math

import cv2
import numpy as np


def seed_masks(annotation, shape):
    """Rasterize normalized brush strokes. Later strokes replace earlier class marks.

    Each stroke: {kind: 'road'|'background', radius: fraction of shorter image side,
                  points: [[x,y], ...]} with coordinates in [0,1].
    """
    if not isinstance(annotation, dict) or not isinstance(annotation.get('strokes'), list):
        raise ValueError('Mark road and background seeds using the image seed editor.')
    strokes = annotation['strokes']
    if len(strokes) > 1000:
        raise ValueError('Too many brush strokes (maximum 1000 per image).')
    h, w = shape
    labels = np.zeros((h, w), np.uint8)
    point_count = 0
    for stroke in strokes:
        if not isinstance(stroke, dict) or stroke.get('kind') not in ('road', 'background'):
            raise ValueError('Each seed stroke must be road or background.')
        radius = stroke.get('radius')
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or not 0.001 <= radius <= 0.1:
            raise ValueError('Seed brush radius must be between 0.001 and 0.1 of the shorter image side.')
        points = stroke.get('points')
        if not isinstance(points, list) or not points:
            raise ValueError('Each seed stroke needs at least one point.')
        point_count += len(points)
        if point_count > 20000:
            raise ValueError('Too many seed points (maximum 20000 per image).')
        pixels = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in point
            ):
                raise ValueError('Seed coordinates must be finite numbers in [0,1].')
            pixels.append((round(point[0] * (w - 1)), round(point[1] * (h - 1))))
        color = 1 if stroke['kind'] == 'road' else 2
        radius_px = max(1, round(radius * min(h, w)))
        for first, second in zip(pixels, pixels[1:]):
            cv2.line(labels, first, second, color, radius_px * 2 + 1)
        for point in pixels:
            cv2.circle(labels, point, radius_px, color, -1)
    road, background = labels == 1, labels == 2
    if road.sum() < 5 or background.sum() < 5:
        raise ValueError('Mark at least 5 road and 5 background pixels at the processing resolution; enlarge the brush if needed.')
    return road, background
