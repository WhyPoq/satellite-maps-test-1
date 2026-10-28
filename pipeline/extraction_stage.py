"""Numerical adapter for the supplied Section 6 extraction implementation."""
import cv2
import numpy as np
from shapely.geometry import LineString, mapping
import shapely

from road_network_extraction import (
    automatic, centreline_response, curves_from_mask, iterative_hough,
    segment_width, track,
)


def polygon_parts(geometry):
    """Yield polygon parts, including those nested in geometry collections."""
    if geometry.geom_type == 'Polygon':
        if not geometry.is_empty:
            yield geometry
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def polygon_mask(geometry, shape):
    result = np.zeros(shape, np.uint8)
    for polygon in polygon_parts(geometry):
        part = np.zeros(shape, np.uint8)
        exterior = np.rint(polygon.exterior.coords).astype(np.int32)
        cv2.fillPoly(part, [exterior], 1)
        for ring in polygon.interiors:
            cv2.fillPoly(part, [np.rint(ring.coords).astype(np.int32)], 0)
        result |= part
    return result.astype(bool)


def run_extraction(arrays, config, report):
    if 'road_mask' not in arrays:
        raise ValueError('Network extraction needs the segmentation road mask.')
    mask = arrays['road_mask']
    report('extraction · preparing road sides')
    if config.net_curve_source == 'mask':
        # Matches the supplied standalone CLI.
        saliency, orientation = curves_from_mask(mask)
    else:
        # The complete pipeline has the actual Stage 1 curve evidence available.
        saliency, orientation = arrays['curve_map'], arrays['theta']
    widths = config.extraction_widths()
    response = centreline_response(saliency, orientation, mask, widths,
        n_orientations=config.net_n_orientations, sigma_across=config.net_sigma_across,
        sigma_along=config.net_sigma_along, progress=report)
    report('extraction · iterative Hough and connected network')
    if config.net_keep_longest:
        lines = automatic(response, mask)
    else:
        lines = dict(enumerate(iterative_hough(response, mask)))
    report('extraction · tracking road sides and polygon union')
    network = track(lines, response)
    # Polygonal output is the footprint; degenerate lines can occur in a union.
    network = shapely.unary_union(list(polygon_parts(network)))
    shape = mask.shape
    center_mask = np.zeros(shape, np.uint8)
    road_map = np.full((*shape, 3), (236, 232, 226), np.uint8)
    vector_lines = []
    for key, vertices in lines.items():
        points = np.rint(vertices).astype(np.int32).reshape(-1, 1, 2)
        width = segment_width(vertices, response)
        vector_lines.append({'id': key, 'points': vertices.tolist(), 'width': width})
        cv2.polylines(center_mask, [points], False, 1, 1)
        cv2.polylines(road_map, [points], False, (150, 150, 150), max(1, round(width)) + 4, cv2.LINE_AA)
    for line in vector_lines:
        points = np.rint(line['points']).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(road_map, [points], False, (255, 255, 255), max(1, round(line['width'])), cv2.LINE_AA)
    output = dict(centreline_magnitude=response.magnitude, centreline_orientation=response.orientation,
                  centreline_width=response.width, network_centrelines=center_mask.astype(bool),
                  network_mask=polygon_mask(network, shape), network_road_map=road_map,
                  extraction_curve_saliency=saliency)
    warnings = []
    if not lines:
        warnings.append('No centreline segments were found. Check the segmentation, road-side input, and road-width range.')
    if config.net_max_width >= min(shape):
        warnings.append('Maximum road width reaches the shorter image dimension; consider smaller widths at this resolution.')
    metadata = dict(centreline_count=len(lines),
                    total_length_px=float(sum(LineString(c).length for c in lines.values())),
                    polygon_area_px2=float(network.area), curve_source=config.net_curve_source,
                    keep_longest=config.net_keep_longest, widths=widths, warnings=warnings,
                    coordinate_system='processed image pixels; origin top left, x right, y down',
                    vectors={'centrelines': vector_lines, 'road_network': mapping(network)})
    return output, metadata
