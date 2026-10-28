"""Disk exports and visualization adapters, separate from numerical processing."""
import json
from pathlib import Path

import cv2
import numpy as np


def normalized(values):
    values = np.maximum(values, 0)
    return values / max(float(values.max()), 1e-12)


def write_png(path, pixels):
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise ValueError(f"Could not encode {path}")
    encoded.tofile(path)


def export_result(directory, arrays, metadata, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    views = []
    downloads = []

    def add(key, label, pixels, description):
        filename = key + ".png"
        write_png(directory / filename, pixels)
        views.append({"key": key, "label": label, "file": filename, "description": description})

    def mono(values):
        return np.rint(normalized(values) * 255).astype(np.uint8)

    add("original", "Input image", arrays["original"], "Input at the processing resolution.")
    add("gray", "Grayscale", mono(arrays["gray"]), "Grayscale input to the filter bank; display scaled to its maximum.")
    add("gabor_combined", "Gabor · all orientations", mono(arrays["gabor_combined"]),
        "Responses summed over scales and orientations; display scaled to its maximum.")
    s, b = arrays["curve_sal"], arrays["junction_sal"]
    shared_max = max(float(np.maximum(s, b).max()), 1e-12)
    saliency = np.stack([b, s, np.zeros_like(s)], axis=-1) / shared_max
    add("saliency", "Curve / junction saliency", np.rint(np.clip(saliency, 0, 1) * 255).astype(np.uint8),
        "Green = curve saliency; blue = junction saliency. Shared linear display scale (Fig. 3c).")
    add("curve_sal", "Curve saliency", mono(s), "Dense curve strength after voting; display scaled to its maximum.")
    add("junction_sal", "Junction saliency", mono(b), "Dense junction strength after voting; display scaled to its maximum.")
    classification = np.zeros_like(arrays["original"])
    active = (s + b) > 1e-12
    classification[active & arrays["is_curve"]] = (90, 225, 65)
    classification[active & ~arrays["is_curve"]] = (245, 110, 45)
    add("classification", "Dominant feature", classification,
        "Green = curve > junction; blue = junction >= curve. Zero response shown black. No road segmentation.")
    add("curve_map", "Classified curve ridges", mono(arrays["curve_map"]),
        "Curve-dominant pixels after non-maximum suppression across the curve (Fig. 3d approximation).")
    overlay = arrays["original"].copy()
    overlay[arrays["curve_points"]] = (70, 235, 120)
    add("overlay", "Curve overlay", overlay, "Classified curve ridges over the input image.")
    hsv = np.zeros_like(arrays["original"])
    hsv[..., 0] = np.floor(arrays["theta"] / np.pi * 179).astype(np.uint8)
    hsv[..., 1] = 220
    hsv[..., 2] = mono(s)
    add("orientation", "Curve orientation", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),
        "Hue encodes tangent angle 0–180°; brightness encodes curve saliency. Red 0°, green 60°, blue 120°.")
    # All orientation images share the bank's normalization, for valid comparison.
    for i, response in enumerate(arrays["gabor_responses"]):
        add(f"gabor_{i:02d}", f"Gabor · {i * 180 / config.n_orient:g}°",
            np.rint(np.clip(response, 0, 1) * 255).astype(np.uint8),
            "Sum across frequencies at this tangent orientation. Shared normalization across the filter bank.")
    if 'road_mask' in arrays:
        mask = arrays['road_mask']
        add('road_mask', 'Segmented road candidates', mask.astype(np.uint8) * 255,
            'White = foreground road candidates. Seeded graph-cut prototype, not a validated road network.')
        segmented_overlay = arrays['original'].copy()
        segmented_overlay[mask] = np.rint(0.5 * segmented_overlay[mask] + 0.5 * np.array([65, 220, 120])).astype(np.uint8)
        add('segmentation_overlay', 'Segmentation overlay', segmented_overlay,
            'Green tint marks road candidates over the processing-resolution input.')
        seeds = arrays['original'].copy()
        seeds[arrays['road_seeds']] = (60, 70, 245)
        seeds[arrays['background_seeds']] = (245, 120, 45)
        add('seed_overlay', 'Road / background seeds', seeds,
            'Red = hard road seeds; blue = hard background seeds used by the color models and graph cut.')
        segment_hsv = np.zeros_like(arrays['original'])
        angles = np.nan_to_num(arrays['segment_orientation_rad'])
        segment_hsv[..., 0] = np.floor(angles / np.pi * 179).astype(np.uint8)
        segment_hsv[..., 1] = 220
        valid_angles = mask & np.isfinite(arrays['segment_orientation_rad'])
        segment_hsv[..., 2] = valid_angles.astype(np.uint8) * 255
        add('segmentation_orientation', 'Segmented road orientation', cv2.cvtColor(segment_hsv, cv2.COLOR_HSV2BGR),
            'Quantized tangent angle on foreground candidates (0–180°). Black when orientation mode is disabled.')
    if 'network_mask' in arrays:
        magnitude = arrays['centreline_magnitude']
        add('centreline_magnitude', 'Centreline filter magnitude', mono(magnitude),
            'Combined road-side × road-area filter response, maximized over widths and orientations.')
        strong = magnitude > 0.1 * magnitude.max()
        hsv = np.zeros_like(arrays['original'])
        hsv[..., 0] = np.floor(arrays['centreline_orientation'] / np.pi * 179).astype(np.uint8)
        hsv[..., 1] = 220
        hsv[..., 2] = strong.astype(np.uint8) * 255
        add('centreline_orientation', 'Centreline filter orientation', cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),
            'Winning tangent angle 0–180°. Display hides responses below 10% of the maximum, as in the script.')
        widths = cv2.applyColorMap(mono(arrays['centreline_width']), cv2.COLORMAP_VIRIDIS)
        widths[~strong] = 0
        add('centreline_width', 'Estimated road width', widths,
            'Winning tested width in processed pixels. Dark-to-bright indicates smaller-to-larger widths; raw values are in arrays.npz.')
        add('network_centrelines', 'Extracted centrelines', arrays['network_centrelines'].astype(np.uint8) * 255,
            'Iterative Hough line segments retained for the extracted network.')
        add('network_mask', 'Network polygon footprint', arrays['network_mask'].astype(np.uint8) * 255,
            'Rasterized union of tracked road polygons. This may extend beyond the segmentation mask.')
        network_overlay = arrays['original'].copy()
        contours, _ = cv2.findContours(arrays['network_mask'].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(network_overlay, contours, -1, (80, 230, 80), 1)
        network_overlay[arrays['network_centrelines']] = (0, 240, 255)
        info = metadata['network_extraction']
        vectors = info['vectors']
        for line in vectors['centrelines']:
            mid = np.rint(np.mean([line['points'][0], line['points'][-1]], axis=0)).astype(int)
            cv2.putText(network_overlay, str(line['id']), tuple(mid), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 240, 255), 1, cv2.LINE_AA)
        add('network_overlay', 'Final road network', network_overlay,
            'Yellow = centreline segments and IDs; green = tracked polygon boundaries.')
        add('network_road_map', 'Schematic road map', arrays['network_road_map'],
            'Road bands drawn at each segment’s median estimated width.')
        export_info = {'coordinate_system': info['coordinate_system'], 'processed_size': metadata['processed_size']}
        (directory / 'network.json').write_text(json.dumps({**export_info, **vectors}, indent=2), encoding='utf-8')
        features = [{'type': 'Feature', 'properties': {'kind': 'centreline', 'id': line['id'], 'width_px': line['width']},
                     'geometry': {'type': 'LineString', 'coordinates': line['points']}}
                    for line in vectors['centrelines']]
        features.append({'type': 'Feature', 'properties': {'kind': 'road_footprint'}, 'geometry': vectors['road_network']})
        (directory / 'network.geojson').write_text(json.dumps({'type': 'FeatureCollection', **export_info, 'features': features}), encoding='utf-8')
        downloads.extend([{'file': 'network.json', 'label': 'Network JSON'},
                          {'file': 'network.geojson', 'label': 'Pixel-coordinate GeoJSON'}])
        # Large vectors live in dedicated downloads rather than every progress response.
        metadata = {**metadata, 'network_extraction': {key: value for key, value in info.items() if key != 'vectors'}}
    np.savez_compressed(directory / "arrays.npz", **arrays)
    manifest = {**metadata, "settings": config.to_dict(), "views": views,
                "raw_arrays": "arrays.npz", 'downloads': downloads}
    (directory / "result.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
