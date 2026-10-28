from itertools import product
import unittest

import numpy as np

from pipeline import PipelineConfig
from pipeline.seeds import seed_masks
from segmentation import Config, binary_cut, energy, expansion_move, segment


class SegmentationTests(unittest.TestCase):
    def test_invalid_config_and_capacity_overflow(self):
        for settings in ({'seg_n_angles': 2.5}, {'seg_max_cycles': 0},
                         {'seg_smoothness': float('inf')}, {'seg_color_sigma': 0},
                         {'segmentation_enabled': 1}, {'seg_capacity_scale': True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                PipelineConfig.from_dict(settings)
        with self.assertRaisesRegex(ValueError, 'Capacity too large'):
            binary_cut(np.zeros(2), np.full(2, 1e100), np.array([0]), np.array([1]), np.ones(1))

    def test_expansion_move_matches_exhaustive_minimum(self):
        rng = np.random.default_rng(7)
        p, q = np.array([0, 1, 0]), np.array([1, 2, 2])
        for _ in range(12):
            unary = rng.random((3, 4)) * 4
            weights = rng.random(3)
            labels = rng.integers(0, 4, 3)
            for alpha in range(4):
                proposal = expansion_move(labels, alpha, unary, p, q, weights, 1000000)
                optimum = min(energy(np.where(switch, alpha, labels), unary, p, q, weights)
                              for switch in product((False, True), repeat=3))
                self.assertAlmostEqual(energy(proposal, unary, p, q, weights), optimum, places=5)

    def test_seed_coordinates_survive_resize_and_last_stroke_wins(self):
        annotation = {'strokes': [
            {'kind': 'background', 'radius': 0.03, 'points': [[0.2, 0.5], [0.8, 0.5]]},
            {'kind': 'road', 'radius': 0.03, 'points': [[0.5, 0.5]]},
        ]}
        for shape in ((64, 128), (128, 256)):
            road, background = seed_masks(annotation, shape)
            h, w = shape
            self.assertTrue(road[round((h - 1) / 2), round((w - 1) / 2)])
            self.assertTrue(background[round((h - 1) / 2), round((w - 1) * 0.2)])
            self.assertFalse((road & background).any())
        with self.assertRaises(ValueError):
            seed_masks({'strokes': [{'kind': 'road', 'radius': 0.01, 'points': [[float('nan'), 0.5]]}]}, (64, 64))
        with self.assertRaisesRegex(ValueError, 'at least 5'):
            seed_masks({'strokes': []}, (64, 64))

    def test_synthetic_segmentation_preserves_seeds_and_reduces_energy(self):
        image = np.full((20, 24, 3), 20, np.uint8)
        image[:, 8:16] = 220
        road = np.zeros(image.shape[:2], bool)
        background = road.copy()
        road[5:10, 10] = True
        background[5:10, 2] = True
        data = dict(image=image, orientation_rad=np.full(road.shape, np.pi / 2),
                    orientation_confidence=np.ones(road.shape), road_seeds=road, background_seeds=background)
        for oriented in (False, True):
            calls = []
            result = segment(data, Config(n_angles=4, max_cycles=3), oriented,
                             progress=lambda *args: calls.append(args))
            self.assertTrue(result['road_mask'][road].all())
            self.assertFalse(result['road_mask'][background].any())
            np.testing.assert_array_equal(result['road_mask'], image[..., 0] > 100)
            self.assertTrue(np.all(np.diff(result['energy_trace']) < 0))
            self.assertGreater(len(calls), 0)
            self.assertEqual(result['label_map'].dtype, np.int16)
            if not oriented:
                self.assertTrue(np.isnan(result['orientation_rad']).all())


if __name__ == '__main__':
    unittest.main()
