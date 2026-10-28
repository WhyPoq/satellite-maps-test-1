import unittest
import numpy as np

from pipeline import PipelineConfig
from pipeline.extraction_stage import run_extraction


class ExtractionTests(unittest.TestCase):
    def config(self, **values):
        return PipelineConfig(segmentation_enabled=True, extraction_enabled=True,
            net_curve_source='mask', net_min_width=8, net_max_width=16,
            net_width_step=4, net_n_orientations=4, **values)

    def arrays(self, mask):
        return {'road_mask': mask, 'curve_map': np.zeros(mask.shape, np.float32),
                'theta': np.zeros(mask.shape, np.float32),
                'original': np.zeros((*mask.shape, 3), np.uint8)}

    def test_extracts_known_road_width_direction_and_polygon(self):
        mask = np.zeros((64, 96), bool)
        mask[26:38, 8:88] = True
        calls = []
        arrays, metadata = run_extraction(self.arrays(mask), self.config(), calls.append)
        self.assertEqual(metadata['centreline_count'], 1)
        line = metadata['vectors']['centrelines'][0]
        self.assertAlmostEqual(line['width'], 12)
        self.assertAlmostEqual(metadata['total_length_px'], 79)
        self.assertGreater(metadata['polygon_area_px2'], 900)
        self.assertTrue(arrays['network_mask'][32, 48])
        self.assertFalse(arrays['network_mask'][0, 0])
        self.assertTrue(arrays['network_centrelines'].any())
        self.assertTrue(any('filters' in call for call in calls))

    def test_empty_road_mask_returns_empty_outputs_and_warning(self):
        arrays, metadata = run_extraction(self.arrays(np.zeros((40, 40), bool)), self.config(), lambda _: None)
        self.assertEqual(metadata['centreline_count'], 0)
        self.assertEqual(metadata['total_length_px'], 0)
        self.assertFalse(arrays['network_mask'].any())
        self.assertFalse(arrays['network_centrelines'].any())
        self.assertTrue(metadata['warnings'])

    def test_all_components_and_longest_component_choices(self):
        mask = np.zeros((96, 128), bool)
        mask[15:27, 8:100] = True
        mask[65:77, 8:65] = True
        _, longest = run_extraction(self.arrays(mask), self.config(), lambda _: None)
        _, all_components = run_extraction(self.arrays(mask), self.config(net_keep_longest=False), lambda _: None)
        self.assertGreater(all_components['centreline_count'], longest['centreline_count'])
        self.assertGreater(all_components['total_length_px'], longest['total_length_px'])

    def test_invalid_settings_and_stage_dependency(self):
        for settings in ({'extraction_enabled': True}, {'net_min_width': 20, 'net_max_width': 10},
                         {'net_width_step': 0}, {'net_n_orientations': 4.5},
                         {'net_curve_source': 'unknown'}, {'net_sigma_along': float('nan')},
                         {'net_min_width': 2, 'net_max_width': 128, 'net_width_step': 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                PipelineConfig.from_dict(settings)


if __name__ == '__main__':
    unittest.main()
