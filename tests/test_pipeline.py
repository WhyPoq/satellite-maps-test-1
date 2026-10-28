import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from gabor import gabor_bank
from pipeline import PipelineConfig, process_image
from tensor_voting import decompose, encode, infer


class NumericalTests(unittest.TestCase):
    def test_invalid_settings_fail_before_computation(self):
        for values in ({"n_scale": 1}, {"n_orient": True}, {"sigma": float("nan")},
                       {"ul": 0.4, "uh": 0.3}, {"ul": 0.3, "uh": 0.31},
                       {"max_dimension": 0}, {"sigma": "12"}, {"unknown": 2}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PipelineConfig.from_dict(values)

    def test_tensor_encoding_retains_known_tangent(self):
        responses = np.zeros((8, 8, 8), dtype=np.float32)
        responses[2] = 1
        stick, ball, theta = decompose(*encode(responses))
        np.testing.assert_allclose(stick, 1, atol=1e-6)
        np.testing.assert_allclose(ball, 0, atol=1e-6)
        np.testing.assert_allclose(theta, np.pi / 4, atol=1e-6)

    def test_empty_input_has_no_spurious_ridges(self):
        bank = gabor_bank(np.zeros((48, 48), np.float32), n_scale=2)
        result = infer(bank, sigma=3, n=2)
        self.assertFalse(result["curve_points"].any())
        np.testing.assert_array_equal(result["curve_map"], 0)
        for values in result.values():
            self.assertTrue(np.isfinite(values).all())

    def test_frequency_bounds_change_filter_responses(self):
        image = np.random.default_rng(42).random((64, 64), dtype=np.float32)
        first = gabor_bank(image, 4, 2, 0.05, 0.4)
        second = gabor_bank(image, 4, 2, 0.1, 0.3)
        self.assertGreater(float(np.abs(first - second).mean()), 0.01)

    def test_pipeline_resizes_and_produces_consistent_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.png"
            image = np.zeros((100, 200, 3), np.uint8)
            cv2.line(image, (10, 50), (190, 50), (255, 255, 255), 3)
            cv2.imwrite(str(path), image)
            stages = []
            arrays, metadata = process_image(path, PipelineConfig(max_dimension=128), stages.append)
        self.assertEqual(metadata["processed_size"], [128, 64])
        self.assertEqual(metadata["original_size"], [200, 100])
        self.assertEqual(stages, ["preprocessing", "gabor", "tensor_voting"])
        self.assertEqual(arrays["gabor_responses"].shape, (8, 64, 128))
        self.assertGreater(arrays["curve_points"].sum(), 0)
        self.assertTrue(np.all(arrays["curve_points"] <= arrays["is_curve"]))
        np.testing.assert_array_equal(arrays["is_curve"], arrays["curve_sal"] > arrays["junction_sal"])
        self.assertTrue(np.all((arrays["theta"] >= 0) & (arrays["theta"] < np.pi)))
        for values in arrays.values():
            self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
