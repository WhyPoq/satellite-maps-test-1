from io import BytesIO
import json
from pathlib import Path
import tempfile
import time
import unittest
import zipfile
from unittest.mock import patch

import cv2
import numpy as np

from web.app import create_app
from web.jobs import JobManager


class WebIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.inputs, self.outputs = root / "inputs", root / "outputs"
        self.inputs.mkdir()
        for index, name in enumerate(("one.png", "two.png")):
            image = np.zeros((48, 64, 3), np.uint8)
            cv2.line(image, (5, 15 + 10 * index), (60, 35), (255, 255, 255), 3)
            cv2.imwrite(str(self.inputs / name), image)
        self.app = create_app(self.inputs, self.outputs)
        self.client = self.app.test_client()

    def tearDown(self):
        self.app.extensions["jobs"].executor.shutdown(wait=True)
        self.temporary.cleanup()

    def wait_for_job(self, job_id):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            job = self.client.get(f"/api/runs/{job_id}").get_json()
            if job["status"] not in ("queued", "running"):
                return job
            time.sleep(0.02)
        self.fail("Background job did not finish.")

    def test_catalog_and_invalid_requests(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        catalog = self.client.get("/api/images").get_json()
        self.assertEqual(len(catalog["images"]), 2)
        thumbnail = self.client.get("/api/images/one.png/thumbnail")
        self.assertEqual(thumbnail.mimetype, "image/jpeg")
        for payload in ({"files": []}, {"files": ["../paper.pdf"]}, {"files": ["one.png", "one.png"]},
                        {"files": ["one.png"], "settings": {"n_scale": 1}},
                        {"files": ["one.png"], "settings": []}):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/api/runs", json=payload).status_code, 400)
        self.assertEqual(self.client.get("/outputs/unknown/x/arrays.npz").status_code, 404)

    def test_multi_image_run_exports_raw_arrays_previews_and_history(self):
        response = self.client.post("/api/runs", json={"files": ["one.png", "two.png"],
            "settings": {"n_orient": 4, "n_scale": 2, "n_angular": 2, "sigma": 3, "max_dimension": 128}})
        self.assertEqual(response.status_code, 202)
        job_id = response.get_json()["id"]
        job = self.wait_for_job(job_id)
        self.assertEqual(job["status"], "completed", job)
        self.assertEqual(job["completed"], 2)
        self.assertEqual(len(job["results"]), 2)
        for result in job["results"]:
            self.assertEqual(result["processed_size"], [64, 48])  # never upscale
            self.assertEqual(len(result["views"]), 14)
            for view in result["views"]:
                preview = self.client.get(f'/outputs/{job_id}/{result["folder"]}/{view["file"]}')
                self.assertEqual(preview.status_code, 200)
                self.assertTrue(preview.data.startswith(b"\x89PNG"))
                preview.close()
            response = self.client.get(f'/outputs/{job_id}/{result["folder"]}/arrays.npz')
            with np.load(BytesIO(response.data)) as arrays:
                self.assertEqual(arrays["gabor_responses"].shape, (4, 48, 64))
                self.assertEqual(arrays["curve_points"].dtype, np.bool_)
            response.close()
        archive_response = self.client.get(f"/api/runs/{job_id}/download")
        with zipfile.ZipFile(BytesIO(archive_response.data)) as archive:
            self.assertIn("run.json", archive.namelist())
            self.assertIn("image_001/arrays.npz", archive.namelist())
            saved = json.loads(archive.read("run.json"))
            self.assertEqual(saved["settings"]["sigma"], 3)
        archive_response.close()
        restored = JobManager(self.inputs, self.outputs)
        try:
            self.assertEqual(restored.snapshot(job_id)["status"], "completed")
            self.assertEqual(len(restored.snapshot(job_id)["results"]), 2)
        finally:
            restored.executor.shutdown()

    def test_restart_marks_unfinished_run_interrupted(self):
        directory = self.outputs / "old"
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(json.dumps({"id": "old", "status": "running"}), encoding="utf-8")
        restored = JobManager(self.inputs, self.outputs)
        try:
            self.assertEqual(restored.snapshot("old")["status"], "interrupted")
        finally:
            restored.executor.shutdown()

    def test_seeded_segmentation_batch_exports_and_replays_annotations(self):
        annotation = {'strokes': [
            {'kind': 'road', 'radius': 0.025, 'points': [[0.5, 0.53], [0.6, 0.57]]},
            {'kind': 'background', 'radius': 0.025, 'points': [[0.2, 0.8], [0.4, 0.8]]},
        ]}
        settings = {'max_dimension': 128, 'segmentation_enabled': True,
                    'seg_n_angles': 4, 'seg_max_cycles': 2}
        # Missing seeds are rejected before starting expensive work.
        response = self.client.post('/api/runs', json={'files': ['one.png'], 'settings': settings})
        self.assertEqual(response.status_code, 400)
        self.assertIn('Mark road', response.get_json()['error'])
        annotations = {'one.png': annotation, 'two.png': annotation}
        response = self.client.post('/api/runs', json={'files': list(annotations),
                                   'settings': settings, 'annotations': annotations})
        self.assertEqual(response.status_code, 202, response.get_json())
        job_id = response.get_json()['id']
        job = self.wait_for_job(job_id)
        self.assertEqual(job['status'], 'completed', job['errors'])
        self.assertEqual(job['annotations'], annotations)
        self.assertEqual(len(job['results']), 2)
        for result in job['results']:
            self.assertIn('segmentation', result['timings'])
            self.assertIn('energy_trace', result['segmentation'])
            self.assertIn('road_mask', {view['key'] for view in result['views']})
            response = self.client.get(f'/outputs/{job_id}/{result["folder"]}/arrays.npz')
            with np.load(BytesIO(response.data)) as arrays:
                self.assertEqual(arrays['road_mask'].shape, (48, 64))
                self.assertTrue(arrays['road_mask'][arrays['road_seeds']].all())
                self.assertFalse(arrays['road_mask'][arrays['background_seeds']].any())
                # Adapter supplies sparse curve-ridge evidence, not arbitrary edge pixels.
                np.testing.assert_array_equal(arrays['segmentation_confidence'], arrays['curve_map'])
            response.close()
            for key in ('road_mask', 'segmentation_overlay', 'segmentation_orientation', 'seed_overlay'):
                response = self.client.get(f'/outputs/{job_id}/{result["folder"]}/{key}.png')
                self.assertEqual(response.status_code, 200)
                response.close()

    def test_full_pipeline_and_cached_extraction_downloads(self):
        annotation = {'strokes': [
            {'kind': 'road', 'radius': 0.025, 'points': [[0.5, 0.53], [0.6, 0.57]]},
            {'kind': 'background', 'radius': 0.025, 'points': [[0.2, 0.8], [0.4, 0.8]]},
        ]}
        settings = {'max_dimension': 128, 'segmentation_enabled': True, 'extraction_enabled': True,
                    'seg_n_angles': 4, 'seg_max_cycles': 1, 'net_min_width': 2, 'net_max_width': 12,
                    'net_width_step': 2, 'net_n_orientations': 4, 'net_curve_source': 'mask'}
        response = self.client.post('/api/runs', json={'files': ['one.png', 'two.png'],
                    'settings': settings, 'annotations': {'one.png': annotation, 'two.png': annotation}})
        self.assertEqual(response.status_code, 202, response.get_json())
        source = self.wait_for_job(response.get_json()['id'])
        self.assertEqual(source['status'], 'completed', source['errors'])
        self.assertEqual(len(source['results']), 2)
        for result in source['results']:
            self.assertIn('network_extraction', result['timings'])
            self.assertNotIn('vectors', result['network_extraction'])
            base = f'/outputs/{source["id"]}/{result["folder"]}/'
            for filename in ('network_overlay.png', 'network_mask.png', 'centreline_width.png', 'network_road_map.png'):
                response = self.client.get(base + filename)
                self.assertEqual(response.status_code, 200)
                response.close()
            response = self.client.get(base + 'network.geojson')
            vectors = response.get_json()
            self.assertEqual(vectors['type'], 'FeatureCollection')
            self.assertIn('pixel', vectors['coordinate_system'])
            response.close()
            response = self.client.get(base + 'network.json')
            self.assertIn('centrelines', response.get_json())
            response.close()
        # Cached extraction must never invoke Gabor, voting, or segmentation again.
        with patch('web.jobs.process_image', side_effect=AssertionError('Upstream stages must be reused')):
            response = self.client.post('/api/runs', json={'files': ['one.png'], 'settings': settings,
                                                         'source_run_id': source['id']})
            self.assertEqual(response.status_code, 202, response.get_json())
            cached = self.wait_for_job(response.get_json()['id'])
        self.assertEqual(cached['status'], 'completed', cached['errors'])
        self.assertEqual(cached['source_run_id'], source['id'])
        result = cached['results'][0]
        self.assertEqual(set(result['timings']), {'network_extraction', 'total'})
        self.assertIn('segmentation', result['source_timings'])
        response = self.client.get(f'/outputs/{cached["id"]}/{result["folder"]}/arrays.npz')
        with np.load(BytesIO(response.data)) as arrays:
            self.assertEqual(arrays['network_mask'].dtype, np.bool_)
            self.assertEqual(arrays['centreline_width'].shape, (48, 64))
        response.close()
        # Changing upstream settings would misrepresent the cached arrays.
        response = self.client.post('/api/runs', json={'files': ['one.png'],
            'settings': {**settings, 'sigma': 5}, 'source_run_id': source['id']})
        self.assertEqual(response.status_code, 400)
        self.assertIn('different upstream settings', response.get_json()['error'])


if __name__ == "__main__":
    unittest.main()
