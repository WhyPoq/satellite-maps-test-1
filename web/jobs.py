"""Sequential background experiments with durable manifests and per-image errors."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from uuid import uuid4
import numpy as np

from pipeline import PipelineConfig, process_image
from pipeline.artifacts import export_result
from pipeline.runner import extract_saved


class JobManager:
    def __init__(self, input_directory, output_directory):
        self.inputs = Path(input_directory)
        self.outputs = Path(output_directory)
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
        self.jobs = {}
        for manifest in self.outputs.glob("*/run.json"):
            try:
                job = json.loads(manifest.read_text(encoding="utf-8"))
                if job["status"] in ("queued", "running"):
                    job.update(status="interrupted", stage="Server stopped; create a new run to retry.")
                self.jobs[job["id"]] = job
            except (OSError, ValueError, KeyError):
                continue

    def _save(self, job):
        target = self.outputs / job["id"] / "run.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
        temporary.replace(target)

    def snapshot(self, job_id=None):
        with self.lock:
            value = self.jobs.get(job_id) if job_id else sorted(
                self.jobs.values(), key=lambda job: job["created_at"], reverse=True)
            return json.loads(json.dumps(value))

    def submit(self, names, config: PipelineConfig, annotations=None, source_run_id=None):
        with self.lock:
            if any(job["status"] in ("queued", "running") for job in self.jobs.values()):
                raise RuntimeError("An experiment is already running. Wait for it to finish.")
            job = {"id": uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
                   "status": "queued", "stage": "queued", "files": names,
                   "settings": config.to_dict(), "annotations": annotations or {},
                   'source_run_id': source_run_id,
                   "completed": 0, "results": [], "errors": []}
            self.jobs[job["id"]] = job
            self._save(job)
        self.executor.submit(self._run, job["id"], config)
        return job["id"]

    def _update(self, job, **values):
        with self.lock:
            job.update(values)
            self._save(job)

    def _run(self, job_id, config):
        job = self.jobs[job_id]
        try:
            self._update(job, status="running")
            for index, name in enumerate(job["files"]):
                try:
                    def progress(stage):
                        self._update(job, stage=stage, current_file=name)

                    if job.get('source_run_id'):
                        source = self.snapshot(job['source_run_id'])
                        saved = next(result for result in source['results'] if result['name'] == name)
                        progress('extraction · loading saved segmentation')
                        with np.load(self.outputs / source['id'] / saved['folder'] / 'arrays.npz', allow_pickle=False) as archive:
                            cached = {key: archive[key] for key in archive.files}
                        arrays, metadata = extract_saved(cached, saved, config, progress)
                        metadata['source_run_id'] = source['id']
                    else:
                        arrays, metadata = process_image(self.inputs / name, config, progress,
                                                         annotation=job.get('annotations', {}).get(name))
                    progress("exporting")
                    folder = f"image_{index:03d}"
                    result = export_result(self.outputs / job_id / folder, arrays, metadata, config)
                    result.update(name=name, folder=folder)
                    with self.lock:
                        job["results"].append(result)
                except Exception as exc:
                    with self.lock:
                        job["errors"].append({"name": name, "message": str(exc)})
                self._update(job, completed=index + 1)
            status = "completed" if not job["errors"] else ("partial" if job["results"] else "failed")
            self._update(job, status=status, stage="finished", current_file=None)
        except Exception as exc:
            self._update(job, status="failed", stage=str(exc))
