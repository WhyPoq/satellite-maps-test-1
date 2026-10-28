"""HTTP adapters. Numerical processing lives in pipeline/ and the original scripts."""
from io import BytesIO
from pathlib import Path
import zipfile

from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory
from PIL import Image

from pipeline import PipelineConfig
from pipeline.seeds import seed_masks
from .jobs import JobManager

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def create_app(input_directory=None, output_directory=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    inputs = Path(input_directory or ROOT / "test_input").resolve()
    outputs = Path(output_directory or ROOT / "outputs").resolve()
    manager = JobManager(inputs, outputs)
    app.extensions["jobs"] = manager

    def catalog():
        items = []
        for path in sorted(inputs.iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS or path.is_symlink():
                continue
            try:
                with Image.open(path) as image:
                    width, height = image.size
                items.append({"name": path.name, "width": width, "height": height})
            except (OSError, ValueError):
                continue
        return items

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/images")
    def images():
        return jsonify(images=catalog(), defaults=PipelineConfig().to_dict())

    @app.get("/api/images/<name>/thumbnail")
    def thumbnail(name):
        if name not in {item["name"] for item in catalog()}:
            abort(404)
        with Image.open(inputs / name) as image:
            image.thumbnail((320, 220))
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=82)
        buffer.seek(0)
        return send_file(buffer, mimetype="image/jpeg", max_age=300)

    @app.get('/api/images/<name>/preview')
    def seed_preview(name):
        if name not in {item['name'] for item in catalog()}:
            abort(404)
        with Image.open(inputs / name) as image:
            image.thumbnail((1024, 1024))
            buffer = BytesIO()
            image.convert('RGB').save(buffer, format='PNG')
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png')

    @app.errorhandler(413)
    def payload_too_large(error):
        return jsonify(error='Annotations exceed 2 MB. Undo or clear some brush strokes.'), 413

    @app.get("/api/runs")
    def runs():
        return jsonify(runs=manager.snapshot())

    @app.post("/api/runs")
    def start_run():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Expected a JSON object."), 400
        try:
            config = PipelineConfig.from_dict(payload.get("settings", {}))
            names = payload.get("files")
            image_info = {item['name']: item for item in catalog()}
            available = set(image_info)
            if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
                raise ValueError("Select at least one test image.")
            if len(names) > len(available) or len(names) != len(set(names)) or any(n not in available for n in names):
                raise ValueError("Select unique images from test_input.")
            annotations = payload.get('annotations', {})
            if not isinstance(annotations, dict):
                raise ValueError('Annotations must be an object keyed by image filename.')
            source_run_id = payload.get('source_run_id')
            if source_run_id is not None:
                if not isinstance(source_run_id, str) or not config.extraction_enabled:
                    raise ValueError('A saved segmentation source requires network extraction to be enabled.')
                source = manager.snapshot(source_run_id)
                if not source or source['status'] not in ('completed', 'partial'):
                    raise ValueError('Choose a completed segmentation run.')
                saved_names = {r['name'] for r in source['results'] if r.get('segmentation')}
                if any(name not in saved_names for name in names):
                    raise ValueError('The source run must contain a successful segmentation for every selected image.')
                upstream = {key: value for key, value in config.to_dict().items()
                            if not key.startswith('net_') and key != 'extraction_enabled'}
                source_settings = PipelineConfig.from_dict(source['settings']).to_dict()
                if any(value != source_settings[key] for key, value in upstream.items()):
                    raise ValueError('Saved segmentation uses different upstream settings. Select the source again to restore them, or choose Full pipeline.')
                annotations = source.get('annotations', {})
            if config.segmentation_enabled and not source_run_id:
                for name in names:
                    item = image_info[name]
                    scale = min(1, config.max_dimension / max(item['height'], item['width']))
                    shape = (max(1, round(item['height'] * scale)), max(1, round(item['width'] * scale)))
                    try:
                        seed_masks(annotations.get(name), shape)
                    except ValueError as exc:
                        raise ValueError(f'{name}: {exc}') from exc
            job_id = manager.submit(names, config, {name: annotations[name] for name in names if name in annotations}, source_run_id)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 409
        return jsonify(id=job_id), 202

    @app.get("/api/runs/<job_id>")
    def run(job_id):
        job = manager.snapshot(job_id)
        if job is None:
            abort(404)
        return jsonify(job)

    @app.get("/outputs/<job_id>/<folder>/<filename>")
    def artifact(job_id, folder, filename):
        job = manager.snapshot(job_id)
        if not job or folder not in {r["folder"] for r in job["results"]}:
            abort(404)
        result = next(r for r in job["results"] if r["folder"] == folder)
        if filename not in {v["file"] for v in result["views"]} | {"arrays.npz", "result.json"} | {d['file'] for d in result.get('downloads', [])}:
            abort(404)
        mimetype = 'application/geo+json' if filename.endswith('.geojson') else 'application/json' if filename.endswith('.json') else None
        return send_from_directory(outputs / job_id / folder, filename, mimetype=mimetype)

    @app.get("/api/runs/<job_id>/download")
    def download(job_id):
        job = manager.snapshot(job_id)
        if not job or job["status"] in ("queued", "running"):
            abort(404)
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in (outputs / job_id).rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(outputs / job_id))
        buffer.seek(0)
        return send_file(buffer, mimetype="application/zip", as_attachment=True,
                         download_name=f"experiment-{job_id[:8]}.zip")

    @app.get("/paper")
    def paper():
        return send_file(ROOT / "paper.pdf", mimetype="application/pdf")

    return app
