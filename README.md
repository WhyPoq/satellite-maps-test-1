# Fieldwork — local image-processing workbench

A local website for feature inference (§4), seeded segmentation (§5 prototype), and automatic network extraction (§6) of Poullis & You (2010), *Delineation and geometric modeling of road networks*, in `paper.pdf`:

**Grayscale → Gabor bank → tensor encoding → dense tensor voting → curve/junction classification → curve ridges → optional seeded segmentation → optional road network extraction.**

## Run on Windows

Dependencies have been installed into the existing `.venv`. From this directory:

```powershell
.\.venv\Scripts\python.exe run.py
```

Open **http://127.0.0.1:5000**. Stop with Ctrl+C. For another port, use `run.py --port 5050`.

To reinstall dependencies using uv (without recreating the environment):

```powershell
uv --cache-dir .uv-cache pip install --python .venv/Scripts/python.exe -r requirements.txt
```

1. Select multiple images from `test_input/`. Add more PNG, JPEG, BMP, TIFF, or WebP files there and reload the page.
2. Adjust filter count, frequency bounds, voting sigma, angular order, and processing resolution.
3. Run the experiment. Images run sequentially in a background thread; progress and completed outputs appear while later images run.
4. Choose an output view to compare images, inspect the orientation bank, or click a preview to enlarge it.
5. Download individual PNGs, raw `.npz` arrays, or the complete run ZIP. Select a saved run to restore its inputs/settings for another experiment.

### Segmentation workflow

1. Click **Mark seeds** for each selected image. Paint examples **inside roads in red**, then change the paint class to **background** and mark non-road areas in blue. Seeds are hard constraints and training samples, not a drawing of the entire road network. Both classes require at least five pixels at the processing resolution.
2. Use **Undo stroke**, **Clear seeds**, and **Save & close** as needed. Later strokes replace earlier marks. Coordinates and brush radii are normalized to image size, so seeds follow resolution changes. Marks also persist in browser storage and saved run manifests.
3. Enable **road segmentation** under Parameters. Start with **256 px** and the default settings. Graph cuts run at the same resolution as feature inference; full-resolution experiments can be substantially slower.
4. Leave orientation guidance enabled for the supplied prototype, or disable it for a **color-only baseline** with identical seeds. Tune angle bins, smoothness, orientation weight, direction boost, color sigma, variance floor, maximum expansion cycles, and integer capacity scale.
5. Inspect the **road mask**, **segmentation overlay**, **segmented road orientation**, and **seed overlay**. The result summary reports foreground fraction, cycles, convergence, and initial/final energy. A cycle-limit message means the solver may benefit from more iterations.

No extra packages are needed for segmentation: it uses the existing NumPy/SciPy installation. Segmentation is disabled by default, and old feature-only runs remain readable. Selecting a saved segmentation run restores its settings and brush annotations.

### Network extraction workflow

1. Enable **Network extraction** under Parameters. Segmentation is enabled automatically because extraction needs its road mask.
2. Choose **Segmentation input**:
   - **Full pipeline** computes all enabled stages using your selected images and seeds.
   - A **saved segmentation run** reuses its Stage 1/2 arrays. Its images and upstream settings are restored and held fixed. Only extraction is computed; Gabor, voting, and graph cuts are skipped. Select Full pipeline to change upstream settings again.
3. Set road width **minimum / maximum / step**, the number of filter orientations, and the along/across-road Gaussian sigmas. All widths and sigmas are in **processed-image pixels**. Defaults are 10–60 in steps of 4 (10, 14, …, 58), 12 orientations, σ_across=1.5, σ_along=6. Tune smaller widths for narrow roads at 256 px.
4. Choose **Stage 1 voted curve ridges** for the complete pipeline’s curve evidence, or **Stage 2 mask boundaries** to match the supplied script’s standalone CLI. These inputs can produce different results. The adapter deliberately exposes the choice.
5. Keep **longest connected network** enabled for the script’s automatic behavior, or disable it to retain disconnected components.
6. Inspect centreline filter magnitude/orientation/width, Hough centrelines, polygon footprint, final network overlay, and schematic road map. The summary reports segment count, length, and polygon area. Empty detections produce an explicit warning rather than a failed run.
7. Download **Network JSON**, **Pixel-coordinate GeoJSON**, raw arrays, or the complete run ZIP. Coordinates are `(x,y)` in the processed image, with origin at its top-left and y increasing downwards. These are **not georeferenced GIS coordinates**. Multiply x/y by original_width/processed_width and original_height/processed_height respectively to map them back to the original raster. Physical lengths/areas need external georeferencing.

Extraction dependencies **Shapely and Matplotlib** are installed in the existing `.venv` and listed in `requirements.txt`. The numerical extraction adapter does not invoke the script’s CLI or write figures itself; rendering and serialization stay in the artifact layer.

The website binds to localhost. This is a local research tool, with no external hosting or image uploads. One experiment runs at a time. Closing the tab does not cancel processing. Restarted runs are marked interrupted rather than silently resumed.

## Organization

```text
gabor.py                 Existing Gabor implementation; configurable frequency bounds
tensor_voting.py         Existing tensor encoding/voting/classification; configurable angular order
segmentation.py          Seeded Gaussian color model + alpha-expansion graph-cut prototype
road_network_extraction.py  Supplied Section 6 filters, Hough extraction, tracking and refinement
pipeline/config.py       Validated, serializable settings
pipeline/runner.py       Preprocessing and numerical pipeline, without UI or HTTP
pipeline/segmentation_stage.py  RGB / tensor-output adapter for segmentation
pipeline/seeds.py        Normalized brush validation and seed-mask rasterization
pipeline/extraction_stage.py   Automatic extraction and polygon-raster adapter
pipeline/artifacts.py    Preview rendering and raw-array exports
web/app.py               Flask routes, input catalog, downloads
web/jobs.py              Background processing and run manifests
web/templates/           HTML interface
web/static/              Browser interactions and styles
tests/                   Numerical and HTTP integration checks
outputs/<run-id>/        Saved settings, metadata, previews and raw arrays (ignored by Git)
```

Segmentation consumes RGB input, `theta`, and `curve_map` through a UI-independent adapter. Extraction consumes `road_mask` plus either the original `curve_map`/`theta` or mask-edge evidence. No Flask imports are needed. Add further processing stages in `pipeline/`, with previews registered separately in `artifacts.py`.

Use the pipeline directly:

```python
from pipeline import PipelineConfig, process_image

arrays, metadata = process_image(
    'test_input/fig3b_copper.png',
    PipelineConfig(n_orient=8, n_scale=5, sigma=12, max_dimension=768),
)
```

For direct segmentation, pass `segmentation_enabled=True` and an annotation:

```python
annotation = {'strokes': [
    {'kind': 'road', 'radius': 0.01, 'points': [[0.4, 0.5], [0.5, 0.5]]},
    {'kind': 'background', 'radius': 0.01, 'points': [[0.1, 0.1], [0.2, 0.1]]},
]}
# These coordinates illustrate the format; choose locations appropriate to your image.
arrays, metadata = process_image(
    'test_input/fig3b_copper.png',
    PipelineConfig(segmentation_enabled=True, max_dimension=256),
    annotation=annotation,
)
```

## Method and interpretation

- The paper specifies **8 orientations × 5 frequency scales**, summing frequency responses within each orientation. The default frequencies **0.05–0.4 cycles/pixel** come from the existing code; these bounds are not specified in the paper.
- The existing code uses **2×2 tensors and steerable FFT voting**, approximating the paper's 3D tensor formulation and Eq. 8 decay. It casts stick-only votes, discarding isotropic Gabor responses as voters. This behavior is preserved, not an exact paper reproduction.
- **Sigma 12** and **angular order 8** are implementation choices. The paper does not report a numeric sigma or use this angular-order parameter.
- Classification compares curve and junction saliencies without a user threshold. Ridge extraction additionally applies the existing normal-direction non-maximum suppression and a **0.1% of maximum** numerical floor. Feature maps are not road segmentation or road centerlines.
- Images are converted to 8-bit BGR/grayscale through the existing OpenCV path. This tool is intended for ordinary test imagery, not radiometrically calibrated or georeferenced raster analysis.
- Aspect ratio is preserved and small images are never enlarged. Sigma and frequencies refer to **processed pixels**. Changing resolution changes their scale relative to the original scene. Metadata records original and processed dimensions.
- Green/blue saliency previews use a shared linear maximum (green = curves, blue = junctions). Single saliency maps are independently scaled to their maxima for display. Gabor orientation previews share a common bank normalization. Preview brightness is not an absolute comparison between different input images.
- Raw `.npz` files retain floating-point numerical maps rather than preview intensities. `original` is BGR uint8, `gray` is float32 in [0,1], `gabor_responses` has shape `(orientations, height, width)`, and `theta` contains tangent angles in radians in [0,π). Boolean masks are stored without display conversion. Runs include all settings, timings, and metadata.

### What is missing from an exact paper implementation in segmentation.py?

The supplied file is a runnable **Section-5-inspired prototype**, not a full reproduction:

- **Color models:** it fits one diagonal Gaussian to each class. The paper uses Gaussian mixture models with the number of components selected by a **Minimum Description Length (MDL)** estimator (§5.3). Multimodal road/background colors may therefore be modeled poorly by this prototype.
- **Energy terms:** it uses an additive axial-angle unary penalty and a contrast-weighted Potts pairwise term with a directional continuity boost. These are **different from Eqs. 11–16** in the paper, particularly its labeling-dependent smoothness measure. Alpha expansion optimizes the prototype energy, not the exact published energy.
- **Input preparation:** the original file expects an RGB image, tangent angles, confidence, and two boolean seed masks; it does not generate them. The adapter now converts BGR to RGB, passes tangent angles, uses normalized curve saliency **only on classified curve ridges** as relative confidence, and rasterizes the user annotations. The paper itself also requires manual foreground/background seeds.
- **Convergence:** the iteration limit can stop optimization early. The website exposes the limit and reports convergence and the monotonically decreasing accepted energy trace. Expansion moves use rounded integer capacities; the original floating-point energy is checked before accepting a move. This does not guarantee a global optimum of the multi-label energy.

The integration preserves the prototype model. Small additions to `segmentation.py` validate finite/configuration values, check capacity overflow before integer conversion, and report per-label progress.

Additional raw exports are `road_mask` (bool), `road_seeds`, `background_seeds`, `segmentation_confidence`, `segment_labels` (int16), `segment_orientation_rad` (float32, NaN outside roads or when orientation is disabled), and `segment_orientation_bin` (int16, -1 outside roads). Label IDs are **B0…B(k−1), R0…R(k−1)**; color-only mode has k=1. Seed coordinates, solver settings, accepted energy trace, timings, and convergence are saved in JSON. Segmentation previews are road **candidates**, not validated road networks or extracted centerlines.

### Extraction implementation notes

The supplied automatic filters, weighted iterative Hough transform, width/orientation sampling, polygon tracking/union, and longest-network selection are retained. The script exposes optional seed/add/edit/merge/delete/smooth/reduce **manual refinement actions through its CLI**; the website currently runs automatic extraction and exports centreline IDs for that workflow. It does not include a graphical centreline-refinement editor.

The script makes assumptions about the single-mode filter (across-road sigma = width/4), Hough discretization (180 normal angles), and connectivity tolerances. These are implementation conventions, not all specified by the paper. Its docstring references `PLAN.md` and `evaluate_road_network.py`, which were not included with this repository; quantitative network evaluation is not part of the website. Extraction quality remains dependent on segmentation and the chosen curve evidence and width range.

The integration adds optional centreline-filter progress reporting and fixes an empty-network case in `longest_network` (an empty Shapely spatial query otherwise raised an error).

Extraction array exports are `centreline_magnitude`, `centreline_orientation` (radians), `centreline_width` (pixels), `network_centrelines` (bool), `network_mask` (bool), `network_road_map` (BGR uint8), and `extraction_curve_saliency`. Variable-length vectors are stored in `network.json` and `network.geojson`, avoiding pickle/object arrays in `.npz`. JSON includes the coordinate system and processed dimensions. Polygon footprints can extend beyond the segmentation mask because they are tracked from estimated line widths rather than clipped to the segmentation.

When extraction reuses a saved run, `timings` reports only the new extraction work. Earlier timings are retained as `source_timings`, and `source_run_id` records provenance. Cached inputs are snapshots of the original experiment, even if a same-named test file is later replaced.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Checks cover tensor behavior, graph-cut expansion against exhaustive small problems, synthetic seed-preserving segmentation, known-width centreline/polygon extraction, empty and disconnected networks, invalid inputs, multi-image full-pipeline HTTP runs, vector/raw exports, history recovery, and verification that cached extraction never reruns upstream processing.

Generated experiments remain in `outputs/` until you delete their run folders. High resolution and larger filter banks require substantially more memory and processing time; start with 512 or 768 pixels for feature inference and **256 pixels when tuning segmentation**.
