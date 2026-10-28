const $ = (id) => document.getElementById(id);
const form = $('settings-form');
let defaults, currentRun = null, activeRun = false, selected = new Set(), imageCatalog = [];
let renderSignature = '';
let sourceRunId = null, historyRuns = [];

function syncStages() {
  if (defaults) for (const key of Object.keys(defaults)) {
    form.elements.namedItem(key).disabled = Boolean(sourceRunId) && !key.startsWith('net_') && key !== 'extraction_enabled';
  }
  $('segmentation-settings').disabled = !form.elements.segmentation_enabled.checked || Boolean(sourceRunId);
  $('extraction-settings').disabled = !form.elements.extraction_enabled.checked;
  $('extraction-source').value = sourceRunId || '';
}

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}
async function api(url, options) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status}).`);
  return body;
}
function setSettings(settings) {
  for (const [key, value] of Object.entries({...defaults, ...settings})) {
    const input = form.elements.namedItem(key);
    if (input) {
      if (input.type === 'checkbox') input.checked = value;
      else input.value = value;
    }
  }
  $('sigma-value').value = `${form.elements.sigma.value} px`;
  syncStages();
}
function readSettings() {
  return Object.fromEntries(Object.keys(defaults).map(key => {
    const input = form.elements.namedItem(key);
    return [key, input.type === 'checkbox' ? input.checked : typeof defaults[key] === 'string' ? input.value : Number(input.value)];
  }));
}
function rememberSettings() {
  try {
    localStorage.setItem('fieldwork-settings', JSON.stringify(readSettings()));
    localStorage.setItem('fieldwork-source-run', sourceRunId || '');
  } catch (_) { /* Optional persistence. */ }
}
function selectionChanged() {
  document.querySelectorAll('.image-card input').forEach(input => { input.checked = selected.has(input.value); });
  $('selection-count').textContent = selected.size ? `${selected.size} image${selected.size === 1 ? '' : 's'} selected` : 'Select images to begin';
  $('select-all').textContent = selected.size === imageCatalog.length && selected.size ? 'Deselect all' : 'Select all';
  $('run-button').disabled = activeRun || !selected.size;
}
function renderCatalog(images) {
  imageCatalog = images;
  $('image-count').textContent = images.length;
  $('image-list').replaceChildren();
  if (!images.length) $('image-list').append(element('p', 'subtle', 'Add images to test_input/ and reload this page.'));
  for (const image of images) {
    const card = element('div', 'image-card');
    const selection = element('label', 'image-selection');
    const thumbnail = element('img');
    thumbnail.src = `/api/images/${encodeURIComponent(image.name)}/thumbnail`;
    thumbnail.alt = image.name;
    thumbnail.loading = 'lazy';
    const input = element('input');
    input.type = 'checkbox'; input.value = image.name; input.setAttribute('aria-label', `Select ${image.name}`);
    input.addEventListener('change', () => {
      if (input.checked) selected.add(image.name); else selected.delete(image.name);
      selectionChanged();
    });
    const info = element('span', 'file-info');
    info.append(element('strong', '', image.name), element('small', '', `${image.width.toLocaleString()} × ${image.height.toLocaleString()} px`));
    selection.append(thumbnail, input, info);
    const mark = element('button', 'seed-button', 'Mark seeds'); mark.type = 'button';
    mark.dataset.image = image.name;
    mark.addEventListener('click', () => window.seedEditor.open(image.name));
    card.append(selection, mark); $('image-list').append(card);
  }
  window.seedEditor?.updateBadges();
  selectionChanged();
}
function artifactUrl(result, filename) {
  return `/outputs/${currentRun.id}/${result.folder}/${encodeURIComponent(filename)}`;
}
function renderResults() {
  $('results').replaceChildren();
  if (!currentRun) return;
  const mode = $('view-select').value;
  const overview = ['original', 'gabor_combined', 'saliency', 'curve_map', 'road_mask', 'segmentation_overlay', 'network_overlay', 'network_road_map'];
  for (const result of currentRun.results) {
    const group = element('article', 'result-group');
    const [width, height] = result.processed_size;
    group.append(element('h3', '', result.name));
    const meta = element('div', 'result-meta', `${result.original_size.join(' × ')} → ${width} × ${height} px · ${result.timings.total.toFixed(2)}s · ${result.curve_points.toLocaleString()} ridge pixels`);
    const raw = element('a', '', 'Raw arrays ↓'); raw.href = artifactUrl(result, 'arrays.npz'); raw.download = `${result.name}-arrays.npz`;
    meta.append(raw); group.append(meta);
    for (const download of result.downloads || []) {
      const link = element('a', '', `${download.label} ↓`);
      link.href = artifactUrl(result, download.file); link.download = download.file;
      meta.append(link);
    }
    if (result.segmentation) {
      const seg = result.segmentation;
      group.append(element('p', 'result-meta', `${seg.method} · ${(100 * seg.road_fraction).toFixed(1)}% road candidates · ${seg.cycles} cycles · ${seg.converged ? 'converged' : 'cycle limit reached; may need more cycles'} · energy ${seg.energy_trace[0].toFixed(2)} → ${seg.energy_trace.at(-1).toFixed(2)}`));
    }
    if (result.network_extraction) {
      const net = result.network_extraction;
      group.append(element('p', 'result-meta', `${net.centreline_count} centreline segments · ${net.total_length_px.toFixed(1)} px total length · ${net.polygon_area_px2.toFixed(1)} px² polygon area · ${net.curve_source === 'tensor' ? 'voted curves' : 'mask boundaries'} · ${net.keep_longest ? 'longest network' : 'all components'}`));
      for (const warning of net.warnings) group.append(element('p', 'subtle', warning));
    }
    const grid = element('div', `view-grid ${!['overview', 'bank'].includes(mode) ? 'single' : ''}`);
    const views = result.views.filter(view => mode === 'overview' ? overview.includes(view.key) : mode === 'bank' ? /^gabor_\d+$/.test(view.key) : mode === view.key);
    if (!views.length) grid.append(element('p', 'subtle', 'This output is unavailable in this run. Enable the required processing stage and run again.'));
    for (const view of views) {
      const card = element('figure', 'output-card');
      const button = element('button'); button.type = 'button'; button.setAttribute('aria-label', `Enlarge ${view.label} for ${result.name}`);
      const img = element('img'); img.src = artifactUrl(result, view.file); img.alt = `${result.name}: ${view.label}`; img.loading = 'lazy';
      button.append(img);
      button.addEventListener('click', () => {
        $('dialog-title').textContent = `${result.name} / ${view.label}`;
        $('dialog-description').textContent = view.description;
        $('dialog-image').src = img.src; $('dialog-image').alt = img.alt;
        $('dialog-download').href = img.src;
        $('dialog-download').download = `${result.name}-${view.file}`;
        $('image-dialog').showModal();
      });
      const caption = element('figcaption', '', view.label);
      caption.append(element('small', '', view.description));
      card.append(button, caption); grid.append(card);
    }
    group.append(grid); $('results').append(group);
  }
}
function showRun(job) {
  currentRun = job;
  activeRun = ['queued', 'running'].includes(job.status);
  $('history').disabled = activeRun;
  $('download').hidden = activeRun;
  $('download').href = `/api/runs/${job.id}/download`;
  $('progress-panel').hidden = !activeRun;
  const stageLabels = {queued: 'Queued', preprocessing: 'Preparing input', gabor: 'Gabor filter bank', tensor_voting: 'Tensor voting', segmentation: 'Seeded road segmentation', exporting: 'Saving outputs'};
  $('progress-text').textContent = `${stageLabels[job.stage] || job.stage}${job.current_file ? ` · ${job.current_file}` : ''}`;
  $('progress-number').textContent = `${job.completed} / ${job.files.length}`;
  $('progress').max = job.files.length; $('progress').value = job.completed;
  $('run-info').hidden = false;
  const settings = job.settings;
  $('run-info').textContent = `${job.status.toUpperCase()} · ${new Date(job.created_at).toLocaleString()} · ${settings.n_orient} orientations × ${settings.n_scale} scales · f=${settings.ul}–${settings.uh} · σ=${settings.sigma} · n=${settings.n_angular} · max ${settings.max_dimension}px`;
  $('run-info').textContent += settings.segmentation_enabled ? ` · segmentation: ${settings.segmentation_orientation ? 'orientation-guided' : 'color only'}, λ=${settings.seg_smoothness}` : ' · feature inference only';
  if (settings.extraction_enabled) $('run-info').textContent += ` · network extraction${job.source_run_id ? ' from saved segmentation' : ''}`;
  $('run-errors').textContent = job.errors.map(error => `${error.name}: ${error.message}`).join('\n');
  if (job.status === 'interrupted') $('run-errors').textContent += 'This run was interrupted by a server restart. Select its inputs and run again.';
  if (job.status === 'failed' && !job.errors.length) $('run-errors').textContent = job.stage;
  $('empty').hidden = activeRun || job.results.length > 0 || ['failed', 'interrupted'].includes(job.status);
  $('result-controls').hidden = !job.results.length;
  const signature = `${job.id}/${job.results.length}`;
  if (signature !== renderSignature) { renderSignature = signature; renderResults(); }
  selectionChanged();
}
async function loadHistory(preferred) {
  const {runs} = await api('/api/runs');
  historyRuns = runs;
  $('extraction-source').replaceChildren(new Option('Full pipeline · compute segmentation', ''));
  for (const job of runs.filter(run => ['completed', 'partial'].includes(run.status) && run.results.some(result => result.segmentation))) {
    const names = job.results.filter(result => result.segmentation).map(result => result.name).join(', ');
    $('extraction-source').append(new Option(`${new Date(job.created_at).toLocaleString()} · ${names}`, job.id));
  }
  $('extraction-source').value = sourceRunId || '';
  $('history').replaceChildren(new Option('Run history', ''));
  for (const job of runs) {
    $('history').append(new Option(`${new Date(job.created_at).toLocaleString()} · ${job.files.length} images · ${job.status}`, job.id));
  }
  if (preferred) $('history').value = preferred;
  return runs;
}
async function watchRun(id) {
  // One outstanding poll at a time, even on long processing runs.
  try {
    const job = await api(`/api/runs/${id}`);
    showRun(job);
    if (['queued', 'running'].includes(job.status)) {
      setTimeout(() => watchRun(id), 1000);
    } else {
      await loadHistory(id);
    }
  } catch (error) {
    $('form-error').textContent = `Progress connection lost: ${error.message} Retrying…`;
    setTimeout(() => { $('form-error').textContent = ''; watchRun(id); }, 4000);
  }
}
form.addEventListener('input', event => {
  if (event.target.id === 'extraction-source') return;
  $('sigma-value').value = `${form.elements.sigma.value} px`;
  if (event.target.name === 'extraction_enabled') {
    if (event.target.checked) form.elements.segmentation_enabled.checked = true;
    else sourceRunId = null;
  }
  if (event.target.name === 'segmentation_enabled' && !event.target.checked) {
    form.elements.extraction_enabled.checked = false; sourceRunId = null;
  }
  syncStages();
  if (defaults) rememberSettings();
});
$('reset').addEventListener('click', () => { sourceRunId = null; setSettings(defaults); rememberSettings(); });
$('extraction-source').addEventListener('change', () => {
  sourceRunId = $('extraction-source').value || null;
  if (sourceRunId) {
    const source = historyRuns.find(job => job.id === sourceRunId);
    const extraction = Object.fromEntries(Object.entries(readSettings()).filter(([key]) => key.startsWith('net_')));
    setSettings({...source.settings, ...extraction, extraction_enabled: true});
    selected = new Set(source.results.filter(result => result.segmentation && imageCatalog.some(image => image.name === result.name)).map(result => result.name));
    window.seedEditor.restore(source.annotations || {});
    selectionChanged();
  }
  syncStages(); rememberSettings();
});
$('select-all').addEventListener('click', () => {
  selected = selected.size === imageCatalog.length ? new Set() : new Set(imageCatalog.map(image => image.name));
  selectionChanged();
});
$('view-select').addEventListener('change', renderResults);
$('close-dialog').addEventListener('click', () => $('image-dialog').close());
$('image-dialog').addEventListener('click', event => { if (event.target === $('image-dialog')) $('image-dialog').close(); });
$('history').addEventListener('change', async () => {
  if (!$('history').value) return;
  try {
    const job = await api(`/api/runs/${$('history').value}`);
    // Loading a run also restores its inputs/settings, ready to tweak and repeat.
    sourceRunId = job.source_run_id || null;
    setSettings(job.settings); rememberSettings();
    window.seedEditor.restore(job.annotations || {});
    selected = new Set(job.files.filter(name => imageCatalog.some(image => image.name === name)));
    showRun(job);
  } catch (error) { $('form-error').textContent = error.message; }
});
form.addEventListener('submit', async event => {
  event.preventDefault();
  $('form-error').textContent = '';
  if (!selected.size || activeRun || !form.reportValidity()) return;
  $('run-button').disabled = true;
  try {
    const {id} = await api('/api/runs', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({files: [...selected], settings: readSettings(), annotations: window.seedEditor.annotations([...selected]), source_run_id: sourceRunId})});
    activeRun = true;
    await loadHistory(id);
    watchRun(id);
  } catch (error) { $('form-error').textContent = error.message; selectionChanged(); }
});
async function initialize() {
  try {
    const data = await api('/api/images'); defaults = data.defaults;
    setSettings(defaults);
    try {
      const saved = JSON.parse(localStorage.getItem('fieldwork-settings'));
      sourceRunId = localStorage.getItem('fieldwork-source-run') || null;
      if (saved && typeof saved === 'object') setSettings({...defaults, ...saved});
    } catch (_) { /* Start from defaults if browser storage is unavailable. */ }
    renderCatalog(data.images);
    const runs = await loadHistory();
    if (sourceRunId && !runs.some(run => run.id === sourceRunId && ['completed', 'partial'].includes(run.status))) sourceRunId = null;
    syncStages();
    const job = runs.find(run => ['queued', 'running'].includes(run.status)) || runs[0];
    if (job) {
      $('history').value = job.id;
      showRun(job);
      if (activeRun) watchRun(job.id);
    }
  } catch (error) { $('form-error').textContent = `Could not load the workspace: ${error.message}`; }
}
for (const [key, label] of [['centreline_magnitude', 'Extraction · centreline response'],
  ['centreline_orientation', 'Extraction · centreline orientation'], ['centreline_width', 'Extraction · road widths'],
  ['network_centrelines', 'Extraction · Hough centrelines'], ['network_mask', 'Extraction · polygon footprint'],
  ['network_overlay', 'Extraction · final network'], ['network_road_map', 'Extraction · schematic map']]) {
  $('view-select').append(new Option(label, key));
}
initialize();
