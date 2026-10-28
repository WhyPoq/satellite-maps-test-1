/* Seed painting is a UI adapter; numerical seed masks are built in pipeline/seeds.py. */
(() => {
  let annotations = {}, name = null, image = null, currentStroke = null;
  const canvas = document.getElementById('seed-canvas');
  const ctx = canvas.getContext('2d');
  const get = id => document.getElementById(id);
  try { annotations = JSON.parse(localStorage.getItem('fieldwork-seeds')) || {}; } catch (_) { /* Optional persistence. */ }

  function persist() {
    try { localStorage.setItem('fieldwork-seeds', JSON.stringify(annotations)); } catch (_) { /* Run manifests also save seeds. */ }
    updateBadges();
  }
  function strokes() { return annotations[name]?.strokes || []; }
  function updateBadges() {
    document.querySelectorAll('.seed-button').forEach(button => {
      const marks = annotations[button.dataset.image]?.strokes || [];
      const road = marks.some(stroke => stroke.kind === 'road');
      const bg = marks.some(stroke => stroke.kind === 'background');
      button.textContent = road && bg ? `Edit seeds · ${marks.length} strokes` : 'Mark seeds' + (marks.length ? ' · incomplete' : '');
    });
  }
  function redraw() {
    if (!image) return;
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    for (const stroke of strokes()) {
      const radius = Math.max(1, Math.round(stroke.radius * Math.min(canvas.width, canvas.height)));
      ctx.strokeStyle = ctx.fillStyle = stroke.kind === 'road' ? '#f5463c' : '#2d78f5';
      ctx.lineWidth = 2 * radius + 1; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.beginPath();
      stroke.points.forEach(([x, y], index) => {
        const px = x * (canvas.width - 1), py = y * (canvas.height - 1);
        if (!index) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
      for (const [x, y] of stroke.points) {
        ctx.beginPath(); ctx.arc(x * (canvas.width - 1), y * (canvas.height - 1), radius, 0, 2 * Math.PI); ctx.fill();
      }
    }
    const road = strokes().filter(stroke => stroke.kind === 'road').length;
    const bg = strokes().length - road;
    get('seed-status').textContent = `${road} road strokes · ${bg} background strokes. Both classes need at least 5 pixels after resizing. Marks save automatically.`;
  }
  function point(event) {
    const box = canvas.getBoundingClientRect();
    return [Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)),
            Math.max(0, Math.min(1, (event.clientY - box.top) / box.height))];
  }
  function finishStroke() { currentStroke = null; persist(); }
  canvas.addEventListener('pointerdown', event => {
    if (!image || event.button !== 0) return;
    event.preventDefault(); canvas.setPointerCapture(event.pointerId);
    currentStroke = {kind: get('seed-kind').value, radius: Number(get('seed-radius').value), points: [point(event)]};
    annotations[name].strokes.push(currentStroke); redraw();
  });
  canvas.addEventListener('pointermove', event => {
    if (!currentStroke) return;
    const next = point(event), last = currentStroke.points.at(-1);
    if (Math.hypot(next[0] - last[0], next[1] - last[1]) < 0.002) return;
    currentStroke.points.push(next); redraw();
  });
  canvas.addEventListener('pointerup', finishStroke);
  canvas.addEventListener('pointercancel', finishStroke);
  get('undo-seed').addEventListener('click', () => { strokes().pop(); redraw(); persist(); });
  get('clear-seeds').addEventListener('click', () => { annotations[name] = {strokes: []}; redraw(); persist(); });
  get('close-seeds').addEventListener('click', () => get('seed-dialog').close());
  get('save-seeds').addEventListener('click', () => get('seed-dialog').close());
  get('seed-dialog').addEventListener('close', finishStroke);
  window.seedEditor = {
    async open(filename) {
      name = filename; image = null; currentStroke = null;
      annotations[name] ||= {strokes: []};
      get('seed-title').textContent = `${name} / segmentation seeds`;
      get('seed-error').textContent = ''; get('seed-status').textContent = 'Loading image…';
      canvas.hidden = true;
      get('seed-dialog').showModal();
      try {
        const loaded = new Image();
        loaded.src = `/api/images/${encodeURIComponent(name)}/preview`;
        await loaded.decode();
        image = loaded; canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
        canvas.hidden = false; redraw();
      } catch (error) { get('seed-error').textContent = `Could not load image: ${error.message}`; }
    },
    annotations(names) { return Object.fromEntries(names.filter(file => annotations[file]).map(file => [file, annotations[file]])); },
    restore(saved) { annotations = {...annotations, ...saved}; persist(); },
    updateBadges
  };
  updateBadges();
})();
