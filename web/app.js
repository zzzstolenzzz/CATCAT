'use strict';

const CONFIG = window.CATCAT_CONFIG;

// --- State ---
let ortSession = null;
let images = [];         // [{file, name}]
let currentIndex = -1;
let currentImageEl = null;
let detectionBoxes = []; // [{x1,y1,x2,y2,conf}] normalized 0-1
let userBoxes = [];      // [{x1,y1,x2,y2}] normalized 0-1
let isDrawing = false;
let drawStart = null;
let sessionCount = 0;
let mousePos = null;

// Canvas layout (recomputed on each image load and resize)
let canvasOffsetX = 0, canvasOffsetY = 0;
let imgDisplayW = 0, imgDisplayH = 0;

// --- DOM refs ---
const imageCanvas = document.getElementById('image-canvas');
const imageCtx = imageCanvas.getContext('2d');
const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');
const hint = document.getElementById('canvas-hint');
const statusEl = document.getElementById('status-text');
const counterEl = document.getElementById('image-counter');
const sessionEl = document.getElementById('session-count');
const mapEl = document.getElementById('map-display');
const modelStatusEl = document.getElementById('model-status');
const acceptBtn = document.getElementById('accept-btn');
const clearBtn = document.getElementById('clear-btn');
const backBtn = document.getElementById('back-btn');
const skipBtn = document.getElementById('skip-btn');
const loadBtn = document.getElementById('load-btn');
const fileInput = document.getElementById('file-input');
const brightnessSlider = document.getElementById('brightness');
const contrastSlider = document.getElementById('contrast');
const sharpenSlider = document.getElementById('sharpen');
const sharpenKernel = document.getElementById('sharpen-kernel');

document.getElementById('theme-toggle').addEventListener('change', e => {
  document.body.dataset.theme = e.target.checked ? 'modern' : '';
});

function applyEnhance() {
  const b = brightnessSlider.value / 100;
  const c = contrastSlider.value / 100;
  const s = sharpenSlider.value / 100;
  const k = s * 0.8;
  sharpenKernel.setAttribute('kernelMatrix', `0 ${-k} 0 ${-k} ${1+4*k} ${-k} 0 ${-k} 0`);
  imageCanvas.style.filter = `brightness(${b}) contrast(${c})${s > 0 ? ' url(#sharpen-filter)' : ''}`;
}

[brightnessSlider, contrastSlider, sharpenSlider].forEach(s => s.addEventListener('input', applyEnhance));
document.getElementById('reset-enhance').addEventListener('click', () => {
  brightnessSlider.value = 100;
  contrastSlider.value = 100;
  sharpenSlider.value = 0;
  applyEnhance();
});

let _statusLocked = false;
const setStatus = (msg, type) => {
  if (_statusLocked && !type) return;
  statusEl.textContent = msg;
  statusEl.className = type || '';
  if (type) {
    _statusLocked = true;
    setTimeout(() => { _statusLocked = false; statusEl.className = ''; }, 4000);
  }
};

// --- Model loading ---
async function initModel() {
  modelStatusEl.textContent = 'Loading model…';
  modelStatusEl.className = '';
  try {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
    ort.env.wasm.numThreads = 1; // GitHub Pages lacks cross-origin isolation for multi-threading
    const urls = [CONFIG.teamModelUrl, CONFIG.modelUrl].filter(Boolean);
    let lastErr;
    for (const url of urls) {
      try {
        ortSession = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
        lastErr = null;
        break;
      } catch (e) { lastErr = e; }
    }
    if (lastErr) throw lastErr;
    modelStatusEl.textContent = 'Model ready';
    modelStatusEl.className = 'ready';
    setStatus('Ready');
  } catch (e) {
    modelStatusEl.textContent = 'Model unavailable';
    modelStatusEl.className = 'error';
    setStatus(`Model failed to load: ${e.message}`);
    console.error('ONNX load error:', e);
  }
}

// --- Preprocessing (letterbox, matching YOLOv8 training preprocessing) ---
let _letterbox = { padX: 0, padY: 0, scale: 1 };

function imageToTensor(img) {
  const size = CONFIG.inputSize;
  const scale = Math.min(size / img.naturalWidth, size / img.naturalHeight);
  const scaledW = Math.round(img.naturalWidth * scale);
  const scaledH = Math.round(img.naturalHeight * scale);
  const padX = Math.floor((size - scaledW) / 2);
  const padY = Math.floor((size - scaledH) / 2);
  _letterbox = { padX, padY, scale };

  const off = document.createElement('canvas');
  off.width = size; off.height = size;
  const offCtx = off.getContext('2d');
  offCtx.fillStyle = 'rgb(114,114,114)'; // YOLOv8 default pad color
  offCtx.fillRect(0, 0, size, size);
  offCtx.drawImage(img, padX, padY, scaledW, scaledH);

  const { data } = offCtx.getImageData(0, 0, size, size);
  const area = size * size;
  const t = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    t[i]          = data[i * 4]     / 255;
    t[area + i]   = data[i * 4 + 1] / 255;
    t[2*area + i] = data[i * 4 + 2] / 255;
  }
  return new ort.Tensor('float32', t, [1, 3, size, size]);
}

// --- NMS ---
function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2-x1) * Math.max(0, y2-y1);
  return inter / ((a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter);
}

function nms(boxes) {
  boxes.sort((a, b) => b.conf - a.conf);
  const keep = [], skip = new Set();
  for (let i = 0; i < boxes.length; i++) {
    if (skip.has(i)) continue;
    keep.push(boxes[i]);
    for (let j = i+1; j < boxes.length; j++) {
      if (!skip.has(j) && iou(boxes[i], boxes[j]) > CONFIG.iouThreshold) skip.add(j);
    }
  }
  return keep;
}

// --- Inference ---
async function detect(img) {
  if (!ortSession) return [];
  const results = await ortSession.run({ images: imageToTensor(img) });
  console.log('Output keys:', Object.keys(results));
  const outputKey = Object.keys(results)[0];
  const outputTensor = results[outputKey];
  console.log('Output shape:', outputTensor.dims);
  const raw = outputTensor.data;
  const totalElements = raw.length;
  // dims: [1, features, anchors] — infer anchors from shape
  const features = outputTensor.dims[1];
  const n = outputTensor.dims[2];
  console.log(`features=${features}, anchors=${n}`);
  const maxConf = Math.max(...Array.from(raw).slice(4 * n, 5 * n));
  console.log('Max confidence in output:', maxConf);
  const size = CONFIG.inputSize;
  const candidates = [];
  const { padX, padY, scale } = _letterbox;
  const scaledW = img.naturalWidth * scale;
  const scaledH = img.naturalHeight * scale;

  for (let i = 0; i < n; i++) {
    let conf = 0;
    for (let c = 4; c < features; c++) {
      if (raw[c * n + i] > conf) conf = raw[c * n + i];
    }
    if (conf <= CONFIG.confThreshold) continue;
    const cx = raw[i], cy = raw[n+i], w = raw[2*n+i], h = raw[3*n+i];
    // Reverse letterbox padding to get normalized image coords
    candidates.push({
      x1: Math.max(0, (cx - w/2 - padX) / scaledW),
      y1: Math.max(0, (cy - h/2 - padY) / scaledH),
      x2: Math.min(1, (cx + w/2 - padX) / scaledW),
      y2: Math.min(1, (cy + h/2 - padY) / scaledH),
      conf,
    });
  }
  return nms(candidates);
}

// --- Rendering ---
function computeLayout() {
  const container = canvas.parentElement;
  const cw = container.clientWidth, ch = container.clientHeight;
  const iw = currentImageEl.naturalWidth, ih = currentImageEl.naturalHeight;
  const scale = Math.min(cw / iw, ch / ih);
  imgDisplayW = iw * scale;
  imgDisplayH = ih * scale;
  canvasOffsetX = (cw - imgDisplayW) / 2;
  canvasOffsetY = (ch - imgDisplayH) / 2;
  canvas.width = cw; canvas.height = ch;
  imageCanvas.width = cw; imageCanvas.height = ch;
}

function render(preview = null) {
  if (!currentImageEl) return;
  imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
  imageCtx.drawImage(currentImageEl, canvasOffsetX, canvasOffsetY, imgDisplayW, imgDisplayH);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const drawBox = (b, color) => {
    const bx = canvasOffsetX + b.x1 * imgDisplayW;
    const by = canvasOffsetY + b.y1 * imgDisplayH;
    const bw = (b.x2 - b.x1) * imgDisplayW;
    const bh = (b.y2 - b.y1) * imgDisplayH;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.strokeRect(bx, by, bw, bh);

    const pad = 2 * 96; // 2 inches at 96 CSS px/inch
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(bx - pad, by - pad, bw + pad * 2, bh + pad * 2);
    ctx.setLineDash([]);
  };

  detectionBoxes.forEach(b => drawBox(b, '#00e676'));
  userBoxes.forEach(b => drawBox(b, '#ffd600'));

  if (preview) {
    ctx.strokeStyle = '#ff2020';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(preview.x, preview.y, preview.w, preview.h);
    ctx.setLineDash([]);
  }

  if (mousePos && currentImageEl) {
    ctx.strokeStyle = '#ff2020';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(mousePos.x, 0); ctx.lineTo(mousePos.x, canvas.height);
    ctx.moveTo(0, mousePos.y); ctx.lineTo(canvas.width, mousePos.y);
    ctx.stroke();
  }
}

// --- Canvas → normalized image coords ---
function toNorm(cx, cy) {
  return {
    x: (cx - canvasOffsetX) / imgDisplayW,
    y: (cy - canvasOffsetY) / imgDisplayH,
  };
}

// --- Mouse events ---
canvas.addEventListener('mousedown', e => {
  if (!currentImageEl) return;
  const r = canvas.getBoundingClientRect();
  drawStart = { x: e.clientX - r.left, y: e.clientY - r.top };
  isDrawing = true;
});

canvas.addEventListener('mousemove', e => {
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  mousePos = { x, y };
  if (isDrawing) {
    render({
      x: Math.min(drawStart.x, x), y: Math.min(drawStart.y, y),
      w: Math.abs(x - drawStart.x), h: Math.abs(y - drawStart.y),
    });
  } else {
    render();
  }
});

canvas.addEventListener('mouseleave', () => { mousePos = null; render(); });

canvas.addEventListener('mouseup', e => {
  if (!isDrawing) return;
  isDrawing = false;
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  const n1 = toNorm(Math.min(drawStart.x, x), Math.min(drawStart.y, y));
  const n2 = toNorm(Math.max(drawStart.x, x), Math.max(drawStart.y, y));
  drawStart = null;

  if (n2.x - n1.x > 0.01 && n2.y - n1.y > 0.01) {
    detectionBoxes = [];
    userBoxes = [{
      x1: Math.max(0, n1.x), y1: Math.max(0, n1.y),
      x2: Math.min(1, n2.x), y2: Math.min(1, n2.y),
    }];
  }
  render();
});

// --- Image loading ---
loadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async e => {
  const files = Array.from(e.target.files);
  if (!files.length) return;
  images = files.map(f => ({ file: f, name: f.name }));
  currentIndex = 0;
  hint.style.display = 'none';
  await loadCurrentImage();
  e.target.value = '';
});

async function loadCurrentImage() {
  if (currentIndex < 0 || currentIndex >= images.length) return;
  const img = images[currentIndex];
  detectionBoxes = [];
  userBoxes = [];
  setStatus(`Loading ${img.name}…`);

  const url = URL.createObjectURL(img.file);
  const el = new Image();
  el.src = url;
  await new Promise(res => { el.onload = res; });
  URL.revokeObjectURL(url);
  currentImageEl = el;

  computeLayout();
  render();

  counterEl.textContent = `Image ${currentIndex + 1} of ${images.length}`;
  acceptBtn.disabled = false;
  clearBtn.disabled = false;
  skipBtn.disabled = false;
  backBtn.disabled = currentIndex === 0;

  if (ortSession) {
    setStatus(`Detecting ships in ${img.name}…`);
    try {
      detectionBoxes = await detect(el);
      setStatus(`Found ${detectionBoxes.length} ship(s) in ${img.name}`);
    } catch (err) {
      setStatus(`Detection error: ${err.message}`);
      console.error(err);
    }
    render();
  } else {
    setStatus('Model not loaded — annotation-only mode');
  }
}

// --- Actions ---
acceptBtn.addEventListener('click', accept);
let acceptCooldown = false;
document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.repeat && !acceptCooldown) {
    e.preventDefault();
    acceptCooldown = true;
    accept();
    setTimeout(() => { acceptCooldown = false; }, 400);
  }
});

clearBtn.addEventListener('click', () => {
  userBoxes = [];
  detectionBoxes = [];
  render();
});

backBtn.addEventListener('click', () => {
  if (currentIndex > 0) { currentIndex--; loadCurrentImage(); }
});

skipBtn.addEventListener('click', advance);

async function accept() {
  if (!currentImageEl) return;
  const boxes = userBoxes.length > 0 ? userBoxes : detectionBoxes;
  sessionCount++;
  sessionEl.textContent = `Session: ${sessionCount}`;

  const corrected = userBoxes.length > 0;
  if (boxes.length > 0) await submitAnnotation(images[currentIndex].name, boxes, images[currentIndex].file, corrected);
  advance();
}

function advance() {
  if (currentIndex < images.length - 1) {
    currentIndex++;
    loadCurrentImage();
  } else {
    setStatus('All images reviewed. Great work!');
    acceptBtn.disabled = true;
    skipBtn.disabled = true;
  }
}

// --- Model version polling ---
let currentModelVersion = null;

async function pollStats() {
  try {
    const data = await (await fetch(`${CONFIG.backendUrl}/stats`)).json();
    if (data.map50 != null) updateMap(data.map50);
    if (data.training) setStatus('Model training in background…');
    if (data.model_version) {
      if (currentModelVersion && data.model_version !== currentModelVersion) {
        setStatus('New model ready — reloading…');
        currentModelVersion = data.model_version;
        await initModel();
        setStatus('Model updated!');
      } else {
        currentModelVersion = data.model_version;
      }
    }
  } catch (_) {}
}

setInterval(pollStats, 60000);

// --- Submit to HF Space ---
async function submitAnnotation(imageName, boxes, imageFile, corrected = false) {
  setStatus(`Submitting annotation for ${imageName}…`);
  try {
    const fd = new FormData();
    fd.append('image', imageFile, imageName);
    fd.append('boxes', JSON.stringify(boxes));
    fd.append('image_name', imageName);
    fd.append('corrected', corrected ? '1' : '0');

    const resp = await fetch(`${CONFIG.backendUrl}/team/annotate`, {
      method: 'POST', body: fd, headers: { 'X-Team-Key': CONFIG.teamKey },
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const trainingNote = data.training ? ' · training started…' : '';
    setStatus(`Saved — ${data.total_annotations} team annotations${trainingNote}`, 'status-ok');
    if (data.map50 != null) updateMap(data.map50);
    if (data.model_version) currentModelVersion = data.model_version;
  } catch (e) {
    setStatus(`Backend error: ${e.message}`, 'status-err');
  }
}

function updateMap(map50) {
  const prev = parseFloat(mapEl.dataset.value || '0');
  mapEl.textContent = `mAP50: ${(map50 * 100).toFixed(1)}%`;
  mapEl.dataset.value = map50;
  mapEl.className = map50 > prev ? 'improved' : map50 < prev ? 'declined' : '';
  setTimeout(() => { mapEl.className = ''; }, 2000);
}

// --- Fetch live stats on load ---
async function fetchStats() {
  try {
    const data = await (await fetch(`${CONFIG.backendUrl}/team/stats`,
      { headers: { 'X-Team-Key': CONFIG.teamKey } })).json();
    if (data.map50 != null) updateMap(data.map50);
  } catch (_) {}
}

window.addEventListener('resize', () => { if (currentImageEl) { computeLayout(); render(); } });

// Boot
initModel();
fetchStats();
