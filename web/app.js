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

// Canvas layout (recomputed on each image load and resize)
let canvasOffsetX = 0, canvasOffsetY = 0;
let imgDisplayW = 0, imgDisplayH = 0;

// --- DOM refs ---
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

const setStatus = msg => { statusEl.textContent = msg; };

// --- Model loading ---
async function initModel() {
  modelStatusEl.textContent = 'Loading model…';
  modelStatusEl.className = '';
  try {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
    ortSession = await ort.InferenceSession.create(CONFIG.modelUrl, {
      executionProviders: ['wasm'],
    });
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

// --- Preprocessing ---
function imageToTensor(img) {
  const size = CONFIG.inputSize;
  const off = document.createElement('canvas');
  off.width = size; off.height = size;
  const offCtx = off.getContext('2d');
  offCtx.drawImage(img, 0, 0, size, size);
  const { data } = offCtx.getImageData(0, 0, size, size);
  const area = size * size;
  const t = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    t[i]          = data[i * 4]     / 255; // R
    t[area + i]   = data[i * 4 + 1] / 255; // G
    t[2*area + i] = data[i * 4 + 2] / 255; // B
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
  const { data: raw } = (await ortSession.run({ images: imageToTensor(img) }))['output0'];
  const n = 8400, size = CONFIG.inputSize;
  const candidates = [];
  for (let i = 0; i < n; i++) {
    const conf = raw[4*n + i];
    if (conf <= CONFIG.confThreshold) continue;
    const cx = raw[i], cy = raw[n+i], w = raw[2*n+i], h = raw[3*n+i];
    candidates.push({
      x1: (cx - w/2) / size, y1: (cy - h/2) / size,
      x2: (cx + w/2) / size, y2: (cy + h/2) / size,
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
  canvas.width = cw;
  canvas.height = ch;
}

function render(preview = null) {
  if (!currentImageEl) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(currentImageEl, canvasOffsetX, canvasOffsetY, imgDisplayW, imgDisplayH);

  const drawBox = (b, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(
      canvasOffsetX + b.x1 * imgDisplayW,
      canvasOffsetY + b.y1 * imgDisplayH,
      (b.x2 - b.x1) * imgDisplayW,
      (b.y2 - b.y1) * imgDisplayH,
    );
  };

  detectionBoxes.forEach(b => drawBox(b, '#00e676'));
  userBoxes.forEach(b => drawBox(b, '#ffd600'));

  if (preview) {
    ctx.strokeStyle = '#ff9800';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(preview.x, preview.y, preview.w, preview.h);
    ctx.setLineDash([]);
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
  if (!isDrawing) return;
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  render({
    x: Math.min(drawStart.x, x), y: Math.min(drawStart.y, y),
    w: Math.abs(x - drawStart.x), h: Math.abs(y - drawStart.y),
  });
});

canvas.addEventListener('mouseup', e => {
  if (!isDrawing) return;
  isDrawing = false;
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  const n1 = toNorm(Math.min(drawStart.x, x), Math.min(drawStart.y, y));
  const n2 = toNorm(Math.max(drawStart.x, x), Math.max(drawStart.y, y));
  drawStart = null;

  if (n2.x - n1.x > 0.01 && n2.y - n1.y > 0.01) {
    detectionBoxes = []; // user correction replaces auto-detections
    userBoxes.push({
      x1: Math.max(0, n1.x), y1: Math.max(0, n1.y),
      x2: Math.min(1, n2.x), y2: Math.min(1, n2.y),
    });
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
document.addEventListener('keydown', e => { if (e.key === 'Enter') accept(); });

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

  if (boxes.length > 0) await submitAnnotation(images[currentIndex].name, boxes);
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

// --- Submit to HF Space ---
async function submitAnnotation(imageName, boxes) {
  setStatus(`Submitting annotation for ${imageName}…`);
  try {
    const resp = await fetch(`${CONFIG.backendUrl}/annotate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_name: imageName, boxes }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    setStatus(`Saved — ${data.total_annotations} total annotations in shared dataset`);
    if (data.map50 != null) updateMap(data.map50);
  } catch (e) {
    setStatus(`Saved locally (backend offline: ${e.message})`);
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
    const data = await (await fetch(`${CONFIG.backendUrl}/stats`)).json();
    if (data.map50 != null) updateMap(data.map50);
  } catch (_) {}
}

window.addEventListener('resize', () => { if (currentImageEl) { computeLayout(); render(); } });

// Boot
initModel();
fetchStats();
