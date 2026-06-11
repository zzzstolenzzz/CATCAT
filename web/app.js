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

let _dirHandle = null; // FileSystemDirectoryHandle for local saves
let _autoContrast = false;
let _autoColor = false;
let _processedCanvas = null; // offscreen canvas with auto corrections baked in

const RULER_SZ = 28; // px reserved for rulers on top and left

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

document.getElementById('auto-contrast')?.addEventListener('change', e => {
  _autoContrast = e.target.checked;
  reprocessImage();
});
document.getElementById('auto-color')?.addEventListener('change', e => {
  _autoColor = e.target.checked;
  reprocessImage();
});

function applyAutoContrast(imageData) {
  const d = imageData.data, n = d.length / 4;
  const lums = new Float32Array(n);
  for (let i = 0; i < n; i++)
    lums[i] = d[i*4]*0.299 + d[i*4+1]*0.587 + d[i*4+2]*0.114;
  lums.sort();
  const clip = Math.max(1, Math.floor(n * 0.02)); // 2% clip — aggressive enough for wide-range satellite images
  const lo = lums[clip], hi = lums[n - 1 - clip];
  if (hi <= lo) return imageData;
  const scale = 255 / (hi - lo);
  for (let i = 0; i < d.length; i += 4) {
    d[i]   = Math.min(255, Math.max(0, (d[i]   - lo) * scale));
    d[i+1] = Math.min(255, Math.max(0, (d[i+1] - lo) * scale));
    d[i+2] = Math.min(255, Math.max(0, (d[i+2] - lo) * scale));
  }
  return imageData;
}

function applyAutoColor(imageData) {
  const d = imageData.data, n = d.length / 4;
  const rs = new Float32Array(n), gs = new Float32Array(n), bs = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    rs[i] = d[i*4]; gs[i] = d[i*4+1]; bs[i] = d[i*4+2];
  }
  rs.sort(); gs.sort(); bs.sort();
  const clip = Math.max(1, Math.floor(n * 0.005));
  const rLo = rs[clip], rHi = rs[n-1-clip];
  const gLo = gs[clip], gHi = gs[n-1-clip];
  const bLo = bs[clip], bHi = bs[n-1-clip];
  for (let i = 0; i < d.length; i += 4) {
    d[i]   = rHi > rLo ? Math.min(255, Math.max(0, (d[i]   - rLo) * 255 / (rHi - rLo))) : d[i];
    d[i+1] = gHi > gLo ? Math.min(255, Math.max(0, (d[i+1] - gLo) * 255 / (gHi - gLo))) : d[i+1];
    d[i+2] = bHi > bLo ? Math.min(255, Math.max(0, (d[i+2] - bLo) * 255 / (bHi - bLo))) : d[i+2];
  }
  return imageData;
}

function reprocessImage() {
  if (!currentImageEl) return;
  _autoContrast = document.getElementById('auto-contrast')?.checked ?? false;
  _autoColor    = document.getElementById('auto-color')?.checked ?? false;
  _processedCanvas = null;
  if (_autoContrast || _autoColor) {
    const w = Math.round(imgDisplayW), h = Math.round(imgDisplayH);
    if (w > 0 && h > 0) {
      try {
        const pc = document.createElement('canvas');
        pc.width = w; pc.height = h;
        const pctx = pc.getContext('2d', { willReadFrequently: true });
        pctx.drawImage(currentImageEl, 0, 0, w, h);
        let id = pctx.getImageData(0, 0, w, h);
        if (_autoContrast) id = applyAutoContrast(id);
        if (_autoColor)    id = applyAutoColor(id);
        pctx.putImageData(id, 0, 0);
        _processedCanvas = pc;
      } catch(e) { console.warn('reprocessImage failed:', e); }
    }
  }
  render();
}

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
  const availW = cw - RULER_SZ, availH = ch - RULER_SZ;
  const scale = Math.min(availW / iw, availH / ih);
  imgDisplayW = iw * scale;
  imgDisplayH = ih * scale;
  canvasOffsetX = RULER_SZ + (availW - imgDisplayW) / 2;
  canvasOffsetY = RULER_SZ + (availH - imgDisplayH) / 2;
  canvas.width = cw; canvas.height = ch;
  imageCanvas.width = cw; imageCanvas.height = ch;
}

function render(preview = null) {
  if (!currentImageEl) return;
  imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
  if (_processedCanvas) {
    imageCtx.drawImage(_processedCanvas, canvasOffsetX, canvasOffsetY, imgDisplayW, imgDisplayH);
  } else {
    imageCtx.drawImage(currentImageEl, canvasOffsetX, canvasOffsetY, imgDisplayW, imgDisplayH);
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const imgLeft   = canvasOffsetX;
  const imgTop    = canvasOffsetY;
  const imgRight  = canvasOffsetX + imgDisplayW;
  const imgBottom = canvasOffsetY + imgDisplayH;

  const drawBox = (b, color) => {
    const bx = imgLeft + b.x1 * imgDisplayW;
    const by = imgTop  + b.y1 * imgDisplayH;
    const bw = (b.x2 - b.x1) * imgDisplayW;
    const bh = (b.y2 - b.y1) * imgDisplayH;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.strokeRect(bx, by, bw, bh);

    // Dotted expansion box, clamped to image bounds
    const pad = 96; // 1 inch at 96 CSS px/inch
    const dx1 = Math.max(imgLeft,   bx - pad);
    const dy1 = Math.max(imgTop,    by - pad);
    const dx2 = Math.min(imgRight,  bx + bw + pad);
    const dy2 = Math.min(imgBottom, by + bh + pad);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(dx1, dy1, dx2 - dx1, dy2 - dy1);
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

  drawRulers(imgLeft, imgTop, imgDisplayW, imgDisplayH);
}

function drawRulers(imgLeft, imgTop, imgW, imgH) {
  const INCH = 96;
  const SZ   = RULER_SZ;

  ctx.save();

  // Backgrounds outside the image
  ctx.fillStyle = 'rgba(15,18,28,0.92)';
  ctx.fillRect(imgLeft - SZ, imgTop - SZ, imgW + SZ, SZ); // top strip + corner
  ctx.fillRect(imgLeft - SZ, imgTop,      SZ,         imgH); // left strip

  ctx.strokeStyle = 'rgba(255,255,255,0.7)';
  ctx.fillStyle   = 'rgba(255,255,255,0.8)';
  ctx.lineWidth   = 1;
  ctx.font        = '9px monospace';

  // Top ruler — ticks rise from the image edge upward
  ctx.textBaseline = 'top';
  for (let px = 0; px <= imgW; px += INCH / 2) {
    const isMajor = px % INCH === 0;
    const x = imgLeft + px;
    const tickH = isMajor ? SZ - 5 : (SZ - 5) * 0.45;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, imgTop);
    ctx.lineTo(x + 0.5, imgTop - tickH);
    ctx.stroke();
    if (isMajor && px > 0) ctx.fillText(`${px / INCH}"`, x + 2, imgTop - SZ + 2);
  }

  // Left ruler — ticks extend left from the image edge
  for (let py = INCH; py <= imgH; py += INCH / 2) {
    const isMajor = py % INCH === 0;
    const y = imgTop + py;
    const tickW = isMajor ? SZ - 5 : (SZ - 5) * 0.45;
    ctx.beginPath();
    ctx.moveTo(imgLeft,        y + 0.5);
    ctx.lineTo(imgLeft - tickW, y + 0.5);
    ctx.stroke();
    if (isMajor) {
      ctx.save();
      ctx.translate(imgLeft - SZ + 2, y);
      ctx.rotate(-Math.PI / 2);
      const label = `${py / INCH}"`;
      ctx.fillText(label, -ctx.measureText(label).width / 2, 0);
      ctx.restore();
    }
  }

  ctx.restore();
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
loadBtn.addEventListener('click', async () => {
  if (window.showDirectoryPicker) {
    try {
      _dirHandle = await window.showDirectoryPicker({ mode: 'readwrite' });
    } catch (e) {
      if (e.name !== 'AbortError') setStatus(`Folder error: ${e.message}`, 'status-err');
      return;
    }
    const found = [];
    for await (const [name, handle] of _dirHandle.entries()) {
      if (handle.kind !== 'file') continue;
      if (!/\.(jpe?g|png|webp|tiff?|bmp)$/i.test(name)) continue;
      found.push({ file: await handle.getFile(), name });
    }
    found.sort((a, b) => a.name.localeCompare(b.name));
    if (!found.length) { setStatus('No images found in that folder', 'status-err'); return; }
    images = found;
    currentIndex = 0;
    hint.style.display = 'none';
    await loadCurrentImage();
  } else {
    fileInput.click();
  }
});

document.getElementById('convert-zip-btn').addEventListener('click', async () => {
  if (!window.showDirectoryPicker) {
    setStatus('Folder picker not supported in this browser.', 'status-err');
    return;
  }
  let dirHandle;
  try {
    dirHandle = await window.showDirectoryPicker({ mode: 'read' });
  } catch (e) {
    if (e.name !== 'AbortError') setStatus(`Folder error: ${e.message}`, 'status-err');
    return;
  }
  const IMAGE_EXTS = new Set(['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp', 'gif']);
  const entries = [];
  for await (const [name, handle] of dirHandle.entries()) {
    if (handle.kind === 'file' && IMAGE_EXTS.has(name.split('.').pop().toLowerCase())) {
      entries.push({ name, handle });
    }
  }
  if (!entries.length) {
    setStatus('No image files found in that folder.', 'status-err');
    return;
  }
  const zip = new JSZip();
  const offscreen = document.createElement('canvas');
  const octx = offscreen.getContext('2d');
  for (let i = 0; i < entries.length; i++) {
    const { name, handle } = entries[i];
    setStatus(`Converting ${i + 1}/${entries.length}: ${name}…`);
    const file = await handle.getFile();
    const url = URL.createObjectURL(file);
    await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        offscreen.width = img.naturalWidth;
        offscreen.height = img.naturalHeight;
        octx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
        offscreen.toBlob(blob => {
          blob.arrayBuffer().then(buf => {
            const stem = name.lastIndexOf('.') > 0 ? name.slice(0, name.lastIndexOf('.')) : name;
            zip.file(stem + '.jpg', buf);
            resolve();
          });
        }, 'image/jpeg', 0.95);
      };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error(`Failed to load ${name}`)); };
      img.src = url;
    });
  }
  setStatus('Creating ZIP…');
  const blob = await zip.generateAsync({ type: 'blob' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = dirHandle.name + '_jpgs.zip';
  a.click();
  URL.revokeObjectURL(a.href);
  setStatus(`Done. Downloaded ${entries.length} JPG(s) as ${dirHandle.name}_jpgs.zip`);
});

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
  reprocessImage();

  counterEl.textContent = `${currentIndex + 1} / ${images.length}`;
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
  if (boxes.length > 0) {
    await saveProcessed(boxes);
    await submitAnnotation(images[currentIndex].name, boxes, images[currentIndex].file, corrected);
  }
  advance();
}

async function saveProcessed(boxes) {
  // If no dir handle yet, try to acquire one; fall back to browser download
  if (!_dirHandle) {
    if (window.showDirectoryPicker) {
      try {
        setStatus('Choose a folder to save processed images…');
        _dirHandle = await window.showDirectoryPicker({ mode: 'readwrite' });
      } catch (e) {
        if (e.name === 'AbortError') { setStatus('Save cancelled'); return; }
        setStatus(`Folder error: ${e.message}`, 'status-err');
        return;
      }
    } else {
      // Browser doesn't support File System Access API — download directly
      await downloadProcessed(boxes);
      return;
    }
  }

  // Compute crop: union of all box bounds + 1-inch pad, clamped to image
  const padNX = 96 / imgDisplayW;
  const padNY = 96 / imgDisplayH;
  let x1 = 1, y1 = 1, x2 = 0, y2 = 0;
  for (const b of boxes) {
    x1 = Math.min(x1, b.x1); y1 = Math.min(y1, b.y1);
    x2 = Math.max(x2, b.x2); y2 = Math.max(y2, b.y2);
  }
  const nx1 = Math.max(0, x1 - padNX), ny1 = Math.max(0, y1 - padNY);
  const nx2 = Math.min(1, x2 + padNX), ny2 = Math.min(1, y2 + padNY);

  const srcX = Math.round(nx1 * currentImageEl.naturalWidth);
  const srcY = Math.round(ny1 * currentImageEl.naturalHeight);
  const srcW = Math.round((nx2 - nx1) * currentImageEl.naturalWidth);
  const srcH = Math.round((ny2 - ny1) * currentImageEl.naturalHeight);

  const off = document.createElement('canvas');
  off.width = srcW; off.height = srcH;
  const offCtx = off.getContext('2d', { willReadFrequently: true });
  const bv = brightnessSlider.value / 100, cv = contrastSlider.value / 100;
  offCtx.filter = `brightness(${bv}) contrast(${cv})`;
  offCtx.drawImage(currentImageEl, srcX, srcY, srcW, srcH, 0, 0, srcW, srcH);
  offCtx.filter = 'none';
  if (_autoContrast || _autoColor) {
    try {
      let id = offCtx.getImageData(0, 0, srcW, srcH);
      if (_autoContrast) id = applyAutoContrast(id);
      if (_autoColor)    id = applyAutoColor(id);
      offCtx.putImageData(id, 0, 0);
    } catch(e) {}
  }

  // Encode JPEG, stepping quality down until under 2.5 MB
  const MAX = 2.5 * 1024 * 1024;
  let quality = 0.92, blob;
  do {
    blob = await new Promise(r => off.toBlob(r, 'image/jpeg', quality));
    quality = Math.max(0.1, quality - 0.1);
  } while (blob.size > MAX && quality > 0.1);

  try {
    const procDir = await _dirHandle.getDirectoryHandle('Processed', { create: true });
    const imgName = images[currentIndex].name;
    const dot  = imgName.lastIndexOf('.');
    const base = dot > 0 ? imgName.slice(0, dot) : imgName;
    const ext  = dot > 0 ? imgName.slice(dot)    : '.jpg';
    const outName = base + '_processed' + ext;
    const fh = await procDir.getFileHandle(outName, { create: true });
    const writable = await fh.createWritable();
    await writable.write(blob);
    await writable.close();
    setStatus(`Saved ${outName} (${(blob.size / 1024).toFixed(0)} KB)`, 'status-ok');
  } catch (e) {
    setStatus(`Save error: ${e.message}`, 'status-err');
    console.warn('saveProcessed error:', e);
  }
}

async function downloadProcessed(boxes) {
  // Fallback: trigger browser download when File System Access API is unavailable
  const padNX = 96 / imgDisplayW;
  const padNY = 96 / imgDisplayH;
  let x1 = 1, y1 = 1, x2 = 0, y2 = 0;
  for (const b of boxes) {
    x1 = Math.min(x1, b.x1); y1 = Math.min(y1, b.y1);
    x2 = Math.max(x2, b.x2); y2 = Math.max(y2, b.y2);
  }
  const nx1 = Math.max(0, x1 - padNX), ny1 = Math.max(0, y1 - padNY);
  const nx2 = Math.min(1, x2 + padNX), ny2 = Math.min(1, y2 + padNY);
  const srcX = Math.round(nx1 * currentImageEl.naturalWidth);
  const srcY = Math.round(ny1 * currentImageEl.naturalHeight);
  const srcW = Math.round((nx2 - nx1) * currentImageEl.naturalWidth);
  const srcH = Math.round((ny2 - ny1) * currentImageEl.naturalHeight);
  const off = document.createElement('canvas');
  off.width = srcW; off.height = srcH;
  const offCtx2 = off.getContext('2d', { willReadFrequently: true });
  const b2 = brightnessSlider.value / 100, c2 = contrastSlider.value / 100;
  offCtx2.filter = `brightness(${b2}) contrast(${c2})`;
  offCtx2.drawImage(currentImageEl, srcX, srcY, srcW, srcH, 0, 0, srcW, srcH);
  offCtx2.filter = 'none';
  if (_autoContrast || _autoColor) {
    try {
      let id = offCtx2.getImageData(0, 0, srcW, srcH);
      if (_autoContrast) id = applyAutoContrast(id);
      if (_autoColor)    id = applyAutoColor(id);
      offCtx2.putImageData(id, 0, 0);
    } catch(e) {}
  }
  const MAX = 2.5 * 1024 * 1024;
  let quality = 0.92, blob;
  do {
    blob = await new Promise(r => off.toBlob(r, 'image/jpeg', quality));
    quality = Math.max(0.1, quality - 0.1);
  } while (blob.size > MAX && quality > 0.1);
  const imgName = images[currentIndex].name;
  const dot = imgName.lastIndexOf('.');
  const base = dot > 0 ? imgName.slice(0, dot) : imgName;
  const outName = base + '_processed.jpg';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = outName;
  a.click();
  URL.revokeObjectURL(a.href);
  setStatus(`Downloaded ${outName} (${(blob.size / 1024).toFixed(0)} KB)`, 'status-ok');
}

function advance() {
  if (currentIndex < images.length - 1) {
    currentIndex++;
    loadCurrentImage();
  } else {
    statusEl.textContent = 'All done!';
    statusEl.className = 'status-done';
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
