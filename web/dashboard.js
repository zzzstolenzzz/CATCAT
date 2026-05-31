'use strict';

const API = window.CATCAT_CONFIG.backendUrl;

// ── Chart setup ────────────────────────────────────────────────────────────────

Chart.defaults.color = 'rgba(255,255,255,0.4)';
Chart.defaults.borderColor = 'rgba(255,255,255,0.07)';
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";

function makeGradient(ctx, color) {
  const g = ctx.createLinearGradient(0, 0, 0, 200);
  g.addColorStop(0, color.replace(')', ',0.25)').replace('rgb', 'rgba'));
  g.addColorStop(1, color.replace(')', ',0)').replace('rgb', 'rgba'));
  return g;
}

const mapCtx = document.getElementById('map-chart').getContext('2d');
const mapChart = new Chart(mapCtx, {
  type: 'line',
  data: { labels: [], datasets: [{
    label: 'mAP50',
    data: [],
    borderColor: '#00ccff',
    backgroundColor: makeGradient(mapCtx, 'rgb(0,204,255)'),
    borderWidth: 2,
    tension: 0.4,
    pointBackgroundColor: '#00ccff',
    pointBorderColor: '#060c18',
    pointBorderWidth: 2,
    pointRadius: 5,
    fill: true,
  }]},
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { maxTicksLimit: 8 } },
      y: {
        min: 0, max: 1,
        grid: { color: 'rgba(255,255,255,0.05)' },
        ticks: { callback: v => (v * 100).toFixed(0) + '%' },
      },
    },
    animation: { duration: 600, easing: 'easeInOutQuart' },
  },
});


// ── DOM refs ───────────────────────────────────────────────────────────────────

const els = id => document.getElementById(id);

// ── Number animation ──────────────────────────────────────────────────────────

const _prev = {};
function animateTo(id, val, suffix = '', decimals = 0) {
  const el = els(id);
  if (!el) return;
  const from = _prev[id] ?? 0;
  if (from === val) return;
  _prev[id] = val;
  const start = performance.now();
  const dur = 600;
  const step = ts => {
    const t = Math.min((ts - start) / dur, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    const cur = from + (val - from) * ease;
    el.textContent = decimals ? cur.toFixed(decimals) + suffix : Math.round(cur) + suffix;
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ── Training ring ─────────────────────────────────────────────────────────────

function updateEngine(stats) {
  const ring     = els('ring-progress');
  const ringStatus = els('ring-status');
  const ringDetail = els('ring-detail');
  const circumference = 314;

  if (stats.training) {
    _elapsedBase = stats.training_elapsed_s ?? 0;
    _elapsedAt = Date.now();
    ring.classList.add('training');
    ringStatus.classList.add('training');
    ringStatus.textContent = 'TRAINING';
    const elapsed = _elapsedBase;
    const mins = Math.floor(elapsed / 60), secs = elapsed % 60;
    const p = stats.training_progress ?? {};
    const epoch = p.epoch ?? 0, epochs = p.epochs ?? 5;
    // Use epochs+1 as denominator so 5/5 = ~83%, leaving arc for export/upload
    const pct = epochs > 0 ? epoch / (epochs + 1) : 0;
    ring.style.strokeDashoffset = circumference * (1 - pct);
    ringDetail.textContent = epoch > 0
      ? `Epoch ${epoch}/${epochs} · ${mins}m ${secs}s`
      : `Starting… ${mins}m ${secs}s`;

    els('eng-epoch').textContent = epoch > 0 ? `${epoch} / ${epochs}` : 'Starting…';
    els('eng-loss').textContent = p.loss != null ? p.loss.toFixed(4) : '—';
  } else {
    _elapsedAt = null;
    ring.classList.remove('training');
    ringStatus.classList.remove('training');
    ringStatus.textContent = 'IDLE';
    ringDetail.textContent = 'Ready';
    ring.style.strokeDashoffset = circumference;
    els('eng-epoch').textContent = '—';
    els('eng-loss').textContent = '—';
  }

  const every = stats.train_every ?? 5;
  const queue = stats.images_in_queue ?? 0;
  const pct = Math.min(queue / every, 1);

  els('eng-next').textContent = `${queue} / ${every} images`;
  els('eng-every').textContent = `${every} annotations`;

  const mv = stats.model_version;
  if (mv && mv !== 'initial') {
    const ts = parseInt(mv, 10);
    if (!isNaN(ts)) {
      const d = new Date(ts * 1000);
      els('eng-version').textContent = `v${stats.training_run_count ?? '—'}`;
      els('eng-updated').textContent = d.toLocaleTimeString();
    }
  } else {
    els('eng-version').textContent = 'base model';
    els('eng-updated').textContent = '—';
  }

  // Queue progress ring
  const offset = 251 * (1 - pct);
  els('p-fill').style.strokeDashoffset = offset;
  els('progress-pct').textContent = Math.round(pct * 100) + '%';
  els('pinfo-done').textContent = `${stats.total_annotations ?? 0} annotated`;
  els('pinfo-queue').textContent = `${queue} in queue`;
  els('pinfo-needed').textContent = `${every} needed to train`;
}

// ── Stats update ──────────────────────────────────────────────────────────────

let lastStats = null;
let _elapsedBase = 0;
let _elapsedAt = null;

setInterval(() => {
  if (!lastStats?.training || _elapsedAt === null) return;
  const elapsed = _elapsedBase + Math.floor((Date.now() - _elapsedAt) / 1000);
  const mins = Math.floor(elapsed / 60), secs = elapsed % 60;
  const p = lastStats.training_progress ?? {};
  const epoch = p.epoch ?? 0, epochs = p.epochs ?? 5;
  const label = epoch >= epochs ? `Finishing… ${mins}m ${secs.toString().padStart(2,'0')}s`
    : epoch > 0 ? `Epoch ${epoch}/${epochs} · ${mins}m ${secs.toString().padStart(2,'0')}s`
    : `Starting… ${mins}m ${secs.toString().padStart(2,'0')}s`;
  const detail = label;
  const ringDetail = document.getElementById('ring-detail');
  if (ringDetail) ringDetail.textContent = detail;
}, 1000);

const TEAM_HEADERS = { 'X-Team-Key': window.CATCAT_CONFIG.teamKey };

async function fetchStats() {
  try {
    const data = await (await fetch(`${API}/team/stats`, { headers: TEAM_HEADERS })).json();
    lastStats = data;

    animateTo('stat-annotations', data.total_annotations ?? 0);
    animateTo('stat-queue', data.images_in_queue ?? 0);
    animateTo('stat-runs', data.training_run_count ?? 0);

    if (data.map50 != null) {
      animateTo('stat-map', data.map50 * 100, '%', 1);
    } else {
      els('stat-map').textContent = '—';
    }

    const next = Math.max(0, (data.train_every ?? 5) - (data.images_in_queue ?? 0));
    els('stat-annotations-sub').textContent =
      data.training ? 'training in progress…' : `${next} until next train`;
    els('stat-queue-sub').textContent =
      data.images_in_queue > 0 ? 'pending training' : 'queue empty';
    els('stat-runs-sub').textContent =
      data.training ? '⚡ running now' : 'completed cycles';

    updateEngine(data);
    els('last-refresh').textContent = 'Updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    console.warn('Stats fetch failed:', e);
  }
}

// ── History update ────────────────────────────────────────────────────────────

async function fetchHistory() {
  try {
    const history = await (await fetch(`${API}/team/history`, { headers: TEAM_HEADERS })).json();
    if (!history.length) return;

    els('map-empty').classList.add('hidden');
    els('run-empty').classList.add('hidden');

    const labels = history.map(h => {
      const d = new Date(h.timestamp * 1000);
      return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' +
             d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    });

    mapChart.data.labels = labels;
    mapChart.data.datasets[0].data = history.map(h => h.map50 ?? 0);
    mapChart.update();

    // Populate runs table with expandable rows
    const tbody = els('runs-tbody');
    tbody.innerHTML = '';
    [...history].reverse().forEach((h, i) => {
      const d = new Date(h.timestamp * 1000);
      const date = d.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
      const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const run = history.length - i;
      const map = h.map50 != null ? (h.map50 * 100).toFixed(1) + '%' : '—';
      const hasImages = h.images && h.images.length > 0;

      const tr = document.createElement('tr');
      tr.className = hasImages ? 'run-row expandable' : 'run-row';
      tr.innerHTML = `<td>${hasImages ? '<span class="expand-arrow">▶</span>' : ''}#${run}</td><td>${date}</td><td>${time}</td><td>${map}</td><td>${h.annotation_count ?? '—'}</td>`;
      tbody.appendChild(tr);

      if (hasImages) {
        const detail = document.createElement('tr');
        detail.className = 'run-detail hidden';
        const imgRows = h.images.map(img =>
          `<tr><td>${img.name}</td><td>${img.detections}</td><td>${img.max_conf > 0 ? (img.max_conf * 100).toFixed(1) + '%' : '—'}</td><td>${img.avg_conf > 0 ? (img.avg_conf * 100).toFixed(1) + '%' : '—'}</td></tr>`
        ).join('');
        detail.innerHTML = `<td colspan="5"><div class="run-detail-inner"><table class="img-table"><thead><tr><th>Image</th><th>Detections</th><th>Max Conf</th><th>Avg Conf</th></tr></thead><tbody>${imgRows}</tbody></table></div></td>`;
        tbody.appendChild(detail);

        tr.addEventListener('click', () => {
          const open = !detail.classList.contains('hidden');
          detail.classList.toggle('hidden', open);
          tr.querySelector('.expand-arrow').textContent = open ? '▶' : '▼';
        });
      }
    });

    const latest = history[history.length - 1];
    if (latest?.map50 != null) {
      els('map-badge').textContent = `mAP50 · ${(latest.map50 * 100).toFixed(1)}%`;
    }
  } catch (e) {
    console.warn('History fetch failed:', e);
  }
}

// ── Global tooltips ───────────────────────────────────────────────────────────
const globalTip = document.getElementById('global-tip');

document.querySelectorAll('.tip').forEach(el => {
  el.addEventListener('mouseenter', () => {
    const r = el.getBoundingClientRect();
    globalTip.textContent = el.dataset.tip;
    globalTip.style.display = 'block';
    const tipW = 210;
    let left = r.left + r.width / 2 - tipW / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - tipW - 8));
    globalTip.style.left = left + 'px';
    const tipH = globalTip.offsetHeight || 60;
    const above = r.top - tipH - 10;
    globalTip.style.top = above >= 8 ? above + 'px' : (r.bottom + 10) + 'px';
  });
  el.addEventListener('mouseleave', () => { globalTip.style.display = 'none'; });
});

// ── Boot ──────────────────────────────────────────────────────────────────────

fetchStats();
fetchHistory();
setInterval(fetchStats, 10000);
setInterval(fetchHistory, 60000);
