// Minimal, dependency-free line chart (canvas) following docs/Design.md:
// hairline solid grid, 2px lines, one y-axis, crosshair + tooltip listing every series,
// a playhead for the animation, colours from CSS tokens (so theme switches just redraw).

const tooltipEl = () => document.getElementById("tooltip");

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function niceTicks(min, max, count = 5) {
  const span = max - min || 1;
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return ticks;
}

function fmt(v, digits) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const d = digits ?? (Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2);
  return v.toFixed(d);
}

function nearestIndex(xs, x) {
  let lo = 0, hi = xs.length - 1;
  if (hi < 0) return -1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < x) lo = mid; else hi = mid;
  }
  return Math.abs(xs[lo] - x) <= Math.abs(xs[hi] - x) ? lo : hi;
}

export class LineChart {
  /**
   * @param {HTMLElement} container
   * @param {{xLabel?: string, xUnit?: string, yFormat?: number, yMin?: number, yMax?: number,
   *          symmetric?: boolean, step?: boolean, playhead?: boolean}} opts
   */
  constructor(container, opts = {}) {
    this.el = container;
    this.opts = { xUnit: "s", playhead: true, ...opts };
    this.canvas = document.createElement("canvas");
    this.el.appendChild(this.canvas);
    this.series = [];
    this.hlines = [];
    this.vlines = [];
    this.playheadX = null;
    this.hoverX = null;
    this.emptyText = "Run a simulation to see results";
    this._empty = document.createElement("div");
    this._empty.className = "chart-empty";
    this.el.appendChild(this._empty);

    new ResizeObserver(() => this.draw()).observe(this.el);
    this.canvas.addEventListener("pointermove", (e) => this._onMove(e));
    this.canvas.addEventListener("pointerleave", () => this._onLeave());
  }

  /** series: [{key, label, color (css var name), x: [], y: [], width?, band?: {lo: [], hi: []}}]
   *  band: an uncertainty band drawn as a light fill of the series colour (e.g. an estimate ± 1σ). */
  setData(series, { hlines = [], vlines = [], xRange = null } = {}) {
    this.series = series.filter((s) => s.x && s.x.length);
    this.hlines = hlines;
    this.vlines = vlines;
    this.xRange = xRange;
    this.draw();
  }

  setPlayhead(x) {
    this.playheadX = x;
    this.draw();
  }

  _bounds() {
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const s of this.series) {
      for (let i = 0; i < s.x.length; i++) {
        const y = s.y[i];
        if (y === null || !Number.isFinite(y)) continue;
        xmin = Math.min(xmin, s.x[i]); xmax = Math.max(xmax, s.x[i]);
        ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
        if (s.band) {
          const lo = s.band.lo[i], hi = s.band.hi[i];
          if (lo !== null && Number.isFinite(lo)) ymin = Math.min(ymin, lo);
          if (hi !== null && Number.isFinite(hi)) ymax = Math.max(ymax, hi);
        }
      }
    }
    for (const h of this.hlines) if (h.includeInRange !== false) { ymin = Math.min(ymin, h.y); ymax = Math.max(ymax, h.y); }
    if (this.xRange) [xmin, xmax] = this.xRange;
    if (this.opts.yMin !== undefined) ymin = this.opts.yMin;
    if (this.opts.yMax !== undefined) ymax = this.opts.yMax;
    if (this.opts.symmetric) { const m = Math.max(Math.abs(ymin), Math.abs(ymax)) || 1; ymin = -m; ymax = m; }
    if (ymin === ymax) { ymin -= 1; ymax += 1; }
    const pad = (ymax - ymin) * 0.06;
    if (this.opts.yMin === undefined) ymin -= pad;
    if (this.opts.yMax === undefined) ymax += pad;
    return { xmin, xmax, ymin, ymax };
  }

  draw() {
    const rect = this.el.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(50, rect.width), H = Math.max(50, rect.height);
    this.canvas.width = Math.round(W * dpr);
    this.canvas.height = Math.round(H * dpr);
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const has = this.series.length > 0;
    this._empty.textContent = has ? "" : this.emptyText;
    this._empty.style.display = has ? "none" : "grid";
    if (!has) return;

    const ink2 = cssVar("--ink-2"), muted = cssVar("--muted"), grid = cssVar("--grid"), axis = cssVar("--axis");
    ctx.font = "11px " + cssVar("--font");
    const b = this._bounds();
    const yTicks = niceTicks(b.ymin, b.ymax, 4);
    const tickW = Math.max(...yTicks.map((t) => ctx.measureText(fmt(t, this.opts.yFormat)).width));
    const m = { l: Math.ceil(tickW) + 12, r: 10, t: 8, b: 24 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const X = (x) => m.l + ((x - b.xmin) / (b.xmax - b.xmin || 1)) * pw;
    const Y = (y) => m.t + (1 - (y - b.ymin) / (b.ymax - b.ymin)) * ph;
    this._geom = { m, pw, ph, X, b, W, H };

    // grid + y ticks
    ctx.lineWidth = 1;
    ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillStyle = muted;
    for (const t of yTicks) {
      const y = Math.round(Y(t)) + 0.5;
      ctx.strokeStyle = t === 0 ? axis : grid;
      ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + pw, y); ctx.stroke();
      ctx.fillText(fmt(t, this.opts.yFormat), m.l - 6, y);
    }
    // x ticks
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of niceTicks(b.xmin, b.xmax, Math.max(2, Math.floor(pw / 80)))) {
      const x = Math.round(X(t)) + 0.5;
      ctx.strokeStyle = grid;
      ctx.beginPath(); ctx.moveTo(x, m.t + ph); ctx.lineTo(x, m.t + ph + 4); ctx.stroke();
      ctx.fillText(fmt(t) + (this.opts.xUnit ? " " + this.opts.xUnit : ""), x, m.t + ph + 7);
    }
    ctx.strokeStyle = axis;
    ctx.beginPath(); ctx.moveTo(m.l, m.t + ph + 0.5); ctx.lineTo(m.l + pw, m.t + ph + 0.5); ctx.stroke();

    // reference lines (limits, flutter speed...) with direct labels
    ctx.save();
    ctx.beginPath(); ctx.rect(m.l, m.t, pw, ph); ctx.clip();
    for (const h of this.hlines) {
      const y = Math.round(Y(h.y)) + 0.5;
      ctx.strokeStyle = cssVar(h.color || "--axis");
      ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + pw, y); ctx.stroke();
      if (h.label) { ctx.fillStyle = ink2; ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillText(h.label, m.l + 4, y - 2); }
    }
    for (const v of this.vlines) {
      const x = Math.round(X(v.x)) + 0.5;
      ctx.strokeStyle = cssVar(v.color || "--ink-2");
      ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + ph); ctx.stroke();
      if (v.label) {
        ctx.fillStyle = cssVar("--ink"); ctx.textBaseline = "top";
        const w = ctx.measureText(v.label).width;
        const left = x + 5 + w < m.l + pw;
        ctx.textAlign = left ? "left" : "right";
        ctx.fillText(v.label, left ? x + 5 : x - 5, m.t + 2 + (v.row || 0) * 14);
      }
    }

    // series
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    for (const s of this.series) {
      if (s.band) this._drawBand(ctx, s, X, Y);
      ctx.strokeStyle = cssVar(s.color);
      ctx.lineWidth = s.width || 2;
      ctx.beginPath();
      let started = false, prevY = null;
      for (let i = 0; i < s.x.length; i++) {
        const yv = s.y[i];
        if (yv === null || !Number.isFinite(yv)) { started = false; continue; }
        const px = X(s.x[i]), py = Y(yv);
        if (!started) { ctx.moveTo(px, py); started = true; }
        else if (this.opts.step) { ctx.lineTo(px, prevY); ctx.lineTo(px, py); }
        else ctx.lineTo(px, py);
        prevY = py;
      }
      ctx.stroke();
      if (s.endMarker) {  // e.g. structural limit exceeded
        const i = s.x.length - 1;
        const px = X(s.x[i]), py = Y(s.y[i]);
        ctx.strokeStyle = cssVar("--critical"); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(px - 4, py - 4); ctx.lineTo(px + 4, py + 4); ctx.moveTo(px + 4, py - 4); ctx.lineTo(px - 4, py + 4); ctx.stroke();
      }
    }

    // playhead + hover crosshair
    ctx.lineWidth = 1;
    if (this.opts.playhead && this.playheadX !== null && this.playheadX >= b.xmin && this.playheadX <= b.xmax) {
      const x = Math.round(X(this.playheadX)) + 0.5;
      ctx.strokeStyle = cssVar("--ink");
      ctx.globalAlpha = 0.5;
      ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + ph); ctx.stroke();
      ctx.globalAlpha = 1;
    }
    if (this.hoverX !== null) {
      const x = Math.round(X(this.hoverX)) + 0.5;
      ctx.strokeStyle = cssVar("--ink-2");
      ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + ph); ctx.stroke();
    }
    ctx.restore();
  }

  _drawBand(ctx, s, X, Y) {
    // fill each run of samples where both bounds exist; follows the step option like the line does
    const ok = (v) => v !== null && v !== undefined && Number.isFinite(v);
    const { lo, hi } = s.band;
    const fillRun = (run) => {
      if (run.length < 2) return;
      ctx.beginPath();
      run.forEach((i, k) => {
        if (k === 0) ctx.moveTo(X(s.x[i]), Y(hi[i]));
        else { if (this.opts.step) ctx.lineTo(X(s.x[i]), Y(hi[run[k - 1]])); ctx.lineTo(X(s.x[i]), Y(hi[i])); }
      });
      for (let k = run.length - 1; k >= 0; k--) {
        const i = run[k];
        ctx.lineTo(X(s.x[i]), Y(lo[i]));
        if (this.opts.step && k > 0) ctx.lineTo(X(s.x[i]), Y(lo[run[k - 1]]));
      }
      ctx.closePath();
      ctx.fill();
    };
    ctx.save();
    ctx.fillStyle = cssVar(s.color);
    ctx.globalAlpha = 0.18;
    let run = [];
    for (let i = 0; i < s.x.length; i++) {
      if (ok(lo[i]) && ok(hi[i])) run.push(i);
      else { fillRun(run); run = []; }
    }
    fillRun(run);
    ctx.restore();
  }

  _onMove(e) {
    if (!this._geom || !this.series.length) return;
    const { m, pw, b } = this._geom;
    const r = this.canvas.getBoundingClientRect();
    const px = e.clientX - r.left;
    if (px < m.l || px > m.l + pw) return this._onLeave();
    const xVal = b.xmin + ((px - m.l) / pw) * (b.xmax - b.xmin);
    // snap to the nearest sample of the longest series
    const ref = this.series.reduce((a, s) => (s.x.length > a.x.length ? s : a), this.series[0]);
    const snapped = ref.x[nearestIndex(ref.x, xVal)];
    this.hoverX = snapped;
    this.draw();

    const tt = tooltipEl();
    tt.replaceChildren();
    const head = document.createElement("div");
    head.className = "tt-head";
    head.textContent = `${this.opts.xLabel || "t"} = ${fmt(snapped, this.opts.xDigits ?? 2)} ${this.opts.xUnit || ""}`;
    tt.appendChild(head);
    for (const s of this.series) {
      const i = nearestIndex(s.x, snapped);
      const inRange = i >= 0 && Math.abs(s.x[i] - snapped) <= (b.xmax - b.xmin) / 200 + 1e-9;
      const row = document.createElement("div");
      row.className = "tt-row";
      const key = document.createElement("span");
      key.className = "key-line";
      key.style.background = cssVar(s.color);
      const val = document.createElement("span");
      val.className = "tt-val";
      val.textContent = inRange ? fmt(s.y[i], this.opts.yFormat) : "ended";
      if (inRange && s.band && Number.isFinite(s.band.hi[i]) && Number.isFinite(s.band.lo[i])) {
        val.textContent += ` ± ${fmt((s.band.hi[i] - s.band.lo[i]) / 2, this.opts.yFormat)}`;
      }
      const lab = document.createElement("span");
      lab.className = "tt-label";
      lab.textContent = s.label;
      row.append(key, val, lab);
      tt.appendChild(row);
    }
    tt.hidden = false;
    const tw = tt.offsetWidth, th = tt.offsetHeight;
    let left = e.clientX + 14, top = e.clientY + 14;
    if (left + tw > window.innerWidth - 8) left = e.clientX - tw - 14;
    if (top + th > window.innerHeight - 8) top = e.clientY - th - 14;
    tt.style.left = left + "px";
    tt.style.top = top + "px";
  }

  _onLeave() {
    this.hoverX = null;
    tooltipEl().hidden = true;
    this.draw();
  }
}
