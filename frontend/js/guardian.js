// Guardian live view: fly one fault-tolerant mission on the server (a background job), then play back the recording.
// Every number shown comes from the recording (src/flutter_ai/fault_tolerance/recording.py). The data is in SI units;
// conversion to mm and degrees happens here, at the presentation boundary.
import { api } from "./api.js";
import { LineChart, cssVar } from "./chart.js";
import { Viewport } from "./viewport.js";

const $ = (id) => document.getElementById(id);
const DEG = 180 / Math.PI;
const POLL_MS = 600;
// colour follows the entity: the wing takes the colour of the controller that is flying it (docs/Design.md)
const CONTROLLERS = {
  lqg: { label: "LQG", color: "--c-lqg" },
  neural: { label: "Neural", color: "--c-neural" },
  lqr: { label: "LQR", color: "--c-lqr" },
};
const MODE_LEVEL = { nominal: 1, robust: 2, cautious: 3, safe: 4 };
const FAULT_LABELS = {
  none: "No fault",
  servo_slew: "Slow flap servo",
  stiffness_loss: "Structural damage (pitch stiffness)",
  sensor_noise: "Noisy sensors",
  turbulence_rise: "Stronger turbulence",
  loop_delay: "Flight-computer delay",
  flap_effectiveness: "Flap damage",
};
const PARAM_NAMES = { f_alpha: "pitch stiffness", f_h: "plunge stiffness", flap_span_fraction: "flap effectiveness" };
const OUTCOMES = {
  switched: { text: "Switched", icon: "M4 11h12.2l-5.6-5.6L12 4l8 8-8 8-1.4-1.4 5.6-5.6H4z" },
  kept: { text: "Kept" },
  blocked: { text: "Blocked", icon: "M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" },
  pending: { text: "Not applied" },
};
const ICON_OK = "M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z";
const ICON_WARN = "M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z";
const ICON_OFF = "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm0 18a8 8 0 1 1 0-16 8 8 0 0 1 0 16z";

const state = {
  options: null,
  supervisor: "guardian",
  flight: null, // the finished recording
  busy: false,
  pollToken: 0,
  playing: false,
  simTime: 0,
  lastFrame: null,
  lastPanels: 0,
  keys: {}, // last-rendered keys per panel, so the DOM is only rebuilt when something changed
  showTruth: false,
  showKept: false,
  lane: null,
  playhead: null, // the mode timeline's playhead element
  displayError: false,
};

const charts = {
  speed: new LineChart($("chart-speed"), { step: true, yFormat: 1 }),
  radius: new LineChart($("chart-radius"), { step: true, yFormat: 3 }),
  h: new LineChart($("chart-h"), { symmetric: true }),
  a: new LineChart($("chart-a"), { symmetric: true }),
  beta: new LineChart($("chart-beta"), { symmetric: true, yFormat: 1 }),
  fault: new LineChart($("chart-fault"), { yMin: 0, step: true, yFormat: 0 }),
};
for (const c of Object.values(charts)) c.emptyText = "Fly a mission to see the recording";
const viewport = new Viewport($("viewport"));
viewport.emptyText = "Fly a mission to animate the wing";

// ---------------------------------------------------------------- utilities
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}
function svgIcon(path) {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("aria-hidden", "true");
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p.setAttribute("d", path);
  s.appendChild(p);
  return s;
}
const finite = (v) => typeof v === "number" && Number.isFinite(v);
const fmt = (v, d = 1) => (finite(v) ? v.toFixed(d) : "–");
const signedPct = (x) => (finite(x) ? `${x >= 0 ? "+" : "−"}${Math.abs(100 * x).toFixed(0)} %` : "–");
const faultLabel = (kind) => FAULT_LABELS[kind] || String(kind).replace(/_/g, " ");
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Index of the last element of the sorted array `times` that is <= t, or -1. */
function lastAtOrBefore(times, t) {
  let lo = 0, hi = times.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= t + 1e-9) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
}

function meter(mode) {
  const m = el("span", "meter");
  const lvl = MODE_LEVEL[mode] || 0;
  for (let k = 1; k <= 4; k++) m.appendChild(el("i", k <= lvl ? "on" : ""));
  return m;
}

function statusCell(kind, text) { // kind: ok | fail | unknown - status is never colour alone (icon + label)
  const cls = { ok: "status-ok", fail: "status-fail", unknown: "status-unknown" }[kind];
  const s = el("span", `status-cell ${cls}`);
  s.append(svgIcon(kind === "ok" ? ICON_OK : kind === "fail" ? ICON_WARN : ICON_OFF), document.createTextNode(text));
  return s;
}

function legend(id, items) {
  const box = $(id);
  box.replaceChildren();
  for (const it of items) {
    const item = el("span", "legend-item"), key = el("span", "key-line");
    key.style.background = `var(${it.color})`;
    if (it.width && it.width < 2) key.style.height = "2px";
    item.append(key, document.createTextNode(it.label));
    box.appendChild(item);
  }
}

function showBanner(msg) {
  const b = $("banner");
  b.textContent = msg || "";
  b.hidden = !msg;
}

function announce(msg) {
  $("announce").textContent = msg;
}

function modeSpeedText(F, mode) {
  const m = F.modes[mode];
  if (!m) return "";
  const ctrl = (CONTROLLERS[m.controller] || { label: m.controller }).label;
  return m.speed_ratio === null || m.speed_ratio === undefined
    ? `${ctrl} at cruise, ${fmt(m.target_speed)} m/s`
    : `${ctrl}, slow to ${fmt(m.target_speed)} m/s (${fmt(m.speed_ratio, 1)} × healthy U_F)`;
}

// ---------------------------------------------------------------- form
function fillForm() {
  const o = state.options;
  const d = o.defaults;
  const presetSel = $("preset");
  presetSel.replaceChildren();
  for (const p of o.presets) {
    const opt = el("option", "", p.name.replace(/_/g, " "));
    opt.value = p.name;
    opt.title = p.description;
    presetSel.appendChild(opt);
  }
  presetSel.value = d.preset;
  updatePresetReadouts();

  const faultSel = $("fault");
  faultSel.replaceChildren();
  for (const [kind, spec] of Object.entries(o.faults)) {
    const opt = el("option", "", faultLabel(kind) + (spec.held_out ? " (held out)" : ""));
    opt.value = kind;
    faultSel.appendChild(opt);
  }
  faultSel.value = d.fault;
  applyFaultDefaults();

  $("duration").value = d.duration;
  $("seed").value = d.seed;
  $("estimator").checked = d.estimator;
  state.supervisor = d.supervisor;
  renderSupervisorButtons();
}

function updatePresetReadouts() {
  const p = state.options.presets.find((x) => x.name === $("preset").value);
  $("ro-flutter").textContent = p ? `${fmt(p.flutter_speed)} m/s at ${fmt(p.flutter_frequency_hz, 2)} Hz` : "–";
}

function applyFaultDefaults() {
  const kind = $("fault").value;
  const spec = state.options.faults[kind];
  if (!spec) return;
  $("onset").value = spec.onset;
  $("ramp").value = spec.ramp;
  $("severity").value = spec.severity;
  const none = kind === "none";
  for (const id of ["onset", "ramp", "severity"]) $(id).disabled = none;
  updateFaultHint();
}

function updateFaultHint() {
  const kind = $("fault").value;
  const spec = state.options.faults[kind];
  if (!spec) return;
  const sev = Number($("severity").value);
  const rel = spec.severity > 0 ? sev / spec.severity : 0;
  $("out-severity").textContent = kind === "none" ? "–" : `${fmt(100 * rel, 0)} % of the benchmark fault`;
  let text = kind === "none" ? "A healthy flight: shows false alarms, if any." : `Benchmark size: ${spec.description}.`;
  if (spec.held_out) text += " Held out: neither the rules nor the LLM guidance describe this fault.";
  if (kind !== "none") text += ` Benchmark timing: onset ${spec.onset} s, ramp ${spec.ramp} s.`;
  $("fault-hint").textContent = text;
}

function renderSupervisorButtons() {
  const wrap = $("supervisor-buttons");
  wrap.replaceChildren();
  for (const [id, s] of Object.entries(state.options.supervisors)) {
    const b = el("button", "", s.label);
    b.type = "button";
    b.setAttribute("aria-pressed", String(id === state.supervisor));
    b.addEventListener("click", () => { state.supervisor = id; renderSupervisorButtons(); });
    wrap.appendChild(b);
  }
  const sup = state.options.supervisors[state.supervisor];
  $("supervisor-hint").textContent = sup ? sup.description : "";
  const box = $("estimator");
  if (sup && sup.needs_estimator) { box.checked = true; box.disabled = true; }
  else box.disabled = false;
  const every = state.options.estimator_every_s;
  $("estimator-hint").textContent = state.supervisor === "guardian"
    ? `Required by the guardian: estimator v2 (control authority, measured delay, worst case over the fit's uncertainty) every ${every} s, plus the learned fault diagnoser every 0.5 s.`
    : sup && sup.needs_estimator
      ? `Required by ${sup.label.toLowerCase()}: it runs every ${every} s of flight time and is the slow part of the computation.`
      : `Runs every ${every} s of flight time, for display only (this supervisor does not read it). Off = much faster.`;
}

function requestBody() {
  const kind = $("fault").value;
  const num = (id) => (($(id).value === "") ? null : Number($(id).value));
  return {
    preset: $("preset").value,
    fault: kind,
    onset: kind === "none" ? null : num("onset"),
    ramp: kind === "none" ? null : num("ramp"),
    severity: kind === "none" ? null : num("severity"),
    supervisor: state.supervisor,
    estimator: $("estimator").checked,
    duration: num("duration"),
    seed: num("seed"),
  };
}

function renderStatusChips() {
  const row = $("status-row");
  row.replaceChildren();
  const chip = (ok, text, title) => {
    const c = el("span", `chip ${ok ? "ok" : "off"}`);
    c.append(svgIcon(ok ? ICON_OK : ICON_OFF), document.createTextNode(text));
    if (title) c.title = title;
    return c;
  };
  const p = state.options.presets.find((x) => x.name === $("preset").value);
  const neural = Boolean(p && p.neural_policy_available);
  row.append(
    chip(true, "Physics model"),
    chip(neural, neural ? "Neural policy" : "Neural policy not trained",
      neural ? "" : `Run: python -m flutter_ai train --preset ${p ? p.name : "uav_flap"}`),
    chip(false, "No LLM in this view", "The Guardian view never calls a language model."),
  );
}

// ---------------------------------------------------------------- flying (background job + polling)
function setBusy(on) {
  state.busy = on;
  const btn = $("fly-btn");
  btn.disabled = on;
  btn.textContent = on ? "Flying…" : "Fly mission";
  document.body.classList.toggle("is-loading", on);
  $("progress").hidden = !on;
  if (on) showProgress({ status: "queued", stage: "queued", progress: 0, sim_time: 0, queued_ahead: 0 });
}

function showProgress(s) {
  let overall = 0, text = "Starting…";
  const duration = Number(s.duration) || Number($("duration").value) || 0;
  if (s.status === "queued") {
    text = s.queued_ahead ? `Queued behind ${s.queued_ahead} other flight(s)…` : "Queued…";
  } else if (s.stage === "flying") {
    overall = 0.9 * s.progress;
    text = `Simulating the flight: ${fmt(s.sim_time)} of ${fmt(duration)} s`;
    if ($("estimator").checked) text += " (the on-board estimator runs every simulated second)";
  } else if (s.stage === "ground_truth") {
    overall = 0.9 + 0.1 * s.progress;
    text = "Computing the ground truth: the true flutter speed at every fault level…";
  } else if (s.status === "running") {
    text = "Simulating the flight (the supervisor starts after the 4.5 s healthy calibration)…";
  } else if (s.status === "done") {
    overall = 1;
    text = "Done.";
  }
  $("progress-fill").style.width = `${Math.round(100 * overall)}%`;
  $("progress-bar").setAttribute("aria-valuenow", String(Math.round(100 * overall)));
  $("progress-text").textContent = text;
}

async function fly() {
  if (state.busy) return;
  const body = requestBody();
  showBanner("");
  const token = ++state.pollToken;
  setBusy(true);
  try {
    let status = await api.guardianStart(body);
    while (status.status !== "done") {
      if (token !== state.pollToken) return;
      showProgress(status);
      await sleep(POLL_MS);
      status = await api.guardianFlight(status.id);
    }
    if (!status.result) status = await api.guardianFlight(status.id); // a reused flight: fetch its recording
    if (token !== state.pollToken) return;
    loadFlight(status.result);
  } catch (err) {
    showBanner(`Flight failed: ${err.message}`);
  } finally {
    if (token === state.pollToken) setBusy(false);
  }
}

// ---------------------------------------------------------------- loading a recording
function loadFlight(F) {
  const s = F.series;
  F._end = s.t[s.t.length - 1];
  F._tickTimes = F.ticks.map((t) => t.time);
  F._estTimes = F.estimates.map((e) => e.time);
  F._decTimes = F.decisions.map((d) => d.time);
  F._switchTimes = F.switches.map((w) => w.time);
  F._hasDiagnosis = F.ticks.some((t) => t.diagnosis !== null && t.diagnosis !== undefined);
  // one gauge scale per flight (no rescaling during playback)
  const vals = [F.flutter_speed, ...s.airspeed.filter(finite)];
  for (const e of F.estimates) {
    if (!finite(e.flutter_speed)) continue;
    const sd = finite(e.flutter_speed_std) ? e.flutter_speed_std : 0;
    vals.push(e.flutter_speed - sd, e.flutter_speed + sd);
  }
  for (const v of F.truth.flutter_speed) if (finite(v)) vals.push(v);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = Math.max(1, 0.08 * (hi - lo));
  F._gauge = [lo - pad, hi + pad];

  state.flight = F;
  state.keys = {};
  state.lane = {
    id: "flight", label: "LQG active", color: "--c-lqg",
    case: { supervisor: "guardian", controller: "lqg" },
    result: {
      t: s.t, h: s.h, alpha: s.alpha, beta: s.beta || s.t.map(() => 0), u: s.u, w_gust: [], mode: s.mode,
      exceeded_limits: F.failed, failure_time: F.failure_time,
    },
  };
  viewport.setData({ params: F.params, velocity: F.cruise }, [state.lane]);
  $("ro-cruise").textContent = `${fmt(F.cruise)} m/s (${fmt(100 * F.cruise / F.flutter_speed, 0)} % of healthy U_F)`;
  $("diag-card").hidden = !F._hasDiagnosis;
  $("csv-btn").disabled = false;
  $("json-btn").disabled = false;

  renderCharts();
  renderTimeline();
  renderSummary();
  state.simTime = 0;
  updatePanels(true);
  const outcome = F.failed ? `the wing exceeded its structural limit at ${fmt(F.failure_time, 2)} s` : "the wing survived";
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  setPlaying(!reduce);
  announce(`Flight computed: ${outcome}. ${reduce ? "Press play to watch it." : "Playing it back."}`);
}

// ---------------------------------------------------------------- static renders (once per flight)
function withEnd(xs, ys, end) {
  if (!xs.length) return [[], []];
  return [[...xs, end], [...ys, ys[ys.length - 1]]];
}

function renderCharts() {
  const F = state.flight;
  if (!F) return;
  const s = F.series, p = F.params, end = F._end;
  const xRange = [0, F.request.duration];
  const tEst = F._estTimes;

  // airspeed vs flutter speed
  const items = [{ label: "Airspeed", color: "--ink" }];
  const series = [{ key: "U", label: "Airspeed", color: "--ink", x: s.t, y: s.airspeed }];
  if (F.estimates.length) {
    const [xe, ye] = withEnd(tEst, F.estimates.map((e) => e.flutter_speed), end);
    const std = F.estimates.map((e) => (finite(e.flutter_speed_std) ? e.flutter_speed_std : null));
    const [, sd] = withEnd(tEst, std, end);
    const lo = ye.map((v, k) => (finite(v) && finite(sd[k]) ? v - sd[k] : null));
    const hi = ye.map((v, k) => (finite(v) && finite(sd[k]) ? v + sd[k] : null));
    series.push({ key: "est", label: "Estimated flutter speed (on-board, ±1σ)", color: "--c-estimate", x: xe, y: ye, band: { lo, hi } });
    items.push({ label: "Estimated flutter speed (on-board, ±1σ band)", color: "--c-estimate" });
  }
  if (state.showTruth) {
    const [xt, yt] = withEnd(F.truth.t, F.truth.flutter_speed, end);
    series.push({ key: "truth", label: "True flutter speed (simulation only)", color: "--muted", width: 1.5, x: xt, y: yt });
    items.push({ label: "True flutter speed (simulation only)", color: "--muted", width: 1.5 });
  }
  legend("legend-speed", items);
  charts.speed.setData(series, {
    xRange,
    hlines: [{ y: F.flutter_speed, label: `healthy wing U_F = ${fmt(F.flutter_speed)} m/s`, color: "--axis" }],
  });

  // per-mode predicted spectral radius
  if (F.estimates.length) {
    const radius = F.mode_order.map((m, k) => {
      const [x, y] = withEnd(tEst, F.estimates.map((e) => (e.mode_spectral_radius || {})[m] ?? null), end);
      return { key: m, label: m, color: `--seq-${k + 1}`, x, y };
    });
    legend("legend-radius", F.mode_order.map((m, k) => ({ label: m, color: `--seq-${k + 1}` })));
    charts.radius.setData(radius, { xRange, hlines: [{ y: 1, label: "ρ = 1: stability limit", color: "--critical" }] });
  } else {
    legend("legend-radius", []);
    charts.radius.emptyText = "The on-board estimator did not run in this flight";
    charts.radius.setData([]);
  }

  const hLim = p.h_limit * 1000, aLim = p.alpha_limit_deg;
  const scaled = (arr, k) => arr.map((v) => (finite(v) ? v * k : null));
  charts.h.setData([{ key: "h", label: "Plunge", color: "--ink", x: s.t, y: scaled(s.h, 1000), endMarker: F.failed }], {
    xRange,
    hlines: [{ y: hLim, label: "structural limit", color: "--critical", includeInRange: false },
             { y: -hLim, color: "--critical", includeInRange: false }],
  });
  charts.a.setData([{ key: "a", label: "Pitch", color: "--ink", x: s.t, y: scaled(s.alpha, DEG), endMarker: F.failed }], {
    xRange,
    hlines: [{ y: aLim, label: "structural limit", color: "--critical", includeInRange: false },
             { y: -aLim, color: "--critical", includeInRange: false }],
  });
  const betaSeries = [];
  if (s.beta) betaSeries.push({ key: "beta", label: "Flap angle β", color: "--ink", x: s.t, y: scaled(s.beta, DEG), endMarker: F.failed });
  if (s.u && s.u[0] && p.actuation === "flap_servo") {
    betaSeries.push({ key: "cmd", label: "Command β_cmd", color: "--muted", width: 1, x: s.t, y: scaled(s.u[0], DEG) });
  }
  legend("legend-beta", betaSeries.map((x) => ({ label: x.label, color: x.color, width: x.width })));
  const uLim = p.u_max[0] * DEG;
  charts.beta.setData(betaSeries, {
    xRange,
    hlines: [{ y: uLim, label: "command limit", color: "--axis", includeInRange: false },
             { y: -uLim, color: "--axis", includeInRange: false }],
  });
  const bench = F.fault.benchmark_severity || 1;
  charts.fault.setData([{ key: "fault", label: "Fault size (% of benchmark)", color: "--ink-2", width: 1.5, x: s.t,
    y: s.fault_level.map((v) => (100 * v) / bench) }], { xRange });
}

function renderTimeline() {
  const F = state.flight;
  const tl = $("timeline");
  tl.replaceChildren();
  const s = F.series, D = F.request.duration, end = F._end;
  const runs = [];
  let start = s.t[0], mode = s.mode[0];
  for (let i = 1; i < s.t.length; i++) {
    if (s.mode[i] !== mode) { runs.push([start, s.t[i], mode]); start = s.t[i]; mode = s.mode[i]; }
  }
  runs.push([start, end, mode]);
  const pct = (t) => `${Math.max(0, Math.min(100, (100 * t) / D))}%`;
  for (const [t0, t1, m] of runs) {
    const seg = el("div", "seg");
    seg.dataset.level = String(MODE_LEVEL[m] || 1);
    seg.style.left = pct(t0);
    seg.style.width = `${Math.max(0, (100 * (t1 - t0)) / D)}%`;
    seg.title = `${m}: ${fmt(t0, 2)}–${fmt(t1, 2)} s`;
    tl.appendChild(seg);
  }
  const hasOnset = F.fault.kind !== "none" && F.fault.onset < end;
  if (hasOnset) {
    const on = el("div", "onset");
    on.style.left = pct(F.fault.onset);
    on.title = `fault onset at ${fmt(F.fault.onset)} s`;
    tl.appendChild(on);
  }
  if (F.failed) {
    const f = el("div", "fail");
    f.style.left = pct(end);
    f.title = `structural limit exceeded at ${fmt(F.failure_time, 2)} s`;
    tl.appendChild(f);
  }
  state.playhead = el("div", "playhead");
  tl.appendChild(state.playhead);
  const desc = runs.map(([t0, t1, m]) => `${m} ${fmt(t0, 1)} to ${fmt(t1, 1)} s`).join(", ");
  tl.setAttribute("aria-label", `Mode timeline: ${desc}.${hasOnset ? ` Fault onset at ${fmt(F.fault.onset)} s.` : ""}` +
    (F.failed ? ` Structural limit exceeded at ${fmt(F.failure_time, 2)} s.` : ""));

  const lg = $("timeline-legend");
  lg.replaceChildren();
  for (const m of F.mode_order) {
    const item = el("span", "legend-item"), sw = el("span", "swatch");
    sw.dataset.level = String(MODE_LEVEL[m] || 1);
    item.append(sw, document.createTextNode(m));
    lg.appendChild(item);
  }
  if (hasOnset) {
    const item = el("span", "legend-item"), sw = el("span", "swatch onset-key");
    item.append(sw, document.createTextNode("fault onset"));
    lg.appendChild(item);
  }
  if (F.failed) {
    const item = el("span", "legend-item");
    item.append(statusCell("fail", "limit exceeded"));
    lg.appendChild(item);
  }
}

function renderSummary() {
  const F = state.flight, m = F.metrics;
  const t = $("summary");
  t.replaceChildren();
  const thead = el("thead"), hr = el("tr");
  hr.append(el("th", "", "Quantity"), el("th", "", "Value"));
  thead.appendChild(hr);
  const tbody = el("tbody");
  const row = (name, value) => {
    const tr = el("tr");
    const td = el("td");
    if (value instanceof Node) td.appendChild(value); else td.textContent = value;
    tr.append(el("td", "", name), td);
    tbody.appendChild(tr);
  };
  row("Outcome", F.failed ? statusCell("fail", `Structural limit exceeded at ${fmt(F.failure_time, 2)} s`)
    : statusCell("ok", "Survived the whole flight"));
  row("Fault", F.fault.kind === "none" ? "None"
    : `${faultLabel(F.fault.kind)}${F.fault.held_out ? " (held out)" : ""}: onset ${fmt(F.fault.onset)} s, ramp ${fmt(F.fault.ramp)} s, ` +
      `${fmt((100 * F.fault.severity) / (F.fault.benchmark_severity || 1), 0)} % of the benchmark size`);
  row("Supervisor", `${F.supervisor.label}; on-board estimator ${F.estimator_every_s ? `every ${F.estimator_every_s} s` : "off"}`);
  row("First response after the onset", m.first_response ? `${m.first_response} after ${fmt(m.reaction_s, 2)} s` : "None");
  row("Final mode", m.final_mode);
  row("Mode switches", String(m.switches));
  row("Supervisor answers", String(m.decisions));
  row("Time below cruise speed", `${fmt(m.slowed_s)} s`);
  row("RMS plunge, whole flight", `${fmt(m.rms_h_mm, 2)} mm`);
  row("RMS plunge after the onset", finite(m.after_onset_rms_h_mm) ? `${fmt(m.after_onset_rms_h_mm, 2)} mm` : "–");
  row("Cost J", finite(m.cost_J) ? fmt(m.cost_J, 3) : "– (not defined for a failed flight)");
  row("Turbulence seed", String(F.request.seed));
  row("Server compute time", `flight ${fmt(F.compute_s.flight, 0)} s, ground truth ${fmt(F.compute_s.ground_truth, 0)} s`);
  t.append(thead, tbody);
}

// ---------------------------------------------------------------- panels that follow the playhead
function changed(name, key) {
  if (state.keys[name] === key) return false;
  state.keys[name] = key;
  return true;
}

function updatePanels(force = false) {
  const F = state.flight;
  if (!F) return;
  if (force) state.keys = {};
  const t = state.simTime, s = F.series;
  const i = Math.max(0, lastAtOrBefore(s.t, t));
  const tickI = lastAtOrBefore(F._tickTimes, t);
  const estI = lastAtOrBefore(F._estTimes, t);
  const decI = lastAtOrBefore(F._decTimes, t);
  const mode = s.mode[i];
  const U = s.airspeed[i];
  const est = estI >= 0 ? F.estimates[estI] : null;
  const truthI = lastAtOrBefore(F.truth.t, t);
  const truthV = truthI >= 0 ? F.truth.flutter_speed[truthI] : null;
  const failedNow = F.failed && t >= F.failure_time - 1e-9;

  // flight status
  if (changed("status", `${failedNow}|${t >= F._end - 1e-9}`)) {
    const box = $("i-status");
    box.replaceChildren(failedNow ? statusCell("fail", `Structural limit exceeded at ${fmt(F.failure_time, 2)} s`)
      : statusCell("ok", t >= F._end - 1e-9 ? "Flight completed within limits" : "Within limits"));
  }
  // mode
  if (changed("mode", mode)) {
    const box = $("i-mode");
    box.replaceChildren(meter(mode), document.createTextNode(mode));
    const w = lastAtOrBefore(F._switchTimes, t);
    $("i-mode-sub").textContent = modeSpeedText(F, mode) + (w >= 0 ? ` · since ${fmt(F.switches[w].time, 2)} s` : "");
  }
  // fault (simulation truth)
  const lvl = s.fault_level[i] || 0;
  if (changed("fault", `${lvl}`)) {
    const f = F.fault;
    $("i-fault").textContent = faultLabel(f.kind);
    let sub = "";
    if (f.kind !== "none") {
      const frac = f.severity > 0 ? lvl / f.severity : 0;
      sub = t < f.onset ? `Not active yet (onset at ${fmt(f.onset)} s)`
        : frac < 1 - 1e-9 ? `Growing: ${fmt(100 * frac, 0)} % of full size` : "At full size";
      sub += ` · full size = ${fmt((100 * f.severity) / (f.benchmark_severity || 1), 0)} % of the benchmark fault`;
      $("i-fault-bar").style.width = `${Math.round(100 * Math.min(1, frac))}%`;
    } else {
      $("i-fault-bar").style.width = "0%";
    }
    $("i-fault-sub").textContent = sub;
  }
  // peak motion from the latest report
  const tick = tickI >= 0 ? F.ticks[tickI] : null;
  if (changed("risk", tickI)) $("i-risk").textContent = tick ? `${fmt(100 * tick.risk, 0)} %` : "–";

  // flutter margin tiles + gauge
  const Ukey = fmt(U);
  if (changed("kpi", `${Ukey}|${estI}|${state.showTruth}|${truthV}`)) {
    $("k-speed").textContent = `${Ukey} m/s`;
    $("k-speed-sub").textContent = `${fmt((100 * U) / F.flutter_speed, 0)} % of the healthy flutter speed`;
    renderEstimateTiles(F, est);
    $("k-truth-tile").hidden = !state.showTruth;
    $("k-truth").textContent = finite(truthV) ? `${fmt(truthV)} m/s` : "none in range";
    renderGauge(F, U, est, truthV);
  }
  // traffic lights
  if (changed("lights", `${estI}|${mode}`)) renderLights(F, est, mode);
  // report
  if (changed("report", tickI)) {
    const pre = $("report");
    if (tick) {
      pre.textContent = tick.report;
      $("report-time").textContent = `t = ${fmt(tick.time, 2)} s`;
    } else {
      pre.textContent = F.ticks.length
        ? `No report yet: the first one is issued at ${fmt(F.ticks[0].time, 2)} s, after the diagnostics calibrate on the healthy wing.`
        : "The supervisor was never asked (the flight ended before the first report).";
      $("report-time").textContent = "";
    }
  }
  // decisions
  if (changed("log", `${decI}|${state.showKept}`)) renderLog(F, decI);
  // optional learned diagnosis
  if (F._hasDiagnosis && changed("diag", tickI)) {
    let k = tickI;
    while (k >= 0 && (F.ticks[k].diagnosis === null || F.ticks[k].diagnosis === undefined)) k--;
    renderDiagnosis(k >= 0 ? F.ticks[k].diagnosis : null);
    $("diag-time").textContent = k >= 0 ? `t = ${fmt(F.ticks[k].time, 2)} s` : "";
  }
}

function renderEstimateTiles(F, est) {
  if (!F.estimator_every_s) {
    $("k-est").textContent = "Off";
    $("k-est-sub").textContent = "The estimator did not run in this flight";
    $("k-margin").textContent = "–";
    $("k-margin-sub").textContent = "";
    return;
  }
  if (!est) {
    $("k-est").textContent = "–";
    $("k-est-sub").textContent = F.estimates.length ? `First estimate at ${fmt(F.estimates[0].time, 1)} s` : "No estimate in this flight";
    $("k-margin").textContent = "–";
    $("k-margin-sub").textContent = "";
    return;
  }
  const params = Object.entries(est.params || {}).map(([k, v]) => `${PARAM_NAMES[k] || k} ×${fmt(v, 2)}`).join(", ");
  if (finite(est.flutter_speed)) {
    $("k-est").textContent = `${fmt(est.flutter_speed)} m/s` + (finite(est.flutter_speed_std) ? ` ± ${fmt(est.flutter_speed_std)}` : "");
  } else {
    $("k-est").textContent = "None found";
  }
  $("k-est-sub").textContent = `Identified at ${fmt(est.time, 1)} s: ${params}` + (est.converged ? "" : " (fit did not converge)") +
    // estimator v2 (guardian flights): worst case over the 95 % confidence region, and the turbulence it measures
    (est.version === 2 && finite(est.flutter_speed_worst) ? `; worst case ${fmt(est.flutter_speed_worst)} m/s` : "") +
    (est.version === 2 && finite(est.turbulence_rms) ? `; turbulence ${fmt(est.turbulence_rms, 2)} m/s RMS` : "");
  if (finite(est.margin)) {
    $("k-margin").textContent = signedPct(est.margin);
    $("k-margin-sub").textContent = est.margin < 0
      ? `At ${fmt(est.airspeed)} m/s: above the estimated open-loop flutter speed; the active controller holds the wing.`
      : `At ${fmt(est.airspeed)} m/s: below the estimated open-loop flutter speed.`;
  } else {
    $("k-margin").textContent = "–";
    $("k-margin-sub").textContent = "No flutter speed found in the analysis range";
  }
}

function renderGauge(F, U, est, truthV) {
  const g = $("gauge");
  g.replaceChildren();
  const [lo, hi] = F._gauge;
  const pos = (v) => Math.max(0, Math.min(100, (100 * (v - lo)) / (hi - lo || 1)));
  const place = (node, p) => {
    node.style.left = `${p}%`;
    node.style.transform = p < 12 ? "none" : p > 88 ? "translateX(-100%)" : "translateX(-50%)";
  };
  g.appendChild(el("div", "track"));
  const hasEst = est && finite(est.flutter_speed);
  if (hasEst) {
    const sd = finite(est.flutter_speed_std) ? est.flutter_speed_std : 0;
    const band = el("div", "band");
    band.style.left = `${pos(est.flutter_speed - sd)}%`;
    band.style.width = `${Math.max(0, pos(est.flutter_speed + sd) - pos(est.flutter_speed - sd))}%`;
    g.appendChild(band);
  }
  const healthy = el("div", "mark healthy");
  healthy.style.left = `${pos(F.flutter_speed)}%`;
  healthy.title = `healthy wing flutter speed ${fmt(F.flutter_speed)} m/s`;
  g.appendChild(healthy);
  if (state.showTruth && finite(truthV)) {
    const m = el("div", "mark truth");
    m.style.left = `${pos(truthV)}%`;
    g.appendChild(m);
    const l = el("span", "lbl row2 muted", `truth ${fmt(truthV)} m/s (simulation only)`);
    place(l, pos(truthV));
    g.appendChild(l);
  }
  if (hasEst) {
    const m = el("div", "mark est");
    m.style.left = `${pos(est.flutter_speed)}%`;
    g.appendChild(m);
    const key = el("span", "key-line");
    key.style.background = "var(--c-estimate)";
    const l = el("span", "lbl row1");
    l.append(key, document.createTextNode(`estimate ${fmt(est.flutter_speed)}` +
      (finite(est.flutter_speed_std) ? ` ± ${fmt(est.flutter_speed_std)}` : "") + " m/s"));
    place(l, pos(est.flutter_speed));
    g.appendChild(l);
  }
  if (finite(U)) {
    const a = el("div", "speed-arrow");
    a.style.left = `${pos(U)}%`;
    g.appendChild(a);
    const l = el("span", "lbl top", `airspeed ${fmt(U)} m/s`);
    place(l, pos(U));
    g.appendChild(l);
  }
  const span = hi - lo, step = span > 30 ? 10 : span > 12 ? 5 : 2;
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    const tk = el("span", "tick", `${v}`);
    tk.style.left = `${pos(v)}%`;
    g.appendChild(tk);
  }
  const unit = el("span", "tick", "m/s");
  unit.style.left = "100%";
  unit.style.transform = "translateX(-100%)";
  unit.style.top = "0";
  if (!finite(U) || pos(U) < 80) g.appendChild(unit);
  g.setAttribute("aria-label", `Flutter margin gauge: airspeed ${fmt(U)} m/s; ` +
    (hasEst ? `estimated flutter speed ${fmt(est.flutter_speed)} m/s; ` : "no flutter-speed estimate yet; ") +
    `healthy wing ${fmt(F.flutter_speed)} m/s` + (state.showTruth && finite(truthV) ? `; true flutter speed ${fmt(truthV)} m/s.` : "."));
}

function renderLights(F, est, mode) {
  const box = $("lights");
  box.replaceChildren();
  const rad = est ? est.mode_spectral_radius || {} : {};
  const verdict = est && est.verdict ? est.verdict : null; // guardian flights: which modes it may choose
  const perMode = est && est.modes ? est.modes : {};
  for (const m of F.mode_order) {
    const r = rad[m];
    const card = el("div", `light${m === mode ? " current" : ""}`);
    const head = el("div", "light-head");
    head.append(meter(m), document.createTextNode(m));
    if (m === mode) head.appendChild(el("span", "tag", "current"));
    card.appendChild(head);
    card.appendChild(el("div", "light-sub", modeSpeedText(F, m)));
    if (verdict) {
      const ok = verdict.admissible[m];
      card.appendChild(statusCell(ok ? "ok" : "fail", ok ? "Admitted" : `Vetoed: ${verdict.why[m] || "not admissible"}`));
      const ratio = (perMode[m] || {}).authority_ratio;
      const bits = [finite(r) ? `ρ = ${r.toFixed(3)}` : null, finite(ratio) ? `authority ${ratio.toFixed(2)}` : null];
      const num = el("div", "light-num", bits.filter(Boolean).join(" · "));
      num.title = "ρ: spectral radius on the identified wing (< 1 stable). Authority: 3σ flap command or rate over its " +
        "limit (≥ 1: saturation expected). After structural damage the guardian holds speed unless the current mode is " +
        "predicted unstable, even if it is vetoed for authority.";
      card.appendChild(num);
    } else if (finite(r)) {
      card.appendChild(statusCell(r < 1 ? "ok" : "fail", r < 1 ? "Predicted stable" : "Predicted unstable"));
      card.appendChild(el("div", "light-num", `ρ = ${r.toFixed(3)}`));
    } else {
      card.appendChild(statusCell("unknown", F.estimator_every_s ? "No estimate yet" : "Estimator off"));
    }
    box.appendChild(card);
  }
  $("lights-time").textContent = !est ? ""
    : verdict ? `the guardian's reading of the estimate from t = ${fmt(est.time, 1)} s (structural damage: it holds speed unless the current mode is predicted unstable)`
      : `on-board estimate from t = ${fmt(est.time, 1)} s`;
}

function renderLog(F, decI) {
  const list = $("decision-log");
  list.replaceChildren();
  const shown = [];
  for (let k = decI; k >= 0 && shown.length < 200; k--) {
    const d = F.decisions[k];
    if (state.showKept || d.outcome !== "kept") shown.push(d);
  }
  if (!shown.length) {
    let text;
    if (F.supervisor.id === "none") text = "No supervisor in this flight: the nominal mode throughout.";
    else if (decI < 0) {
      text = F.ticks.length ? `No decisions yet: the first report is issued at ${fmt(F.ticks[0].time, 2)} s.` : "No decisions.";
    } else text = `${decI + 1} answers so far, all “keep the current mode”. Tick “Show keep answers” to list them.`;
    list.appendChild(el("li", "empty", text));
    return;
  }
  shown.forEach((d, k) => {
    const li = el("li", k === 0 ? "latest" : "");
    const pill = el("span", "mode-pill");
    pill.append(meter(d.mode), document.createTextNode(d.mode));
    const o = OUTCOMES[d.outcome] || { text: d.outcome };
    const out = el("span", `outcome ${d.outcome}`);
    if (o.icon) out.appendChild(svgIcon(o.icon));
    out.appendChild(document.createTextNode(o.text));
    if (d.outcome === "blocked") out.title = `A controller switch within ${F.min_dwell_s} s of the last one is dropped (dwell-time guarantee); the supervisor is asked again.`;
    if (d.outcome === "pending") out.title = "The flight ended before this answer arrived.";
    const why = el("span", "why", d.reasoning);
    const src = d.latency_s > 0 ? `${d.source}, answered after ${fmt(d.latency_s, 1)} s` : d.source;
    why.appendChild(el("span", "src", `  (${src})`));
    li.append(el("span", "t", `${fmt(d.time, 2)} s`), pill, out, why);
    list.appendChild(li);
  });
}

function renderDiagnosis(d) {
  // The learned diagnoser's output format is not fixed yet: numeric maps (e.g. class probabilities) become bars,
  // everything else a key/value list. All text goes in with textContent.
  const box = $("diagnosis");
  box.replaceChildren();
  if (d === null || d === undefined) { box.appendChild(el("p", "hint", "No diagnosis yet.")); return; }
  if (typeof d !== "object") { box.appendChild(el("p", "", String(d))); return; }
  const nice = (k) => String(k).replace(/_/g, " ");
  const kv = el("dl", "diag-kv");
  for (const [k, v] of Object.entries(d)) {
    const numericMap = v && typeof v === "object" && !Array.isArray(v) && Object.keys(v).length &&
      Object.values(v).every((x) => finite(x));
    if (numericMap) {
      const block = el("div", "diag-rows");
      block.appendChild(el("h3", "", nice(k)));
      const entries = Object.entries(v).sort((a, b) => b[1] - a[1]);
      const probs = entries.every(([, x]) => x >= 0 && x <= 1);
      const max = probs ? 1 : Math.max(...entries.map(([, x]) => Math.abs(x)), 1e-12);
      for (const [name, x] of entries) {
        const row = el("div", "diag-row");
        const bar = el("div", "diag-bar"), fill = el("i");
        fill.style.width = `${Math.round((100 * Math.abs(x)) / max)}%`;
        bar.appendChild(fill);
        row.append(el("span", "", faultLabel(name)), bar, el("span", "diag-val", probs ? `${fmt(100 * x, 0)} %` : fmt(x, 2)));
        block.appendChild(row);
      }
      box.appendChild(block);
    } else {
      const text = finite(v) ? fmt(v, 3) : typeof v === "object" && v !== null ? JSON.stringify(v) : String(v);
      kv.append(el("dt", "", nice(k)), el("dd", "", k === "fault" || k === "label" ? faultLabel(v) : text));
    }
  }
  if (kv.children.length) box.appendChild(kv);
}

// ---------------------------------------------------------------- playback
function setPlaying(on) {
  const F = state.flight;
  if (on && F && state.simTime >= F._end - 1e-6) state.simTime = 0;
  state.playing = Boolean(on && F);
  $("play-btn").classList.toggle("playing", state.playing);
  $("play-btn").setAttribute("aria-label", state.playing ? "Pause" : "Play");
}

function frame(now) {
  requestAnimationFrame(frame); // first, so an error below can never stop the animation loop
  try {
    const wallDt = state.lastFrame === null ? 0 : Math.min(0.1, (now - state.lastFrame) / 1000);
    state.lastFrame = now;
    const speed = Number($("play-speed").value) || 1;
    const F = state.flight;
    if (F && state.playing) {
      state.simTime += wallDt * speed;
      if (state.simTime >= F._end) { state.simTime = F._end; setPlaying(false); }
    }
    if (F) {
      const i = Math.max(0, lastAtOrBefore(F.series.t, state.simTime));
      const mode = F.modes[F.series.mode[i]];
      const ctrl = CONTROLLERS[mode ? mode.controller : "lqg"] || CONTROLLERS.lqg;
      state.lane.color = ctrl.color;
      state.lane.label = `${ctrl.label} active`;
    }
    viewport.render(state.simTime, state.playing ? wallDt : 0, speed);
    $("time-readout").textContent = `t = ${state.simTime.toFixed(2)} s`;
    if (F && document.activeElement !== $("scrub")) $("scrub").value = String(Math.round((state.simTime / F._end) * 1000));
    if (F && now - state.lastPanels > 80) {
      for (const c of Object.values(charts)) c.setPlayhead(state.simTime);
      if (state.playhead) state.playhead.style.left = `${Math.min(100, (100 * state.simTime) / F.request.duration)}%`;
      updatePanels();
      state.lastPanels = now;
    }
  } catch (err) {
    if (!state.displayError) {
      state.displayError = true;
      showBanner(`Display error: ${err.message}`);
      console.error(err);
    }
  }
}

function redrawAll() {
  Object.values(charts).forEach((c) => c.draw());
  viewport.render(state.simTime, 0);
}

// ---------------------------------------------------------------- downloads (the table view of every chart)
function download(name, type, text) {
  const blob = new Blob([text], { type });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function stem() {
  const r = state.flight.request;
  return `guardian_${r.fault}_${r.supervisor.replace("+", "-")}_seed${r.seed}`;
}

function downloadCSV() {
  const F = state.flight;
  if (!F) return;
  const s = F.series;
  const header = ["t_s", "h_mm", "alpha_deg", "beta_deg", "beta_cmd_deg", "airspeed_mps", "mode", "fault_level_of_benchmark",
    "est_flutter_speed_mps", "est_flutter_speed_std_mps", "true_flutter_speed_mps"];
  const rows = [header.join(",")];
  const bench = F.fault.benchmark_severity || 1;
  const v = (x, k = 1, d = 6) => (finite(x) ? (x * k).toFixed(d) : "");
  for (let i = 0; i < s.t.length; i++) {
    const t = s.t[i];
    const e = F.estimates[lastAtOrBefore(F._estTimes, t)];
    const tr = lastAtOrBefore(F.truth.t, t);
    rows.push([
      t, v(s.h[i], 1000, 4), v(s.alpha[i], DEG, 4), s.beta ? v(s.beta[i], DEG, 4) : "",
      s.u && s.u[0] ? v(s.u[0][i], DEG, 4) : "", v(s.airspeed[i], 1, 3), s.mode[i], v(s.fault_level[i] / bench, 1, 4),
      e ? v(e.flutter_speed, 1, 3) : "", e ? v(e.flutter_speed_std, 1, 3) : "",
      tr >= 0 ? v(F.truth.flutter_speed[tr], 1, 3) : "",
    ].join(","));
  }
  download(`${stem()}.csv`, "text/csv", rows.join("\n"));
}

function downloadJSON() {
  const F = state.flight;
  if (!F) return;
  const clean = Object.fromEntries(Object.entries(F).filter(([k]) => !k.startsWith("_")));
  download(`${stem()}.json`, "application/json", JSON.stringify(clean, null, 1));
}

// ---------------------------------------------------------------- events + boot
function bindEvents() {
  $("fly-btn").addEventListener("click", fly);
  $("play-btn").addEventListener("click", () => setPlaying(!state.playing));
  $("restart-btn").addEventListener("click", () => { state.simTime = 0; setPlaying(true); });
  $("scrub").addEventListener("input", (e) => {
    const F = state.flight;
    if (!F) return;
    setPlaying(false);
    state.simTime = (Number(e.target.value) / 1000) * F._end;
  });
  $("preset").addEventListener("change", () => { updatePresetReadouts(); renderStatusChips(); });
  $("fault").addEventListener("change", applyFaultDefaults);
  $("severity").addEventListener("input", updateFaultHint);
  $("truth-toggle").addEventListener("change", (e) => {
    state.showTruth = e.target.checked;
    renderCharts();
    updatePanels(true);
  });
  $("kept-toggle").addEventListener("change", (e) => { state.showKept = e.target.checked; updatePanels(); });
  $("csv-btn").addEventListener("click", downloadCSV);
  $("json-btn").addEventListener("click", downloadJSON);
  $("theme-toggle").addEventListener("click", () => {
    const dark = cssVar("--page").toLowerCase() === "#0d0d0d";
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("flutterlab-theme", next); } catch (e) { /* storage unavailable: theme not remembered */ }
    redrawAll();
  });
  if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);
}

function renderEmpty() {
  $("report").textContent = "Fly a mission to see the supervisor's reports, one every 0.25 s of flight.";
  $("decision-log").replaceChildren(el("li", "empty", "The supervisor's answers and their reasoning appear here during playback."));
  const g = $("gauge");
  g.replaceChildren(el("div", "empty", "Fly a mission to see the flutter margin"));
  const box = $("lights");
  box.replaceChildren();
  for (const m of state.options ? state.options.mode_order : []) {
    const card = el("div", "light");
    const head = el("div", "light-head");
    head.append(meter(m), document.createTextNode(m));
    const spec = state.options.modes[m] || {};
    const ctrl = (CONTROLLERS[spec.controller] || { label: spec.controller || "" }).label;
    const sub = spec.speed_ratio === null || spec.speed_ratio === undefined
      ? `${ctrl} at cruise` : `${ctrl}, slow to ${spec.speed_ratio} × healthy U_F`;
    card.title = spec.description || "";
    card.append(head, el("div", "light-sub", sub), statusCell("unknown", "No estimate yet"));
    box.appendChild(card);
  }
}

async function boot() {
  bindEvents();
  requestAnimationFrame(frame);
  try {
    state.options = await api.guardianOptions();
  } catch (err) {
    renderEmpty();
    showBanner(`Could not reach the Flutter Lab server (${err.message}). Start it with: python -m flutter_ai serve`);
    return;
  }
  fillForm();
  renderStatusChips();
  renderEmpty();
}

boot();
