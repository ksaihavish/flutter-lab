import { api } from "./api.js";
import { LineChart, cssVar } from "./chart.js";
import { Viewport } from "./viewport.js";

// Colour follows the entity: each case keeps its colour everywhere (docs/Design.md).
const CASES = [
  { id: "open_loop", controller: "open_loop", supervisor: "none", label: "Open loop", color: "--c-open_loop", on: true, width: 1.5 },
  { id: "lqr", controller: "lqr", supervisor: "none", label: "LQR", color: "--c-lqr", on: true },
  { id: "neural", controller: "neural", supervisor: "none", label: "Neural (physics-trained)", color: "--c-neural", on: true, needsNeural: true },
  { id: "lqr+rules", controller: "lqr", supervisor: "rules", label: "LQR + rule supervisor", color: "--c-lqr-rules", on: false },
  { id: "neural+rules", controller: "neural", supervisor: "rules", label: "Neural + rule supervisor", color: "--c-neural-rules", on: true, needsNeural: true },
  { id: "neural+llm", controller: "neural", supervisor: "llm", label: "Neural + Claude supervisor", color: "--c-llm", on: false, needsNeural: true, needsLLM: true },
];

const SCENARIO_LABELS = {
  turbulence_patch: "Turbulence patch",
  discrete_gust: "Discrete gust",
  high_speed_gust: "High speed",
  below_flutter: "Below flutter",
};

const $ = (id) => document.getElementById(id);
const state = {
  health: null, presets: [], scenarios: {}, scenarioName: "turbulence_patch",
  flutter: null, response: null, shownCases: [],
  playing: false, simTime: 0, lastFrame: null, lastChartDraw: 0,
};

const charts = {
  h: new LineChart($("chart-h"), { symmetric: true }),
  a: new LineChart($("chart-a"), { symmetric: true }),
  u: new LineChart($("chart-u"), { yMin: -1.05, yMax: 1.05, yFormat: 1 }),
  beta: new LineChart($("chart-beta"), { symmetric: true, yFormat: 1 }),
  agg: new LineChart($("chart-agg"), { yMin: 0, yMax: 1.05, step: true, yFormat: 1 }),
  gust: new LineChart($("chart-gust"), { symmetric: true, yFormat: 1 }),
  vg: new LineChart($("chart-vg"), { xUnit: "m/s", xLabel: "U", xDigits: 1, playhead: false }),
};
charts.agg.emptyText = "No supervised controller in this run";
const viewport = new Viewport($("viewport"));

// ---------------------------------------------------------------- utilities
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
const svgIcon = (path) => {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("aria-hidden", "true");
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p.setAttribute("d", path);
  s.appendChild(p);
  return s;
};
const ICON_OK = "M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z";
const ICON_WARN = "M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z";
const ICON_OFF = "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm0 18a8 8 0 1 1 0-16 8 8 0 0 1 0 16z";
const num = (v, d = 1) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));

function showBanner(msg) {
  const b = $("banner");
  b.textContent = msg;
  b.hidden = !msg;
}

// ---------------------------------------------------------------- setup
function renderStatus() {
  const row = $("status-row");
  row.replaceChildren();
  const chip = (ok, text, title) => {
    const c = el("span", `chip ${ok === true ? "ok" : ok === false ? "off" : "bad"}`);
    c.append(svgIcon(ok === true ? ICON_OK : ok === false ? ICON_OFF : ICON_WARN), document.createTextNode(text));
    if (title) c.title = title;
    return c;
  };
  const h = state.health;
  const preset = $("preset").value;
  const neural = neuralAvailable();
  row.append(
    chip(true, "Physics model"),
    chip(neural, neural ? "Neural policy" : "Neural policy not trained",
      neural ? "" : `Run: python -m flutter_ai train --preset ${preset}`),
    chip(h.llm.api_key_set && h.llm.installed, h.llm.api_key_set ? `Claude (${h.llm.model})` : "Claude: no API key",
      h.llm.api_key_set ? "" : "Set ANTHROPIC_API_KEY before starting the server to enable the Claude supervisor"),
  );
}

function currentPreset() {
  return state.presets.find((x) => x.name === $("preset").value);
}

function neuralAvailable() {
  return Boolean(currentPreset()?.neural_policy_available);
}

function renderCaseOptions() {
  for (const c of CASES) if (c.defaultOn === undefined) c.defaultOn = c.on;
  const list = $("case-list");
  list.replaceChildren();
  for (const c of CASES) {
    const disabled = (c.needsNeural && !neuralAvailable()) ||
      (c.needsLLM && !(state.health.llm.api_key_set && state.health.llm.installed));
    if (disabled) c.on = false;
    else if (c.defaultOn !== undefined && c.wasDisabled) c.on = c.defaultOn;
    c.wasDisabled = disabled;
    const label = el("label", `case-option${disabled ? " disabled" : ""}`);
    const box = document.createElement("input");
    box.type = "checkbox"; box.checked = c.on; box.disabled = disabled;
    box.addEventListener("change", () => { c.on = box.checked; updateLLMHint(); });
    const key = el("span", "key-line");
    key.style.background = `var(${c.color})`;
    label.append(box, key, document.createTextNode(c.label));
    list.appendChild(label);
  }
  updateLLMHint();
}

function updateLLMHint() {
  const llm = CASES.find((c) => c.needsLLM);
  const h = state.health.llm;
  let text = "";
  if (!(h.api_key_set && h.installed)) text = "Claude supervisor: set ANTHROPIC_API_KEY and restart to enable.";
  else if (llm.on) {
    const calls = Math.min(40, Math.ceil(Number($("duration").value) / Number($("sup-interval").value)));
    text = `Claude supervisor makes about ${calls} API calls per run (capped at 40). Its measured latency is applied in simulation time, so expect a slower run.`;
  }
  $("llm-hint").textContent = text;
}

function renderScenarioButtons() {
  const wrap = $("scenario-buttons");
  wrap.replaceChildren();
  for (const [name] of Object.entries(state.scenarios)) {
    const b = el("button", "", SCENARIO_LABELS[name] || name);
    b.type = "button";
    b.setAttribute("aria-pressed", String(name === state.scenarioName));
    b.addEventListener("click", () => { applyScenario(name); run(); });
    wrap.appendChild(b);
  }
}

function applyScenario(name) {
  state.scenarioName = name;
  const s = state.scenarios[name];
  $("speed").value = s.velocity_ratio;
  $("duration").value = s.duration;
  $("delay").value = s.input_delay_steps;
  $("h0").value = Math.round(s.h0 * 1000);
  $("a0").value = s.alpha0_deg;
  $("gust-kind").value = s.gust.kind;
  $("gust-amp").value = s.gust.amplitude;
  $("gust-start").value = s.gust.start;
  $("gust-dur").value = s.gust.duration;
  $("sup-interval").value = s.supervisor_interval;
  $("scenario-description").textContent = s.description;
  renderScenarioButtons();
  updateOutputs();
}

function updateOutputs() {
  const ratio = Number($("speed").value);
  const uf = state.flutter?.flutter_speed;
  $("out-speed").textContent = `${Math.round(ratio * 100)}% of U_F${uf ? ` · ${(ratio * uf).toFixed(1)} m/s` : ""}`;
  const kind = $("gust-kind").value;
  const amp = Number($("gust-amp").value).toFixed(1);
  $("out-gust").textContent = kind === "none" ? "–" : kind === "turbulence" ? `${amp} m/s RMS` : `${amp} m/s peak`;
  drawFlutterChart();
}

async function loadFlutter() {
  const preset = $("preset").value;
  state.flutter = await api.flutter(preset);
  const p = state.presets.find((x) => x.name === preset);
  $("ro-flutter").textContent = `${state.flutter.flutter_speed.toFixed(1)} m/s`;
  $("ro-freq").textContent = `${state.flutter.flutter_frequency_hz.toFixed(2)} Hz`;
  $("ro-mu").textContent = p.derived.mass_ratio.toFixed(1);
  $("ro-dof").textContent = p.n_dof === 3 ? "3 (plunge, pitch, flap)" : "2 (plunge, pitch)";
  $("ro-act").textContent = p.actuation === "flap_servo"
    ? `Flap servo ±${((p.u_max[0] * 180) / Math.PI).toFixed(0)}°` : "Force + moment (idealised)";
  updateOutputs();
}

function drawFlutterChart() {
  const f = state.flutter;
  if (!f) return;
  const current = Number($("speed").value) * f.flutter_speed;
  const leg = $("vg-legend");
  leg.replaceChildren();
  f.damping.forEach((_, k) => {
    const item = el("span", "legend-item"), key = el("span", "key-line");
    key.style.background = `var(--c-mode${k + 1})`;
    item.append(key, document.createTextNode(`Mode ${k + 1}`));
    leg.appendChild(item);
  });
  charts.vg.setData(
    f.damping.map((d, k) => ({ key: `m${k + 1}`, label: `Mode ${k + 1}`, color: `--c-mode${k + 1}`, x: f.velocities, y: d })),
    {
      hlines: [{ y: 0, color: "--axis" }],
      vlines: [
        { x: f.flutter_speed, label: `U_F = ${f.flutter_speed.toFixed(1)} m/s`, color: "--ink-2" },
        { x: current, label: `current ${current.toFixed(1)} m/s`, color: "--ink", row: 1 },
      ],
    },
  );
}

// ---------------------------------------------------------------- run
function requestBody() {
  return {
    preset: $("preset").value,
    scenario: {
      velocity_ratio: Number($("speed").value),
      duration: Number($("duration").value),
      h0_mm: Number($("h0").value),
      alpha0_deg: Number($("a0").value),
      input_delay_steps: Number($("delay").value),
      gust: {
        kind: $("gust-kind").value,
        amplitude: Number($("gust-amp").value),
        start: Number($("gust-start").value),
        duration: Number($("gust-dur").value),
        seed: state.scenarios[state.scenarioName]?.gust.seed ?? 7,
      },
      supervisor_interval: Number($("sup-interval").value),
    },
    cases: CASES.filter((c) => c.on).map((c) => ({ controller: c.controller, supervisor: c.supervisor })),
  };
}

async function run() {
  const body = requestBody();
  if (!body.cases.length) { showBanner("Select at least one controller."); return; }
  showBanner("");
  const btn = $("run-btn");
  btn.disabled = true;
  btn.textContent = CASES.some((c) => c.on && c.needsLLM) ? "Running (consulting Claude)…" : "Running…";
  document.body.classList.add("is-loading");
  try {
    const shown = CASES.filter((c) => c.on);
    const res = await api.simulate(body);
    state.response = res;
    state.shownCases = shown.map((c, k) => ({ ...c, case: res.results[k].case, result: res.results[k] }));
    const warnings = [...new Set(res.results.flatMap((r) => r.warnings))];
    if (warnings.length) showBanner(`Note: ${warnings.join(" ")}`);
    renderResults();
    state.simTime = 0;
    setPlaying(true);
  } catch (err) {
    showBanner(`Simulation failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run simulation";
    document.body.classList.remove("is-loading");
  }
}

function renderResults() {
  const res = state.response;
  const cases = state.shownCases;
  const p = res.params;
  viewport.setData(res, cases);

  // legend (always present for >= 2 series)
  const legend = $("legend");
  legend.replaceChildren();
  for (const c of cases) {
    const item = el("span", "legend-item");
    const key = el("span", "key-line");
    key.style.background = `var(${c.color})`;
    item.append(key, document.createTextNode(c.label));
    legend.appendChild(item);
  }

  const series = (fn, filter = () => true) =>
    cases.filter(filter).map((c) => ({
      key: c.id, label: c.label, color: c.color, width: c.width,
      x: c.result.t, y: fn(c.result), endMarker: c.result.exceeded_limits,
    }));
  const hLim = p.h_limit * 1000, aLim = p.alpha_limit_deg;
  charts.h.setData(series((r) => r.h.map((v) => v * 1000)), {
    hlines: [{ y: hLim, label: "structural limit", color: "--critical", includeInRange: false },
             { y: -hLim, color: "--critical", includeInRange: false }],
  });
  charts.a.setData(series((r) => r.alpha.map((v) => (v * 180) / Math.PI)), {
    hlines: [{ y: aLim, label: "structural limit", color: "--critical", includeInRange: false },
             { y: -aLim, color: "--critical", includeInRange: false }],
  });
  const j = p.primary_input;
  $("chart-u-title").textContent = p.actuation === "flap_servo"
    ? `Flap command β_cmd (fraction of ±${((p.u_max[0] * 180) / Math.PI).toFixed(0)}° limit)`
    : "Pitch-moment command M_α (fraction of actuator limit)";
  charts.u.setData(series((r) => r.u[j].map((v) => v / p.u_max[j]), (c) => c.controller !== "open_loop"), {
    hlines: [{ y: 1, label: "saturation", color: "--axis" }, { y: -1, color: "--axis" }],
  });
  $("card-beta").hidden = p.n_dof !== 3;
  if (p.n_dof === 3) {
    const lim = (p.u_max[0] * 180) / Math.PI;
    charts.beta.setData(series((r) => r.beta.map((v) => (v * 180) / Math.PI)), {
      hlines: p.actuation === "flap_servo"
        ? [{ y: lim, label: "command limit", color: "--axis", includeInRange: false }, { y: -lim, color: "--axis", includeInRange: false }]
        : [],
    });
  }
  $("viewport-note").textContent = p.n_dof === 3
    ? "True scale (plunge and chord share one scale). Air flows left to right. Solid flap = actual angle; thin outline = commanded angle (the gap is servo lag)."
    : "True scale (plunge and chord share one scale). Air flows left to right. Arrows show the idealised plunge force and pitch moment.";
  charts.agg.setData(series((r) => r.aggressiveness, (c) => c.supervisor !== "none"));
  const first = cases[0].result;
  const longest = cases.reduce((a, c) => (c.result.t.length > a.t.length ? c.result : a), first);
  charts.gust.setData([{ key: "gust", label: "Gust", color: "--ink-2", x: longest.t, y: longest.w_gust, width: 1.5 }]);

  renderTable(cases);
  renderDecisions(cases);
  $("scrub").max = 1000;
}

function renderTable(cases) {
  const t = $("metrics-table");
  t.replaceChildren();
  const cols = [
    ["Controller"], ["Status"], ["Peak plunge", "mm"], ["RMS plunge", "mm"], ["Peak pitch", "°"],
    ["Settling", "s"], ["Actuator effort", "s"], ["Command rate (RMS)"], ["Saturated", "%"], ["Mode switches"],
  ];
  const thead = el("thead"), hr = el("tr");
  for (const [name, unit] of cols) hr.appendChild(el("th", "", unit ? `${name} (${unit})` : name));
  thead.appendChild(hr);
  const tbody = el("tbody");
  for (const c of cases) {
    const m = c.result.metrics;
    const tr = el("tr");
    const name = el("td"), wrap = el("span", "case-cell"), key = el("span", "key-line");
    key.style.background = `var(${c.color})`;
    wrap.append(key, document.createTextNode(c.label));
    name.appendChild(wrap);
    const status = el("td"), s = el("span", `status-cell ${m.exceeded_limits ? "status-fail" : "status-ok"}`);
    s.append(svgIcon(m.exceeded_limits ? ICON_WARN : ICON_OK),
      document.createTextNode(m.exceeded_limits ? `Limit exceeded at ${num(m.failure_time_s, 2)} s` : "Within limits"));
    status.appendChild(s);
    tr.append(name, status,
      el("td", "", num(m.peak_h_mm, 1)), el("td", "", num(m.rms_h_mm, 2)), el("td", "", num(m.peak_alpha_deg, 2)),
      el("td", "", m.exceeded_limits ? "–" : num(m.settling_time_s, 2)),
      el("td", "", c.controller === "open_loop" ? "–" : num(m.control_energy, 3)),
      el("td", "", c.controller === "open_loop" ? "–" : num(m.control_rate_rms, 3)),
      el("td", "", c.controller === "open_loop" ? "–" : num(m.saturation_fraction * 100, 1)),
      el("td", "", c.supervisor === "none" ? "–" : String(m.mode_switches)));
    tbody.appendChild(tr);
  }
  t.append(thead, tbody);
}

function renderDecisions(cases) {
  const list = $("decision-log");
  list.replaceChildren();
  const levels = { efficient: 1, balanced: 2, aggressive: 3, emergency: 4 };
  const items = [];
  for (const c of cases.filter((x) => x.supervisor !== "none")) {
    let prev = "balanced";
    for (const d of c.result.decisions) {
      if (d.mode !== prev || d.source !== "rules") items.push({ c, d });
      prev = d.mode;
    }
  }
  items.sort((a, b) => a.d.time - b.d.time);
  if (!items.length) {
    list.appendChild(el("li", "empty", "Mode changes and their reasoning appear here when a supervised controller is selected."));
    return;
  }
  for (const { c, d } of items) {
    const li = el("li");
    const who = el("span", "who"), key = el("span", "key-line");
    key.style.background = `var(${c.color})`;
    who.append(key, document.createTextNode(c.label));
    const pill = el("span", "mode-pill"), meter = el("span", "meter");
    for (let k = 1; k <= 4; k++) meter.appendChild(el("i", k <= levels[d.mode] ? "on" : ""));
    pill.append(meter, document.createTextNode(d.mode));
    const why = el("span", "why", d.reasoning);
    const src = d.source === "llm" ? `Claude · ${d.latency_s.toFixed(1)} s` : d.source === "llm_fallback" ? "fallback to rules" : "rules";
    why.appendChild(el("span", "src", `  (${src})`));
    li.append(el("span", "t", `${d.time.toFixed(2)} s`), who, pill, why);
    list.appendChild(li);
  }
}

// ---------------------------------------------------------------- playback
function duration() {
  if (!state.shownCases.length) return 0;
  return Math.max(...state.shownCases.map((c) => c.result.t[c.result.t.length - 1]));
}

function setPlaying(on) {
  if (on && state.simTime >= duration() - 1e-6) state.simTime = 0;
  state.playing = on;
  $("play-btn").classList.toggle("playing", on);
  $("play-btn").setAttribute("aria-label", on ? "Pause" : "Play");
}

function frame(now) {
  const wallDt = state.lastFrame === null ? 0 : Math.min(0.1, (now - state.lastFrame) / 1000);
  state.lastFrame = now;
  const speed = Number($("play-speed").value);
  if (state.playing && state.shownCases.length) {
    state.simTime += wallDt * speed;
    if (state.simTime >= duration()) { state.simTime = duration(); setPlaying(false); }
  }
  viewport.render(state.simTime, state.playing ? wallDt : 0, speed);
  $("time-readout").textContent = `t = ${state.simTime.toFixed(2)} s`;
  const d = duration();
  if (d > 0 && document.activeElement !== $("scrub")) $("scrub").value = String(Math.round((state.simTime / d) * 1000));
  if (now - state.lastChartDraw > 60) {
    for (const k of ["h", "a", "u", "beta", "agg", "gust"]) charts[k].setPlayhead(state.simTime);
    state.lastChartDraw = now;
  }
  requestAnimationFrame(frame);
}

function redrawAll() {
  Object.values(charts).forEach((c) => c.draw());
  viewport.render(state.simTime, 0);
}

// ---------------------------------------------------------------- events
function bindEvents() {
  $("run-btn").addEventListener("click", run);
  $("play-btn").addEventListener("click", () => setPlaying(!state.playing));
  $("restart-btn").addEventListener("click", () => { state.simTime = 0; setPlaying(true); });
  $("scrub").addEventListener("input", (e) => {
    setPlaying(false);
    state.simTime = (Number(e.target.value) / 1000) * duration();
  });
  for (const id of ["speed", "gust-amp", "gust-kind"]) $(id).addEventListener("input", updateOutputs);
  for (const id of ["duration", "sup-interval"]) $(id).addEventListener("input", updateLLMHint);
  $("preset").addEventListener("change", async () => {
    renderStatus();
    renderCaseOptions();
    state.scenarios = await api.scenarios($("preset").value);  // disturbances are scaled per rig
    applyScenario(state.scenarioName);
    await loadFlutter();
    run();
  });
  $("theme-toggle").addEventListener("click", () => {
    const dark = cssVar("--page").toLowerCase() === "#0d0d0d";
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("flutterlab-theme", next); } catch (e) { /* storage unavailable */ }
    redrawAll();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);
  $("download-btn").addEventListener("click", downloadCSV);
}

function downloadCSV() {
  if (!state.shownCases.length) return;
  const cases = state.shownCases;
  const longest = cases.reduce((a, c) => (c.result.t.length > a.result.t.length ? c : a), cases[0]).result;
  const p = state.response.params;
  const header = ["t_s", "gust_mps"];
  const inputCols = p.inputs.map((inp) => `${inp.name}_${inp.unit.replace(/[^a-zA-Z]/g, "")}`);
  for (const c of cases) {
    header.push(`${c.id}_h_mm`, `${c.id}_alpha_deg`);
    if (p.n_dof === 3) header.push(`${c.id}_beta_deg`);
    header.push(...inputCols.map((col) => `${c.id}_${col}`), `${c.id}_mode`);
  }
  const width = 3 + (p.n_dof === 3 ? 1 : 0) + p.inputs.length;
  const rows = [header.join(",")];
  for (let i = 0; i < longest.t.length; i++) {
    const row = [longest.t[i], longest.w_gust[i]];
    for (const c of cases) {
      const r = c.result;
      if (i < r.t.length) {
        row.push((r.h[i] * 1000).toFixed(4), ((r.alpha[i] * 180) / Math.PI).toFixed(4));
        if (p.n_dof === 3) row.push(((r.beta[i] * 180) / Math.PI).toFixed(4));
        row.push(...r.u.map((ch) => ch[i]), r.mode[i]);
      } else row.push(...Array(width).fill(""));
    }
    rows.push(row.join(","));
  }
  const blob = new Blob([rows.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `flutter_${state.scenarioName}_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---------------------------------------------------------------- boot
async function boot() {
  bindEvents();
  requestAnimationFrame(frame);
  try {
    [state.health, state.presets, state.scenarios] = await Promise.all([api.health(), api.presets(), api.scenarios("uav_flap")]);
  } catch (err) {
    showBanner(`Could not reach the Flutter Lab server (${err.message}). Start it with: python -m flutter_ai serve`);
    return;
  }
  const sel = $("preset");
  for (const p of state.presets) {
    const o = el("option", "", `${p.name.replace(/_/g, " ")} (${p.n_dof}-DOF)`);
    o.value = p.name;
    o.title = p.description;
    sel.appendChild(o);
  }
  renderStatus();
  renderCaseOptions();
  applyScenario(state.scenarioName);
  await loadFlutter();
  await run();
}

boot();
