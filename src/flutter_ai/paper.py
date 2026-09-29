"""
Paper draft generator (Phase 4.8): every number and every comparative statement in the draft is computed
from the result files, so nothing is typed by hand (Rules.md §1).

    python -m flutter_ai paper        # -> docs/research/paper-draft.md

Inputs (run these first): outputs/<preset>/results.json (evaluate), robustness.json (robustness),
ablations.json (ablations), llm_benchmark.json (llm-benchmark), outputs/validation/bact.json (validate),
models/classical_tuning_<preset>.json (tune). Missing inputs become a visible "not run yet" note.
Edit the prose here; never edit numbers in the generated file.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .params import PRESETS
from .paths import MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT

PRESET_ORDER = ("uav_flap", "uav_section", "bact")
SCEN_LABEL = {"turbulence_patch": "Turbulence, 1.1 U_F", "discrete_gust": "Discrete gust, 1.1 U_F",
              "high_speed_gust": "Gust, 1.3 U_F", "below_flutter": "Gust, 0.9 U_F"}
CTRL_LABEL = {"open_loop": "Open loop", "lqr": "LQR", "lqg": "LQG", "neural": "Neural",
              "lqr+rules": "LQR + rules", "neural+rules": "Neural + rules"}


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _pct(x: float) -> str:
    return f"{100 * x:+.0f} %"


def _p(p: float) -> str:
    return "p < 0.001" if p < 1e-3 else f"p = {p:.2g}"


def _cell(agg: dict) -> str:
    s = f"{agg['rms_h_mm']['mean']:.2f} ± {agg['rms_h_mm']['std']:.2f}"
    if agg.get("cost_J"):
        s += f" · J {agg['cost_J']['mean']:.3g}"
    if agg["failures"]:
        s += f" · **{agg['failures']}/{agg['n']} failed**"
    return s


def _results_table(r: dict) -> list[str]:
    cases = [c for c in ("open_loop", "lqr", "lqg", "neural") if c in next(iter(r["scenarios"].values()))["cases"]]
    lines = ["| Scenario | " + " | ".join(CTRL_LABEL[c] for c in cases) + " |", "|---" * (len(cases) + 1) + "|"]
    for s, d in r["scenarios"].items():
        lines.append(f"| {SCEN_LABEL.get(s, s)} | " + " | ".join(_cell(d["cases"][c]) for c in cases) + " |")
    return lines


def _best_by_J(d: dict) -> tuple[str, int]:
    """(controller with the lowest mean J among those that never failed, number of failure-free controllers)."""
    ok = {c: a["cost_J"]["mean"] for c, a in d["cases"].items() if "+" not in c and a.get("cost_J") and not a["failures"]}
    return (min(ok, key=lambda c: ok[c]), len(ok)) if ok else ("none", 0)


def _stat_sentence(d: dict, key: str, a: str, b: str) -> str:
    st = d["statistics"].get(key)
    if not st or "rel_diff" not in st.get("cost_J", {}):
        return f"{a} vs {b}: not comparable (too few runs where both survived)."
    j = st["cost_J"]
    word = "lower" if j["mean_diff"] < 0 else "higher"
    return (f"{a} has {abs(100 * j['rel_diff']):.0f} % {word} cost J than {b} "
            f"(95 % CI of the difference [{j['ci95'][0]:+.3g}, {j['ci95'][1]:+.3g}], Wilcoxon {_p(j['wilcoxon_p'])}, "
            f"n = {j['n']}; failures {st['failures']['candidate']} vs {st['failures']['reference']} of {st['n_seeds']}).")


def build(out_path: Path | None = None) -> Path:
    out_path = Path(out_path or PROJECT_ROOT / "docs" / "research" / "paper-draft.md")
    res = {p: _load(OUTPUTS_DIR / p / "results.json") for p in PRESET_ORDER}
    rob = {p: _load(OUTPUTS_DIR / p / "robustness.json") for p in PRESET_ORDER}
    tun = {p: _load(MODELS_DIR / f"classical_tuning_{p}.json") for p in PRESET_ORDER}
    abl = _load(OUTPUTS_DIR / "uav_flap" / "ablations.json")
    llm = _load(OUTPUTS_DIR / "uav_flap" / "llm_benchmark.json")
    bact = _load(OUTPUTS_DIR / "validation" / "bact.json")
    L: list[str] = []
    w = L.append

    w("# Paper draft (generated)\n")
    w(f"*Generated {datetime.now(timezone.utc).isoformat(timespec='minutes')} by `python -m flutter_ai paper` "
      f"(flutter_ai {__version__}). Every number comes from the result files listed in `src/flutter_ai/paper.py`; "
      "edit the prose there, never the numbers here. Related work: `related-work.md`.*\n")
    w("**Working title:** *On-board language-model supervision for active flutter suppression: a fair benchmark*\n")

    # ---------------------------------------------------------------- abstract
    w("## Abstract (draft)\n")
    r = res["uav_flap"]
    if r:
        n_seeds = len(r["metadata"]["seeds"])
        wins: dict[str, int] = {}
        for d in r["scenarios"].values():
            c, _ = _best_by_J(d)
            wins[c] = wins.get(c, 0) + 1
        best = ", ".join(f"{CTRL_LABEL.get(c, c)} in {k}" for c, k in sorted(wins.items(), key=lambda kv: -kv[1]))
        w(f"Active flutter suppression is studied on a validated typical-section model with a servo-driven "
          f"trailing-edge flap, a realistic actuator slew-rate limit and a 5 ms loop delay. Three inner-loop "
          f"controllers minimise the same quadratic cost: LQR, a turbulence-aware LQG and a neural policy trained "
          f"through differentiable simulation; the classical designs' free knobs are tuned for the real actuator on "
          f"episodes disjoint from the test set. Over {n_seeds} randomised runs per scenario, the lowest-cost "
          f"failure-free controller per scenario is: {best}. A language-model supervisor running on board "
          + (f"(median decision latency {llm['summary']['latency_s']['median']:.1f} s) " if llm else "")
          + "is evaluated against rule-based and random supervisors under its measured latency. "
          "*[Finish after the fault-tolerant supervision study.]*\n")
    else:
        w("*Run `python -m flutter_ai evaluate` first.*\n")

    # ---------------------------------------------------------------- model & validation
    w("## 1. Model and validation\n")
    for p in PRESET_ORDER:
        rr = res[p]
        if rr:
            f = rr["flutter"]
            pk = f.get("pk_exact_theodorsen")
            agree = f" (independent p-k with exact Theodorsen: {pk:.2f} m/s, {100 * (f['state_space'] - pk) / pk:+.2f} %)" if pk else ""
            w(f"- **{p}**: flutter at {f['state_space']:.2f} m/s, {f['frequency_hz']:.2f} Hz{agree}. "
              f"{PRESETS[p].description}")
    if bact:
        m = bact["reference"]["measured"]
        pr = next(iter(bact["predictions"].values()))  # the preset (first variant)
        if pr.get("flutter_q_psf"):
            w(f"- **Wind-tunnel validation (NASA BACT):** predicted flutter {pr['flutter_q_psf']:.0f} psf at "
              f"{pr['flutter_freq_hz']:.2f} Hz vs measured ≈{m['flutter_q_psf']:.0f} psf at ≈{m['flutter_freq_hz']:.0f} Hz "
              f"({100 * (pr['flutter_q_psf'] - m['flutter_q_psf']) / m['flutter_q_psf']:+.0f} % in dynamic pressure, "
              f"{100 * (pr['flutter_freq_hz'] - m['flutter_freq_hz']) / m['flutter_freq_hz']:+.1f} % in frequency). "
              "The pressure error is attributed to compressibility (Mach 0.77) in an incompressible model.")
    w("")

    # ---------------------------------------------------------------- controllers
    w("## 2. Controllers and fairness\n")
    w("All controllers see the same measurements, loop delay, actuator position and slew-rate limits, and minimise "
      "the same cost J = Σ xᵀQx + c(a)·uᵀRu + ΔuᵀWΔu (`controllers/costs.py`). LQR and LQG are designed on the "
      "linear model; LQG adds an AR(1) gust model and a Kalman filter; the neural policy is trained through the "
      "rate-limited plant. Design knobs tuned for the actual actuator (never on test seeds):\n")
    for p in PRESET_ORDER:
        t = tun[p]
        if t:
            w(f"- {p}: LQR rate multiplier {t['lqr']['rate_multiplier']:g}; LQG rate multiplier "
              f"{t['lqg']['rate_multiplier']:g}, assumed sensor noise {t['lqg']['measurement_noise']:g} of reference scale.")
    w("")

    # ---------------------------------------------------------------- results
    w("## 3. Results\n")
    w("Cells: RMS plunge (mm, mean ± std over seeds) · mean cost J over surviving runs (lower is better) · failures.\n")
    for p in PRESET_ORDER:
        rr = res[p]
        if not rr:
            w(f"*{p}: not run yet.*\n")
            continue
        w(f"### {p} ({len(rr['metadata']['seeds'])} seeds)\n")
        L.extend(_results_table(rr))
        w("")
        for s, d in rr["scenarios"].items():
            best, n_ok = _best_by_J(d)
            w(f"- *{SCEN_LABEL.get(s, s)}*: lowest J without failures: **{CTRL_LABEL.get(best, best)}** "
              f"({n_ok} controllers failure-free). {_stat_sentence(d, 'neural vs lqg', 'Neural', 'LQG')} "
              f"{_stat_sentence(d, 'lqg vs lqr', 'LQG', 'LQR')}")
        ds = rr["delay_sweep"]
        tol = {c: max([x["delay_steps"] * ds["dt"] * 1000 for x in ds[c] if not x["exceeded_limits"]] or [float("nan")])
               for c in ds if c not in ("delays", "dt")}
        w("- Delay tolerance (ms, discrete gust): " + ", ".join(f"{c} {v:.0f}" for c, v in tol.items() if not math.isnan(v)) + ".")
        vs = rr["velocity_sweep"]
        env = {c: max([x["ratio"] for x in vs[c] if not x["exceeded_limits"]] or [0.0]) for c in vs if c != "ratios"}
        w("- Envelope with a gust (highest U/U_F survived): " + ", ".join(f"{c} {v:.2f}" for c, v in env.items()) + ".\n")

    # ---------------------------------------------------------------- robustness
    w("## 4. Robustness (controllers designed for the nominal wing)\n")
    for p in PRESET_ORDER:
        rb = rob[p]
        if not rb:
            w(f"*{p}: not run yet.*")
            continue
        ctrls = rb["controllers"]
        w(f"**{p}** (failed runs out of {rb['metadata']['draws']}, turbulence):\n")
        w("| Condition | " + " | ".join(CTRL_LABEL.get(c, c) for c in ctrls) + " |")
        w("|---" * (len(ctrls) + 1) + "|")
        for lv, row in rb["parameters"]["turbulence_patch"].items():
            w(f"| model error ±{100 * float(lv):.0f} % | " + " | ".join(str(row[c]["failures"]) for c in ctrls) + " |")
        for nz, row in rb["sensor_noise"].items():
            w(f"| sensor noise {100 * float(nz):.0f} % | " + " | ".join(str(row[c]["failures"]) for c in ctrls) + " |")
        for lim, row in rb.get("rate_limit", {}).items():
            w(f"| slew limit {lim}{'' if lim == 'none' else ' °/s'} | " + " | ".join(str(row[c]["failures"]) for c in ctrls) + " |")
        w("")

    # ---------------------------------------------------------------- ablations
    w("## 5. Ablations (uav_flap)\n")
    if abl:
        w("| Case | " + " | ".join(SCEN_LABEL[s] for s in abl["scenarios"]) + " |")
        w("|---" * (len(abl["scenarios"]) + 1) + "|")
        for case in abl["cases"]:
            cells = []
            for d in abl["scenarios"].values():
                a = d["cases"][case]
                st = d["statistics"].get(f"{case} vs neural", {}).get("cost_J", {})
                rel = f" ({_pct(st['rel_diff'])} J vs full)" if "rel_diff" in st else ""
                cells.append((f"J {a['cost_J']['mean']:.3g}" if a.get("cost_J") else f"{a['failures']}/{a['n']} failed") + rel)
            w(f"| {case} | " + " | ".join(cells) + " |")
        w("")
    else:
        w("*Not run yet (`python -m flutter_ai ablations`).*\n")

    # ---------------------------------------------------------------- LLM
    w("## 6. Language-model supervision (effort modes)\n")
    if llm:
        s = llm["summary"]
        w(f"Backend {llm['metadata']['backend']} / {llm['metadata']['model']}; {s['decisions']} decisions, "
          f"{s['fallbacks']} fallbacks; latency median {s['latency_s']['median']:.2f} s, 95th percentile "
          f"{s['latency_s']['p95']:.2f} s; same mode as the rules for {100 * s['agreement_with_rules']:.0f} % of reports.\n")
        for sc, d in llm["scenarios"].items():
            w(f"- *{SCEN_LABEL.get(sc, sc)}*: {_stat_sentence(d, 'neural+llm vs neural', 'Neural + LLM', 'Neural alone')} "
              f"{_stat_sentence(d, 'neural+llm vs neural+rules', 'Neural + LLM', 'Neural + rules')}")
        w("")
    else:
        w("*Not run yet (`python -m flutter_ai llm-benchmark`).*\n")

    w("## 7. Discussion points (to write)\n")
    w("- Why the ranking of controllers depends on the actuator model (slew-rate limit) and on disturbance knowledge.")
    w("- LQG's lack of guaranteed margins (Doyle 1978) and what the tuned knobs cost in performance.")
    w("- Why effort-mode supervision cannot help (the knob barely moves the trade-off), motivating fault-tolerant "
      "supervision with pre-verified controllers and dwell-time switching (Hespanha & Morse 1999).")
    w("- Limitations: linear typical section, simulation only, incompressible aerodynamics, synthetic identification data.\n")
    w("## Reproducibility\n")
    w("`python -m flutter_ai tune|evaluate|robustness|ablations|llm-benchmark|validate`, then `python -m flutter_ai paper`. "
      "Model files are identified by the hashes reported at `/api/health`.")
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_path
