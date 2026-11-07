"""
Evaluation of the on-board physics estimator (margin.py v1 and v2) on development seeds.

    python -m flutter_ai margin-eval --preset uav_flap     # -> outputs/<preset>/margin_eval.json + fig15_margin.png

For every fault and development seed (>= 1000; the reserved seeds 0-19, 100-119 and 200-204 are refused):
  1. the four fixed-mode flights of the oracle (supervisors.oracle_mode: each mode applied 1 s after onset), which say
     which modes survive that fault; and the nominal flight (no supervisor) with its full record;
  2. both estimators replayed on the nominal flight exactly as they run on board (same windows, baselines and code
     paths as ft_simulate with margin_every_s), every `every_s` from the end of the take-off window to the end of the
     flight (or its failure), next to the truth: faults.plant_at + flutter_analysis for the flutter speed, the fault
     schedule for turbulence, noise and loop delay, and the measured flap command statistics of the same window
     (validation of the Lyapunov prediction; a truth-model prediction separates model error from estimation error);
  3. per-mode predictions against the fixed-mode outcomes, at the latest estimate once the fault is fully developed
     (window entirely at full severity) or, if the nominal flight failed earlier, the last estimate before it failed;
     warning time (first estimate that flags the nominal mode before the nominal flight fails) and false alarms;
  4. a small dev-seed comparison of the symptom rules, rules + physics (v1) and rules + physics v2 (tiered and strict)
     on the same flights, with the oracle as reference.
Every number in the output file is computed here.
"""

from __future__ import annotations

import functools
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import __version__
from ..flutter import flutter_analysis
from ..params import WingParams, get_preset
from ..paths import OUTPUTS_DIR
from .benchmark import DURATION, KEEP, MARGIN_EVERY_S, _summary, paired
from .diagnostics import LOOKBACK_S
from .faults import FAULTS, benchmark_fault, delay_at, gust_gain_at, plant_at, sensor_noise_at
from .margin import (DESIGN_TURBULENCE_RMS, PARAMS, _authority_ratio, applied_commands, command_statistics, estimate,
                     estimate_v2, fit_window, fit_window_v2)
from .simulate import MODE_ORDER, SPEED_STEP, _delays, ft_simulate
from .stability import mode_systems
from .supervisors import FTPhysicsRuleSupervisor, FTPhysicsRuleSupervisorV2, FTRuleSupervisor, oracle_mode

SEEDS = tuple(range(1000, 1005))
RESERVED = set(range(0, 20)) | set(range(100, 120)) | set(range(200, 205))
BASELINE = (0.5, 4.5)  # ft_simulate's take-off window
SUPERVISORS = ("rules", "rules+physics", "rules+physics_v2", "rules+physics_v2_strict")
PAIRS = (("rules+physics_v2", "rules+physics"), ("rules+physics_v2", "rules"),
         ("rules+physics_v2_strict", "rules+physics_v2"), ("rules+physics", "rules"))
PREDICTORS = {"v1 stable": ("v1", "stable"), "v2 admissible (robust)": ("v2", "admissible"),
              "v2 admissible (point)": ("v2", "admissible_point")}


def _check_seeds(seeds) -> None:
    bad = sorted(set(seeds) & RESERVED)
    if bad or min(seeds) < 1000:
        raise ValueError(f"margin-eval is a development study: use seeds >= 1000 (got {sorted(seeds)}); "
                         "0-19, 100-119 and 200-204 are reserved for the benchmarks")


@functools.lru_cache(maxsize=256)
def _true_flutter(q: WingParams) -> float | None:
    return flutter_analysis(q).flutter_speed


def _window_stats(u: np.ndarray, dt: float, a: int, k: int, u_max: float) -> dict:
    """Measured flap command statistics over samples a..k-1 (the estimator's window)."""
    w = u[a:k, 0]
    du = np.diff(u[max(a - 1, 0):k, 0]) / dt
    return {"command_rms": float(np.sqrt(np.mean(w**2))), "command_rate_rms": float(np.sqrt(np.mean(du**2))),
            "saturation_fraction": float(np.mean(np.abs(w) >= 0.999 * u_max))}


def _truth_prediction(p: WingParams, fault, t: float, U: float, dt: float) -> dict:
    """The same Lyapunov prediction for the nominal controller, fed the true wing, turbulence, noise and delay."""
    sysm = mode_systems(p, U, dt, plant=plant_at(p, fault, t), delay=delay_at(fault, t))["lqg"]
    stats = command_statistics(p, sysm, dt, DESIGN_TURBULENCE_RMS * gust_gain_at(fault, t), sensor_noise_at(fault, t))
    return {"spectral_radius": float(np.max(np.abs(np.linalg.eigvals(sysm[0])))),
            "command_rms": None if stats is None else float(stats[0][0]),
            "command_rate_rms": None if stats is None else float(stats[1][0]),
            "authority_ratio": _authority_ratio(p, stats)}


def _replay_job(job: tuple) -> dict:
    """Fixed-mode flights + both estimators replayed on the nominal flight (one fault, one seed)."""
    preset, kind, seed, every_s = job
    t0 = time.time()
    p = get_preset(preset)
    fa = flutter_analysis(p)
    Uf, fhz = fa.flutter_speed, fa.flutter_frequency_hz
    if Uf is None or fhz is None:
        raise ValueError(f"No flutter found for '{preset}'")
    fault = benchmark_fault(kind)
    best, runs = oracle_mode(p, Uf, fhz, fault, seed, duration=DURATION)
    r = ft_simulate(p, Uf, fhz, fault, None, duration=DURATION, seed=seed)
    if r.failed != runs["nominal"]["failed"]:
        raise RuntimeError("the nominal flight does not reproduce the oracle's nominal run")
    dt = float(r.t[1] - r.t[0])
    cruise = float(r.airspeed[0])
    U = round(cruise / SPEED_STEP) * SPEED_STEP  # ft_simulate's U_level while flying nominal
    n = len(r.t)
    ua1 = applied_commands(p, r.u, n, dt)  # causal: slices equal the on-board computation at every k
    ua2 = applied_commands(p, r.u, n, dt, _delays(fault, r.t, n))
    i0, i1 = int(BASELINE[0] / dt), int(round(BASELINE[1] / dt))
    base1 = fit_window(p, U, dt, r.y[i0:i1], ua1[i0:i1])
    base2 = fit_window_v2(p, U, dt, r.y[i0:i1], ua2[i0:i1])
    u_max = float(p.u_max[0])
    series = []
    for tk in np.arange(BASELINE[1], float(r.t[-1]) + 1e-9, every_s):
        k = int(round(tk / dt))
        a = max(0, k - int(LOOKBACK_S / dt))
        e1, _ = estimate(p, Uf, float(tk), U, dt, r.y[a:k], ua1[a:k], base1, cruise=cruise)
        e2, _ = estimate_v2(p, Uf, float(tk), U, dt, r.y[a:k], ua2[a:k], base2, delay=delay_at(fault, float(tk)),
                            cruise=cruise)
        q = plant_at(p, fault, float(tk))
        series.append({
            "t": float(tk), "level": fault.level(float(tk)),
            "true": {"f_alpha": q.f_alpha / p.f_alpha, "flap_span_fraction": q.flap_span_fraction / p.flap_span_fraction,
                     "flutter_speed": _true_flutter(q), "turbulence_rms": DESIGN_TURBULENCE_RMS * gust_gain_at(fault, tk),
                     "noise_std": sensor_noise_at(fault, float(tk)), "delay_samples": delay_at(fault, float(tk))},
            "v1": {"params": e1.params, "flutter_speed": e1.flutter_speed, "flutter_speed_std": e1.flutter_speed_std,
                   "stable": {m: rho < 1.0 for m, rho in e1.mode_spectral_radius.items()}},
            "v2": {k_: v for k_, v in e2.to_dict().items() if k_ not in ("time", "airspeed", "mode_spectral_radius")},
            "measured": _window_stats(r.u, dt, a, k, u_max),
            "truth_model": _truth_prediction(p, fault, float(tk), U, dt),
        })
    return {"fault": kind, "seed": seed, "oracle_mode": best, "fixed_modes": {m: {k_: runs[m][k_] for k_ in KEEP}
                                                                              for m in MODE_ORDER},
            "nominal_failure_time_s": r.failure_time, "series": series, "wall_s": round(time.time() - t0, 1)}


def _make_supervisor(kind: str):
    return {"rules": FTRuleSupervisor, "rules+physics": FTPhysicsRuleSupervisor,
            "rules+physics_v2": FTPhysicsRuleSupervisorV2,
            "rules+physics_v2_strict": lambda: FTPhysicsRuleSupervisorV2(strict=True)}[kind]()


def _compare_job(job: tuple) -> dict:
    preset, kind, seed, sup = job
    p = get_preset(preset)
    fa = flutter_analysis(p)
    if fa.flutter_speed is None or fa.flutter_frequency_hz is None:
        raise ValueError(f"No flutter found for '{preset}'")
    r = ft_simulate(p, fa.flutter_speed, fa.flutter_frequency_hz, benchmark_fault(kind), _make_supervisor(sup),
                    duration=DURATION, seed=seed, margin_every_s=None if sup == "rules" else MARGIN_EVERY_S,
                    margin_version=2 if sup.startswith("rules+physics_v2") else 1)
    return {"fault": kind, "seed": seed, "supervisor": sup, "metrics": {k: r.metrics[k] for k in KEEP},
            "switches": [(float(a), m, float(b)) for a, m, b in r.switch_log]}


def _eval_record(rec: dict) -> dict | None:
    """The estimate used for the per-mode predictions: the latest one whose window lies entirely at full severity
    (onset + ramp + look-back), or the last one before the nominal flight failed if it failed earlier."""
    fault = benchmark_fault(rec["fault"])
    t_full = fault.onset + fault.ramp + LOOKBACK_S
    ok = [s for s in rec["series"] if s["t"] <= t_full + 1e-9]
    return ok[-1] if ok else None


def _at_eval_row(r: dict) -> dict:
    """What each predictor said about every mode at the evaluation estimate, next to the fixed-mode outcomes."""
    row: dict = {"seed": r["seed"], "survived": {m: not r["fixed_modes"][m]["failed"] for m in MODE_ORDER}}
    s = _eval_record(r)
    if s is None:
        return {**row, "t": None}
    v2 = s["v2"]["modes"]
    return {**row, "t": s["t"], "level": s["level"], "v1_stable": s["v1"]["stable"],
            "v2_admissible": {m: v2[m]["admissible"] for m in MODE_ORDER},
            "v2_admissible_point": {m: v2[m]["admissible_point"] for m in MODE_ORDER},
            "v2_authority_ratio": {m: v2[m]["authority_ratio"] for m in MODE_ORDER}}


def _confusion(pairs: list[tuple[bool, bool]]) -> dict:
    """(predicted admissible/stable, survived) -> counts; 'missed' = predicted fine but the fixed mode failed."""
    c = {"ok_survived": 0, "ok_failed_missed": 0, "flagged_survived_conservative": 0, "flagged_failed": 0}
    for pred, alive in pairs:
        c[("ok_survived" if alive else "ok_failed_missed") if pred else
          ("flagged_survived_conservative" if alive else "flagged_failed")] += 1
    n = len(pairs)
    return {**c, "n": n, "accuracy": (c["ok_survived"] + c["flagged_failed"]) / n if n else None}


def _flags_nominal(s: dict, predictor: str) -> bool:
    est, key = PREDICTORS[predictor]
    return not (s["v1"]["stable"]["nominal"] if est == "v1" else s["v2"]["modes"]["nominal"][key])


def _stats(x: list[float]) -> dict:
    a = np.array([v for v in x if v is not None and math.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}


def _analyse(replays: list[dict]) -> dict:
    faults = [k for k in FAULTS if any(r["fault"] == k for r in replays)]
    by_fault = {k: [r for r in replays if r["fault"] == k] for k in faults}
    acc: dict = {"flutter_speed": {}, "turbulence": {}, "noise": {}, "delay": {}, "command_rms": {}}
    for k, recs in by_fault.items():
        ss = [s for r in recs for s in r["series"] if s["t"] > benchmark_fault(k).onset]
        err: dict[str, list[float]] = {"v1": [], "v2": []}
        covered, cov1 = [], []
        for s in ss:
            tru = s["true"]["flutter_speed"]
            for v in ("v1", "v2"):
                if tru is not None and s[v]["flutter_speed"] is not None:
                    err[v].append(s[v]["flutter_speed"] / tru - 1.0)
            if tru is not None and s["v2"]["flutter_speed_worst"] is not None:
                covered.append(s["v2"]["flutter_speed_worst"] <= tru)
            if tru is not None and s["v2"]["flutter_speed"] is not None and s["v2"]["flutter_speed_std"] is not None:
                cov1.append(abs(s["v2"]["flutter_speed"] - tru) <= s["v2"]["flutter_speed_std"])
        acc["flutter_speed"][k] = {
            "relative_error_v1": _stats(err["v1"]), "relative_error_v2": _stats(err["v2"]),
            "abs_relative_error_v1": _stats([abs(e) for e in err["v1"]]),
            "abs_relative_error_v2": _stats([abs(e) for e in err["v2"]]),
            "worst_case_below_truth_share": float(np.mean(covered)) if covered else None,
            "within_1sigma_share": float(np.mean(cov1)) if cov1 else None, "n_estimates": len(ss)}
        acc["turbulence"][k] = {"ratio_estimate_to_true": _stats([s["v2"]["turbulence_rms"] / s["true"]["turbulence_rms"]
                                                                  for s in ss])}
        acc["noise"][k] = {"estimate_minus_true": _stats([s["v2"]["noise_std"] - s["true"]["noise_std"] for s in ss])}
        acc["delay"][k] = {"correct_share": float(np.mean([s["v2"]["delay_samples"] == s["true"]["delay_samples"]
                                                          for s in ss])) if ss else None}
        # command statistics of the nominal mode: predicted (on board, and with the true inputs) / measured, same window
        allw = [s for r in recs for s in r["series"]]
        unsat = [s for s in allw if s["measured"]["saturation_fraction"] == 0.0]
        rows = {}
        for label, subset in (("all_windows", allw), ("windows_without_saturation", unsat)):
            row: dict = {"n_windows": len(subset)}
            for q in ("command_rms", "command_rate_rms"):
                for src, get in (("estimator", lambda s, q=q: s["v2"]["modes"]["nominal"][q]),
                                 ("truth_model", lambda s, q=q: s["truth_model"][q])):
                    row[f"{src}_{q}_predicted_over_measured"] = _stats(
                        [get(s) / s["measured"][q] for s in subset if get(s) is not None])
            rows[label] = row
        acc["command_rms"][k] = rows

    # per-mode predictions vs the fixed-mode outcomes
    preds: dict = {name: {"per_fault": {}, "pooled_faulty": None, "pooled_all": None} for name in PREDICTORS}
    at_eval: dict = {}
    for name, (est, key) in PREDICTORS.items():
        pooled_f, pooled_a = [], []
        for k, recs in by_fault.items():
            pairs = []
            for r in recs:
                s = _eval_record(r)
                if s is None:
                    continue
                for m in MODE_ORDER:
                    pred = s["v1"]["stable"][m] if est == "v1" else s["v2"]["modes"][m][key]
                    pairs.append((bool(pred), not r["fixed_modes"][m]["failed"]))
            preds[name]["per_fault"][k] = _confusion(pairs)
            pooled_a += pairs
            pooled_f += pairs if k != "none" else []
        preds[name]["pooled_faulty"] = _confusion(pooled_f)
        preds[name]["pooled_all"] = _confusion(pooled_a)
    for k, recs in by_fault.items():
        at_eval[k] = [_at_eval_row(r) for r in recs]

    # warning time before the nominal flight fails, and false alarms where it survives
    warn: dict = {}
    for k, recs in by_fault.items():
        onset = benchmark_fault(k).onset
        wrow: dict = {}
        for name in PREDICTORS:
            lead, alarms = [], []
            for r in recs:
                after = [s for s in r["series"] if s["t"] > onset]
                if r["nominal_failure_time_s"] is not None:
                    first = next((s["t"] for s in after if _flags_nominal(s, name)), None)
                    lead.append(None if first is None else r["nominal_failure_time_s"] - first)
                else:
                    alarms += [_flags_nominal(s, name) for s in r["series"]]
            wrow[name] = {"nominal_failed_flights": len(lead), "warned": sum(v is not None for v in lead),
                          "warning_time_s": _stats([v for v in lead if v is not None]),
                          "nominal_survived_flights": sum(r["nominal_failure_time_s"] is None for r in recs),
                          "flagged_share_of_estimates_when_nominal_survived":
                              float(np.mean(alarms)) if alarms else None}
        warn[k] = wrow
    return {"accuracy": acc, "mode_predictions": preds, "at_evaluation": at_eval, "nominal_warning": warn}


def _comparison(results: list[dict], replays: list[dict], seeds) -> dict:
    faults = [k for k in FAULTS if any(r["fault"] == k for r in results)]
    out: dict = {"faults": {}, "pooled_faulty": {}}
    pooled: dict[str, list[dict]] = {}
    for k in faults:
        per: dict[str, list[dict]] = {}
        oracle_modes = []
        for s in seeds:
            rep = next(r for r in replays if r["fault"] == k and r["seed"] == s)
            oracle_modes.append(rep["oracle_mode"])
            best = rep["oracle_mode"] if rep["oracle_mode"] in MODE_ORDER else "safe"
            per.setdefault("oracle", []).append(rep["fixed_modes"][best])
            for sup in SUPERVISORS:
                res = next((x for x in results if x["fault"] == k and x["seed"] == s and x["supervisor"] == sup), None)
                if res is not None:
                    per.setdefault(sup, []).append(res["metrics"])
        out["faults"][k] = {"oracle_modes": oracle_modes, "supervisors": {sup: _summary(v, oracle_modes)
                                                                          for sup, v in per.items()},
                            "per_seed": {sup: [{"failed": m["failed"], "slowed_s": m["slowed_s"],
                                                "final_mode": m["final_mode"], "reaction_s": m["reaction_s"]}
                                               for m in v] for sup, v in per.items()},
                            "paired": {f"{a} vs {b}": paired(per[a], per[b]) for a, b in PAIRS if a in per and b in per}}
        if k != "none":
            for sup, v in per.items():
                pooled.setdefault(sup, []).extend(v)
    out["pooled_faulty"] = {"supervisors": {sup: _summary(v, [""] * len(v)) for sup, v in pooled.items()},
                            "paired": {f"{a} vs {b}": paired(pooled[a], pooled[b]) for a, b in PAIRS
                                       if a in pooled and b in pooled}}
    out["false_alarm_slowed_s"] = {sup: out["faults"]["none"]["supervisors"][sup]["slowed_s_mean"]
                                   for sup in out["faults"].get("none", {}).get("supervisors", {})}
    return out


def run_margin_eval(p: WingParams, seeds=SEEDS, compare_seeds=SEEDS, every_s: float = 1.0, workers: int = 4,
                    faults: tuple[str, ...] | None = None, out_dir: Path | None = None, make_figure: bool = True,
                    log=print) -> dict:
    seeds, compare_seeds = tuple(seeds), tuple(compare_seeds)
    _check_seeds(seeds + compare_seeds)
    if not set(compare_seeds) <= set(seeds):
        raise ValueError("compare_seeds must be a subset of seeds (the oracle comes from the replay flights)")
    kinds = tuple(faults or FAULTS)
    t0 = time.time()
    replay_jobs = [(p.name, k, s, every_s) for k in kinds for s in seeds]
    compare_jobs = [(p.name, k, s, sup) for k in kinds for s in compare_seeds for sup in SUPERVISORS]
    with ProcessPoolExecutor(max_workers=max(1, min(4, workers))) as ex:
        fut_r = [ex.submit(_replay_job, j) for j in replay_jobs]
        fut_c = [ex.submit(_compare_job, j) for j in compare_jobs]
        replays = []
        for f in fut_r:
            replays.append(f.result())
            log(f"  replay {replays[-1]['fault']:18s} seed {replays[-1]['seed']}: oracle {replays[-1]['oracle_mode']:9s}"
                f" ({replays[-1]['wall_s']:.0f} s)")
        compares = [f.result() for f in fut_c]
    Uf = flutter_analysis(p).flutter_speed
    res: dict = {
        "metadata": {"version": __version__, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "preset": p.name, "seeds": list(seeds), "compare_seeds": list(compare_seeds),
                     "faults": list(kinds), "duration_s": DURATION, "estimate_every_s": every_s,
                     "lookback_s": LOOKBACK_S, "params": list(PARAMS), "design_turbulence_rms": DESIGN_TURBULENCE_RMS,
                     "flutter_speed_nominal": Uf, "wall_time_s": round(time.time() - t0, 1),
                     "evaluation_estimate": "latest estimate whose window is entirely at full severity (onset + ramp + "
                                            "look-back), or the last one before the nominal flight failed",
                     "note": "Development seeds only. Every value in this file is produced by "
                             "flutter_ai.fault_tolerance.margin_eval."},
        **_analyse(replays),
        "comparison": _comparison(compares, replays, compare_seeds),
        "replays": replays,
    }
    out_dir = Path(out_dir or OUTPUTS_DIR / p.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "margin_eval.json"
    path.write_text(json.dumps(res, indent=1))
    log(f"Saved {path} ({res['metadata']['wall_time_s']:.0f} s)")
    for name, d in res["mode_predictions"].items():
        c = d["pooled_faulty"]
        log(f"  {name:24s}: accuracy {c['accuracy']:.2f} over {c['n']} (fault, seed, mode); predicted fine but failed "
            f"{c['ok_failed_missed']}, flagged but survived {c['flagged_survived_conservative']}")
    for sup, s in res["comparison"]["pooled_faulty"]["supervisors"].items():
        log(f"  {sup:24s}: survived {s['survival']:.0%} of faulty flights, slowed {s['slowed_s_mean'] or 0:.1f} s")
    if make_figure:
        log(f"Saved {margin_figure(res, out_dir / 'fig15_margin.png')}")
    return res


def margin_figure(res: dict, path: Path) -> Path:
    """Estimated vs true flutter speed over time per fault; per-mode prediction accuracy; command RMS validation."""
    from .. import plots

    plt = plots.plt
    faults = res["metadata"]["faults"]
    c_true, c_v1, c_v2 = plots.INK, plots.MUTED, "#2a78d6"
    ncol = 3
    nrow = math.ceil((len(faults) + 2) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.7 * nrow), squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax, k in zip(flat, faults):
        recs = [r for r in res["replays"] if r["fault"] == k]
        longest = max(recs, key=lambda r: len(r["series"]))["series"]  # the truth is the same schedule for every seed
        for i, r in enumerate(recs):
            tt = [s["t"] for s in r["series"]]
            ax.plot(tt, [np.nan if s["v1"]["flutter_speed"] is None else s["v1"]["flutter_speed"] for s in r["series"]],
                    color=c_v1, lw=1.0, ls=":", zorder=3, label="v1 estimate" if i == 0 else None)
            ax.plot(tt, [np.nan if s["v2"]["flutter_speed"] is None else s["v2"]["flutter_speed"] for s in r["series"]],
                    color=c_v2, lw=1.2, label="v2 estimate" if i == 0 else None)
            ax.plot(tt, [np.nan if s["v2"]["flutter_speed_worst"] is None else s["v2"]["flutter_speed_worst"]
                         for s in r["series"]], color=c_v2, lw=0.9, ls="--", alpha=0.7,
                    label="v2 worst case (95 % region)" if i == 0 else None)
            if r["nominal_failure_time_s"] is not None:
                ax.axvline(r["nominal_failure_time_s"], color=plots.CRITICAL, lw=0.7, alpha=0.5,
                           label="nominal flight fails" if i == 0 else None)
        ax.plot([s["t"] for s in longest], [np.nan if s["true"]["flutter_speed"] is None else s["true"]["flutter_speed"]
                                            for s in longest], color=c_true, lw=1.8, label="true")
        ax.axhline(longest[0]["v2"]["modes"]["nominal"]["target_airspeed"], color=plots.AXIS, lw=0.8)  # cruise
        ax.set_title(k + (" (held out)" if FAULTS[k]["held_out"] else ""), fontsize=9)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("flutter speed (m/s)")
    flat[0].legend(fontsize=6.5, loc="lower left")
    # per-mode predictions: share correct, and 'predicted fine but failed' (the dangerous error)
    ax = flat[len(faults)]
    names = list(res["mode_predictions"])
    x = np.arange(len(names))
    accs = [100 * (res["mode_predictions"][nm]["pooled_faulty"]["accuracy"] or 0.0) for nm in names]
    missed = [res["mode_predictions"][nm]["pooled_faulty"]["ok_failed_missed"] for nm in names]
    n = res["mode_predictions"][names[0]]["pooled_faulty"]["n"]
    ax.bar(x, accs, 0.6, color=[c_v1, c_v2, "#7fb0ea"])
    for xi, a, mi in zip(x, accs, missed):
        ax.text(xi, a + 2, f"{a:.0f} %\n{mi} missed", ha="center", fontsize=7, color=plots.INK2)
    ax.set_xticks(x, [nm.replace(" (", "\n(") for nm in names], fontsize=7)
    ax.set_ylim(0, 115)
    ax.set_ylabel("correct mode predictions (%)")
    ax.set_title(f"Mode survival predicted (faulty, n = {n})", fontsize=9)
    # command RMS validation: predicted vs measured, windows without saturation
    ax = flat[len(faults) + 1]
    for i, k in enumerate(("none", "turbulence_rise", "flap_effectiveness", "sensor_noise")):
        pts = [(s["measured"]["command_rate_rms"], s["v2"]["modes"]["nominal"]["command_rate_rms"])
               for r in res["replays"] if r["fault"] == k for s in r["series"]
               if s["measured"]["saturation_fraction"] == 0.0 and s["v2"]["modes"]["nominal"]["command_rate_rms"]]
        if pts:
            m_, p_ = np.degrees(np.array(pts)).T
            ax.scatter(m_, p_, s=10, color=("#0b0b0b", "#eb6834", "#4a3aa7", "#1baf7a")[i], label=k, alpha=0.7,
                       edgecolors="white", linewidths=0.4)
    lim = ax.get_xlim()[1], ax.get_ylim()[1]
    top = max(lim)
    ax.plot([0, top], [0, top], color=plots.AXIS, lw=0.8)
    ax.set_xlabel("measured flap rate RMS (deg/s)")
    ax.set_ylabel("predicted on board (deg/s)")
    ax.set_title("Nominal mode, windows without saturation", fontsize=9)
    ax.legend(fontsize=6.5)
    for ax in flat[len(faults) + 2:]:
        ax.set_visible(False)
    fig.suptitle("On-board physics estimator (dev seeds): flutter speed over time, mode predictions, flap command "
                 "statistics", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
