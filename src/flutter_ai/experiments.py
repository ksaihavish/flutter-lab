"""
Experiment runner: every number and figure in the paper should come from here,
so results are reproducible from a single command:

    python -m flutter_ai evaluate --preset uav_flap     # -> outputs/uav_flap/results.json + figures
"""

from __future__ import annotations

import json
import platform
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .controllers import LQGController, LQRController, NeuralController, OpenLoop, policy_path
from .flutter import flutter_analysis, pk_flutter_speed
from .params import WingParams
from .paths import OUTPUTS_DIR
from .scenarios import SCENARIOS, scenarios_for
from .simulation import Scenario, SimResult, simulate
from .supervisor import RuleSupervisor, make_llm_supervisor

CONTROLLERS = ("open_loop", "lqr", "lqg", "neural")
SUPERVISORS = ("none", "rules", "llm")

# The comparison table in the paper
STANDARD_CASES = [
    ("open_loop", "none"),
    ("lqr", "none"),
    ("lqg", "none"),
    ("neural", "none"),
    ("lqr", "rules"),
    ("neural", "rules"),
]


def make_controller(kind: str):
    if kind == "open_loop":
        return OpenLoop()
    if kind == "lqr":
        return LQRController()
    if kind == "lqg":
        return LQGController()
    if kind == "neural":
        return NeuralController()
    raise ValueError(f"Unknown controller '{kind}'. Use one of {CONTROLLERS}")


def make_supervisor(kind: str | None):
    if kind in (None, "none"):
        return None
    if kind == "rules":
        return RuleSupervisor()
    if kind == "llm":
        return make_llm_supervisor()
    raise ValueError(f"Unknown supervisor '{kind}'. Use one of {SUPERVISORS}")


def neural_available(p: WingParams) -> bool:
    return policy_path(p.name).exists()


def run_case(p: WingParams, Uf: float, scenario: Scenario, controller: str, supervisor: str = "none",
             aggressiveness: float | None = None) -> SimResult:
    if controller == "open_loop" and supervisor not in (None, "none"):
        raise ValueError("A supervisor needs a feedback controller (lqr or neural).")
    return simulate(p, make_controller(controller), scenario, Uf, make_supervisor(supervisor), aggressiveness)


def tradeoff_sweep(p: WingParams, Uf: float, scenario: Scenario, controller: str,
                   levels=np.linspace(0.0, 1.0, 11)) -> list[dict]:
    """Effort-vs-response curve for fixed aggressiveness (the frontier a supervisor must beat)."""
    return [{"aggressiveness": float(a), **run_case(p, Uf, scenario, controller, "none", float(a)).metrics}
            for a in levels]


def velocity_sweep(p: WingParams, Uf: float, controllers, ratios=np.round(np.arange(0.8, 1.51, 0.05), 3),
                   base: Scenario | None = None) -> dict:
    base = base or SCENARIOS["discrete_gust"]["scenario"]
    out: dict = {"ratios": [float(r) for r in ratios]}
    for c in controllers:
        out[c] = []
        for r in ratios:
            m = run_case(p, Uf, replace(base, velocity_ratio=float(r)), c).metrics
            out[c].append({"ratio": float(r), "exceeded_limits": m["exceeded_limits"],
                           "peak_h_mm": m["peak_h_mm"], "rms_h_mm": m["rms_h_mm"]})
    return out


def delay_sweep(p: WingParams, Uf: float, controllers, delays=(0, 1, 2, 3, 4, 6),
                base: Scenario | None = None) -> dict:
    """
    Robustness to computation/actuation delay (1 step = dt = 5 ms).
      neural        one fixed policy (trained on 0-2 steps)
      lqr           re-designed for the true delay (it is told the delay - optimistic)
      lqr_nominal   designed for the nominal 1 step and kept fixed (the fair comparison)
      lqg / lqg_nominal   the same two variants of the LQG baseline
    """
    base = base or SCENARIOS["discrete_gust"]["scenario"]
    out: dict = {"delays": list(delays), "dt": base.dt}
    variants = {c: (lambda c=c: make_controller(c)) for c in controllers}
    if "lqr" in controllers:
        variants["lqr_nominal"] = lambda: LQRController(design_delay=1)
    if "lqg" in controllers:
        variants["lqg_nominal"] = lambda: LQGController(design_delay=1)
    for c, factory in variants.items():
        out[c] = []
        for d in delays:
            m = simulate(p, factory(), replace(base, input_delay_steps=int(d)), Uf).metrics
            out[c].append({"delay_steps": int(d), "exceeded_limits": m["exceeded_limits"],
                           "failure_time_s": m["failure_time_s"], "rms_h_mm": m["rms_h_mm"],
                           "control_rate_rms": m["control_rate_rms"]})
    return out


def draw_scenario(base: Scenario, s: int) -> Scenario:
    """
    Seed s of a benchmark scenario. Turbulence: a new realisation of the same statistics. Deterministic gusts:
    seed 0 is the nominal case; other seeds draw the gust amplitude x[0.75, 1.25], length [0.2, 0.5] s and
    start [2.5, 3.5] s, and scale the initial plunge / pitch by [0.5, 1.5] with random signs, so a statistic
    covers a family of gusts rather than one hand-picked gust.
    """
    if base.gust.kind == "turbulence" or base.sensor_noise > 0:
        return replace(base, gust=replace(base.gust, seed=base.gust.seed + s), seed=s)
    if s == 0 or base.gust.kind == "none":
        return base
    rng = np.random.default_rng(1000 + s)
    gust = replace(base.gust, amplitude=base.gust.amplitude * rng.uniform(0.75, 1.25),
                   duration=rng.uniform(0.2, 0.5), start=rng.uniform(2.5, 3.5))
    sign = rng.choice([-1.0, 1.0], size=2)
    return replace(base, gust=gust, seed=s, h0=base.h0 * rng.uniform(0.5, 1.5) * sign[0],
                   alpha0_deg=base.alpha0_deg * rng.uniform(0.5, 1.5) * sign[1])


# Paired comparisons reported for every scenario: (candidate, reference)
COMPARISONS = [("neural", "lqr"), ("neural", "lqg"), ("lqg", "lqr"), ("neural+rules", "neural"), ("lqr+rules", "lqr")]
STAT_METRICS = ("rms_h_mm", "cost_J", "control_energy")


def paired_statistics(a: list[dict], b: list[dict], n_boot: int = 10000) -> dict:
    """
    Candidate a vs reference b over the same seeds. Per metric: mean difference (a - b) over the seeds where
    both survived, a 95 % bootstrap CI of that mean, and the two-sided Wilcoxon signed-rank p-value.
    Failures are counted separately (a failed run has no comparable RMS or J).
    """
    from scipy.stats import wilcoxon

    both = [i for i in range(len(a)) if not a[i]["exceeded_limits"] and not b[i]["exceeded_limits"]]
    out: dict = {"n_seeds": len(a), "n_both_survived": len(both),
                 "failures": {"candidate": sum(m["exceeded_limits"] for m in a),
                              "reference": sum(m["exceeded_limits"] for m in b)}}
    rng = np.random.default_rng(0)
    for key in STAT_METRICS:
        d = np.array([a[i][key] - b[i][key] for i in both if a[i][key] is not None and b[i][key] is not None])
        if len(d) < 2:
            out[key] = {"n": int(len(d)), "note": "fewer than 2 paired runs; no statistics"}
            continue
        boot = rng.choice(d, size=(n_boot, len(d))).mean(axis=1)
        p = 1.0 if np.allclose(d, 0.0) else float(wilcoxon(d).pvalue)
        out[key] = {"n": int(len(d)), "mean_diff": float(d.mean()), "rel_diff": float(d.mean() / np.mean(
            [b[i][key] for i in both])), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wilcoxon_p": p}
    return out


def _aggregate(metric_list: list[dict]) -> dict:
    """Mean and std across seeds for numeric metrics; failure count for limits."""
    out: dict = {"n": len(metric_list), "failures": sum(m["exceeded_limits"] for m in metric_list)}
    for key in metric_list[0]:
        vals = [m[key] for m in metric_list if isinstance(m[key], (int, float)) and not isinstance(m[key], bool)]
        if len(vals) == len(metric_list):
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def run_benchmark(p: WingParams, seeds=tuple(range(20)), out_dir: Path | None = None, make_figures: bool = True,
                  log=print) -> dict:
    out_dir = Path(out_dir or OUTPUTS_DIR / p.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    fa = flutter_analysis(p)
    Uf = fa.flutter_speed
    Uf_pk = pk_flutter_speed(p)
    log(f"Preset '{p.name}' ({p.n_dof}-DOF, actuation {p.actuation})")
    log(f"Flutter speed: {Uf:.3f} m/s (state-space)   {Uf_pk:.3f} m/s (p-k, exact Theodorsen)   "
        f"difference {100 * (Uf - Uf_pk) / Uf_pk:+.2f}%")

    has_neural = neural_available(p)
    cases = [c for c in STANDARD_CASES if has_neural or c[0] != "neural"]
    controllers = [c for c in CONTROLLERS if has_neural or c != "neural"]
    if not has_neural:
        log(f"NOTE: no neural policy for '{p.name}' - neural cases skipped "
            f"(train with: python -m flutter_ai train --preset {p.name})")

    results: dict = {
        "metadata": {
            "version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "seeds": list(seeds),
            "neural_policy": None,
            "note": "Every value in this file is produced by flutter_ai.experiments.run_benchmark.",
        },
        "params": p.to_dict(),
        "flutter": {"state_space": Uf, "pk_exact_theodorsen": Uf_pk,
                    "frequency_hz": fa.flutter_frequency_hz, "divergence_speed": fa.divergence_speed},
        "scenarios": {},
    }
    if has_neural:
        from .controllers import NeuralPolicy

        meta = NeuralPolicy.load(policy_path(p.name)).meta
        results["metadata"]["neural_policy"] = {k: meta[k] for k in ("trained_at", "validation", "lqr_validation",
                                                                    "delay_range", "velocity_ratio_range")}

    scenarios = scenarios_for(p.name)
    examples: dict[str, list[SimResult]] = {}
    for name, spec in scenarios.items():
        base: Scenario = spec["scenario"]
        log(f"\nScenario '{name}': {spec['description']}")
        table, raw = {}, {}
        for ctrl, sup in cases:
            per_seed = []
            for s in seeds:
                res = run_case(p, Uf, draw_scenario(base, s), ctrl, sup)
                per_seed.append(res.metrics)
                if s == seeds[0]:
                    examples.setdefault(name, []).append(res)
            key = ctrl if sup == "none" else f"{ctrl}+{sup}"
            raw[key] = per_seed
            table[key] = agg = _aggregate(per_seed)
            log(f"  {key:14s} failures {agg['failures']}/{agg['n']}   "
                f"RMS h {agg['rms_h_mm']['mean']:6.2f} +/- {agg['rms_h_mm']['std']:.2f} mm   "
                f"peak h {agg['peak_h_mm']['mean']:6.2f} mm   effort {agg['control_energy']['mean']:.3f}   "
                f"rate {agg['control_rate_rms']['mean']:.3f}")
        stats = {f"{a} vs {b}": paired_statistics(raw[a], raw[b]) for a, b in COMPARISONS if a in raw and b in raw}
        for label, st in stats.items():
            r = st["rms_h_mm"]
            if "mean_diff" in r:
                log(f"  {label:22s} RMS h diff {r['mean_diff']:+6.2f} mm ({100 * r['rel_diff']:+5.1f} %), "
                    f"95% CI [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}], Wilcoxon p = {r['wilcoxon_p']:.2g}, "
                    f"n = {r['n']}, failures {st['failures']['candidate']}/{st['failures']['reference']}")
        results["scenarios"][name] = {"description": spec["description"], "scenario": base.to_dict(), "cases": table,
                                      "statistics": stats}

    log("\nEffort/response trade-off sweeps (turbulence_patch)...")
    tb = scenarios["turbulence_patch"]["scenario"]
    feedback = [c for c in ("lqr", "lqg", "neural") if c in controllers]
    results["tradeoff"] = {c: tradeoff_sweep(p, Uf, tb, c) for c in feedback}
    results["tradeoff"]["supervised"] = {f"{c}+rules": run_case(p, Uf, tb, c, "rules").metrics
                                         for c in feedback if c != "lqg"}
    log("Velocity sweep...")
    results["velocity_sweep"] = velocity_sweep(p, Uf, controllers, base=scenarios["discrete_gust"]["scenario"])
    log("Delay-robustness sweep...")
    results["delay_sweep"] = delay_sweep(p, Uf, feedback, base=scenarios["discrete_gust"]["scenario"])
    for c in [k for k in results["delay_sweep"] if k not in ("delays", "dt")]:
        ok = [r["delay_steps"] for r in results["delay_sweep"][c] if not r["exceeded_limits"]]
        log(f"  {c:12s} survives delays (steps of 5 ms): {ok}")

    path = out_dir / "results.json"
    path.write_text(json.dumps(results, indent=2))
    log(f"\nSaved {path}")

    if make_figures:
        from . import plots

        for f in plots.make_all(p, fa, Uf_pk, examples, results, out_dir):
            log(f"Saved {f}")
    return results
