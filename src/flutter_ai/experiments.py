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
from .controllers import LQRController, NeuralController, OpenLoop, policy_path
from .flutter import flutter_analysis, pk_flutter_speed
from .params import WingParams
from .paths import OUTPUTS_DIR
from .scenarios import SCENARIOS, scenarios_for
from .simulation import Scenario, SimResult, simulate
from .supervisor import LLMSupervisor, RuleSupervisor

CONTROLLERS = ("open_loop", "lqr", "neural")
SUPERVISORS = ("none", "rules", "llm")

# The comparison table in the paper
STANDARD_CASES = [
    ("open_loop", "none"),
    ("lqr", "none"),
    ("neural", "none"),
    ("lqr", "rules"),
    ("neural", "rules"),
]


def make_controller(kind: str):
    if kind == "open_loop":
        return OpenLoop()
    if kind == "lqr":
        return LQRController()
    if kind == "neural":
        return NeuralController()
    raise ValueError(f"Unknown controller '{kind}'. Use one of {CONTROLLERS}")


def make_supervisor(kind: str | None):
    if kind in (None, "none"):
        return None
    if kind == "rules":
        return RuleSupervisor()
    if kind == "llm":
        return LLMSupervisor()
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
    """
    base = base or SCENARIOS["discrete_gust"]["scenario"]
    out: dict = {"delays": list(delays), "dt": base.dt}
    variants = {c: (lambda c=c: make_controller(c)) for c in controllers}
    if "lqr" in controllers:
        variants["lqr_nominal"] = lambda: LQRController(design_delay=1)
    for c, factory in variants.items():
        out[c] = []
        for d in delays:
            m = simulate(p, factory(), replace(base, input_delay_steps=int(d)), Uf).metrics
            out[c].append({"delay_steps": int(d), "exceeded_limits": m["exceeded_limits"],
                           "failure_time_s": m["failure_time_s"], "rms_h_mm": m["rms_h_mm"],
                           "control_rate_rms": m["control_rate_rms"]})
    return out


def _aggregate(metric_list: list[dict]) -> dict:
    """Mean and std across seeds for numeric metrics; failure count for limits."""
    out: dict = {"n": len(metric_list), "failures": sum(m["exceeded_limits"] for m in metric_list)}
    for key in metric_list[0]:
        vals = [m[key] for m in metric_list if isinstance(m[key], (int, float)) and not isinstance(m[key], bool)]
        if len(vals) == len(metric_list):
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def run_benchmark(p: WingParams, seeds=(0, 1, 2, 3, 4), out_dir: Path | None = None, make_figures: bool = True,
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
        stochastic = base.gust.kind == "turbulence" or base.sensor_noise > 0  # only these need several seeds
        run_seeds = seeds if stochastic else seeds[:1]
        table = {}
        for ctrl, sup in cases:
            per_seed = []
            for s in run_seeds:
                gust = replace(base.gust, seed=base.gust.seed + s)
                res = run_case(p, Uf, replace(base, gust=gust, seed=s), ctrl, sup)
                per_seed.append(res.metrics)
                if s == run_seeds[0]:
                    examples.setdefault(name, []).append(res)
            key = ctrl if sup == "none" else f"{ctrl}+{sup}"
            table[key] = agg = _aggregate(per_seed)
            log(f"  {key:14s} failures {agg['failures']}/{agg['n']}   "
                f"RMS h {agg['rms_h_mm']['mean']:6.2f} +/- {agg['rms_h_mm']['std']:.2f} mm   "
                f"peak h {agg['peak_h_mm']['mean']:6.2f} mm   effort {agg['control_energy']['mean']:.3f}   "
                f"rate {agg['control_rate_rms']['mean']:.3f}")
        results["scenarios"][name] = {"description": spec["description"], "scenario": base.to_dict(), "cases": table}

    log("\nEffort/response trade-off sweeps (turbulence_patch)...")
    tb = scenarios["turbulence_patch"]["scenario"]
    feedback = [c for c in ("lqr", "neural") if c in controllers]
    results["tradeoff"] = {c: tradeoff_sweep(p, Uf, tb, c) for c in feedback}
    results["tradeoff"]["supervised"] = {f"{c}+rules": run_case(p, Uf, tb, c, "rules").metrics for c in feedback}
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
