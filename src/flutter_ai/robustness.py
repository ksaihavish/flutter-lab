"""
Robustness study (Phase 4.3): every controller is designed for the nominal wing and flies a different one.

    python -m flutter_ai robustness --preset uav_flap     # -> outputs/<preset>/robustness.json + figure

Three sweeps, each reported per controller as the failure rate and, over the surviving runs, RMS plunge and
cost J. Every controller flies the same draws.
  parameters    mass, plunge / pitch stiffness, structural damping and (flap wings) servo bandwidth, each drawn
                uniformly within +/-level of nominal; the true flutter speed of every draw is recorded
  sensor_noise  measurement-noise std as a fraction of the reference scales (turbulence scenario)
  rate_limit    (flap wings) actuator slew-rate limits the designs do not know about (turbulence scenario)
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .experiments import make_controller, neural_available
from .flutter import flutter_analysis
from .params import WingParams
from .paths import OUTPUTS_DIR
from .scenarios import scenarios_for
from .simulation import Scenario, simulate

LEVELS = (0.05, 0.10, 0.20)
NOISE_LEVELS = (0.0, 0.02, 0.05, 0.10)
RATE_LIMITS_DEG_S = (None, 500.0, 250.0, 125.0)


def perturbed_wing(p: WingParams, level: float, rng: np.random.Generator) -> WingParams:
    """
    A wing whose mass, stiffnesses, damping ratios and servo bandwidth each differ from nominal by a uniform
    factor in [1 - level, 1 + level]. Mass scales every inertia (S_alpha, I_alpha, I_beta), so the frequencies
    follow f = f0 * sqrt(stiffness factor / mass factor).
    """
    def factor() -> float:
        return 1.0 + float(rng.uniform(-level, level))

    m, kh, ka, zh, za = factor(), factor(), factor(), factor(), factor()
    ov = dict(mass=p.mass * m, f_h=p.f_h * math.sqrt(kh / m), f_alpha=p.f_alpha * math.sqrt(ka / m),
              zeta_h=p.zeta_h * zh, zeta_alpha=p.zeta_alpha * za)
    if p.n_dof == 3:
        ov["f_beta"] = p.f_beta * factor()
    return p.with_overrides(**ov)


def _realisation(base: Scenario, i: int) -> Scenario:
    """Turbulence: a new realisation per draw. Deterministic gusts stay nominal, so only the wing varies."""
    if base.gust.kind == "turbulence":
        return replace(base, gust=replace(base.gust, seed=base.gust.seed + i), seed=i)
    return base


def _summary(metrics: list[dict]) -> dict:
    ok = [m for m in metrics if not m["exceeded_limits"]]
    out: dict = {"n": len(metrics), "failures": len(metrics) - len(ok), "failure_rate": 1.0 - len(ok) / len(metrics)}
    for key in ("rms_h_mm", "cost_J"):
        vals = [m[key] for m in ok if m[key] is not None]
        out[key] = ({"mean": float(np.mean(vals)), "median": float(np.median(vals)), "std": float(np.std(vals))}
                    if vals else None)
    return out


def robustness_study(p: WingParams, n_draws: int = 20, controllers: list[str] | None = None,
                     out_dir: Path | None = None, make_figure: bool = True, log=print) -> dict:
    out_dir = Path(out_dir or OUTPUTS_DIR / p.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    Uf = flutter_analysis(p).flutter_speed
    controllers = controllers or [c for c in ("lqr", "lqg", "neural") if c != "neural" or neural_available(p)]
    scen = scenarios_for(p.name)
    turb = scen["turbulence_patch"]["scenario"]
    res: dict = {
        "metadata": {"version": __version__, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "draws": n_draws, "note": "Every value in this file is produced by flutter_ai.robustness."},
        "controllers": controllers, "levels": list(LEVELS), "parameters": {}, "sensor_noise": {}, "rate_limit": {},
    }

    def run(wings_or_params, scenario_fn, design=p):
        return {c: _summary([simulate(w, make_controller(c), scenario_fn(i), Uf, design_params=design).metrics
                             for i, w in enumerate(wings_or_params)]) for c in controllers}

    log(f"Robustness study for '{p.name}' ({n_draws} draws per point; controllers designed for the nominal wing)")
    wings_by_level = {}
    for level in LEVELS:
        rng = np.random.default_rng(int(round(level * 1000)))
        wings = [perturbed_wing(p, level, rng) for _ in range(n_draws)]
        speeds = [flutter_analysis(w).flutter_speed for w in wings]
        ratios = [None if s is None else float(turb.velocity_ratio * Uf / s) for s in speeds]
        wings_by_level[level] = (wings, ratios)
    for sname in ("discrete_gust", "turbulence_patch"):
        base = scen[sname]["scenario"]
        res["parameters"][sname] = {}
        for level in LEVELS:
            wings, ratios = wings_by_level[level]
            row = run(wings, lambda i, b=base: _realisation(b, i))
            valid = [r for r in ratios if r is not None]
            row["true_speed_ratio"] = {"min": min(valid), "max": max(valid), "mean": float(np.mean(valid))}
            res["parameters"][sname][f"{level:.2f}"] = row
            log(f"  {sname:17s} ±{100 * level:3.0f} %  (true U/U_F {row['true_speed_ratio']['min']:.2f}"
                f"–{row['true_speed_ratio']['max']:.2f})  " + "  ".join(
                    f"{c} fail {row[c]['failures']:2d}/{n_draws}" for c in controllers))

    for noise in NOISE_LEVELS:
        row = run([p] * n_draws, lambda i, n=noise: _realisation(replace(turb, sensor_noise=n), i))
        res["sensor_noise"][f"{noise:.2f}"] = row
        log(f"  sensor noise {noise:.2f}  " + "  ".join(
            f"{c} fail {row[c]['failures']:2d}, J {row[c]['cost_J']['mean']:.2f}" if row[c]["cost_J"]
            else f"{c} fail {row[c]['failures']:2d}" for c in controllers))

    if p.actuation == "flap_servo":
        for lim in RATE_LIMITS_DEG_S:
            plant = replace(p, u_rate_max=None if lim is None else (math.radians(lim),))
            row = run([plant] * n_draws, lambda i: _realisation(turb, i))
            key = "none" if lim is None else f"{lim:.0f}"
            res["rate_limit"][key] = row
            log(f"  rate limit {key:>4s} deg/s  " + "  ".join(
                f"{c} fail {row[c]['failures']:2d}/{n_draws}" for c in controllers))

    path = out_dir / "robustness.json"
    path.write_text(json.dumps(res, indent=2))
    log(f"Saved {path}")
    if make_figure:
        from . import plots

        log(f"Saved {plots.robustness_figure(res, out_dir / 'fig10_robustness.png')}")
    return res
