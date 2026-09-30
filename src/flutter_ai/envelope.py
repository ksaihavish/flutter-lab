"""
Operating envelope: up to which airspeed does each controller keep the wing within limits?

    python -m flutter_ai envelope --preset uav_flap     # -> outputs/<preset>/envelope.json + fig16_envelope.png

The benchmark tests turbulence at one airspeed (1.1 U_F) and gusts at three; a single web-app run is one draw of the
disturbance. This sweeps the airspeed and reports, per controller:
  survival      failures over the benchmark's seed draws (experiments.draw_scenario) for the turbulence and discrete-gust
                scenarios, with the preset's actuator and with its slew-rate limit removed, so a failure can be traced
                to the servo's speed or to the control law itself
  small_signal  spectral radius of the linear closed loop (no saturation, no slew limit, nominal wing, 1-sample delay):
                LQR as implemented (lqr.closed_loop_radius, aero-lag observer), LQG and the neural policy's
                linearisation on the controller-bank state (fault_tolerance.stability.mode_matrices). < 1 = stable.
  rate demand   the commanded flap rate (before the slew limit) in the unlimited-servo turbulence runs: RMS, 99th
                percentile, share of steps above the real servo's limit
Each controller is designed (gain-scheduled) at the airspeed it flies; the neural policy has one set of weights and
extrapolates outside its training range (model metadata).
"""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .experiments import draw_scenario, neural_available, run_case
from .flutter import flutter_analysis
from .params import WingParams, get_preset
from .paths import OUTPUTS_DIR
from .scenarios import scenarios_for

RATIOS = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)
SCENARIO_NAMES = ("turbulence_patch", "discrete_gust")


def _run(args: tuple) -> dict:
    preset, scen, ratio, ctrl, unlimited, seed, Uf = args
    p = get_preset(preset)
    limit = None if p.u_rate_max is None else np.degrees(np.asarray(p.u_rate_max, dtype=float))
    if unlimited:
        p = replace(p, u_rate_max=None)
    base = replace(scenarios_for(preset)[scen]["scenario"], velocity_ratio=ratio)
    res = run_case(p, Uf, draw_scenario(base, seed), ctrl)
    m = res.metrics
    out = {"failed": bool(m["exceeded_limits"]), "failure_time_s": m["failure_time_s"], "rms_h_mm": m["rms_h_mm"],
           "cost_J": m["cost_J"]}
    if len(res.u) > 1:
        rate = np.degrees(np.abs(np.diff(res.u, axis=0)) / base.dt).max(axis=1)
        out["rate_rms_deg_s"] = float(np.sqrt(np.mean(rate**2)))
        out["rate_p99_deg_s"] = float(np.percentile(rate, 99))
        out["rate_above_limit"] = None if limit is None else float(np.mean(rate > limit.min()))
    return out


def small_signal(p: WingParams, Uf: float, ratios=RATIOS, dt: float = 0.005) -> dict:
    """Spectral radius and the worst mode's growth rate [1/s] per controller and airspeed (linear, see module doc)."""
    from .controllers import LQRController
    from .controllers.lqr import closed_loop_radius
    from .model import discrete_state_space

    out: dict = {}
    for r in ratios:
        U = r * Uf
        rho = {"open_loop": float(np.max(np.abs(np.linalg.eigvals(discrete_state_space(p, U, dt).Ad))))}
        lqr = LQRController()
        lqr.reset(p, U, dt, 1)
        rho["lqr"] = closed_loop_radius(p, U, dt, lqr._gain(), 1)
        if neural_available(p):
            from .fault_tolerance.stability import mode_matrices

            mm = mode_matrices(p, U, dt)
            rho["lqg"] = float(np.max(np.abs(np.linalg.eigvals(mm["lqg"]))))
            rho["neural"] = float(np.max(np.abs(np.linalg.eigvals(mm["neural"]))))
        out[f"{r:.2f}"] = {k: {"spectral_radius": v, "growth_per_s": math.log(v) / dt} for k, v in rho.items()}
    return out


def _highest_clean(rows: list[dict]) -> float | None:
    """Highest tested ratio >= 1.0 at which this and every lower tested ratio >= 1.0 had no failure."""
    best = None
    for row in rows:
        if row["ratio"] < 1.0 - 1e-9:
            continue
        if row["failures"]:
            break
        best = row["ratio"]
    return best


def envelope_study(p: WingParams, seeds: int = 20, ratios=RATIOS, controllers: list[str] | None = None,
                   workers: int = 0, out_dir: Path | None = None, make_figure: bool = True, log=print) -> dict:
    out_dir = Path(out_dir or OUTPUTS_DIR / p.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    fa = flutter_analysis(p)
    Uf = fa.flutter_speed
    if Uf is None:
        raise ValueError(f"No flutter found for '{p.name}'")
    controllers = controllers or [c for c in ("lqr", "lqg", "neural") if c != "neural" or neural_available(p)]
    servos = ("actual", "unlimited") if p.u_rate_max is not None else ("actual",)
    training = None
    if neural_available(p):
        from .controllers import policy_path
        from .controllers.neural import NeuralPolicy

        training = NeuralPolicy.load(policy_path(p.name)).meta.get("velocity_ratio_range")
    workers = workers or max(1, min(4, (os.cpu_count() or 2) // 2))
    keys = [(sc, r, c, sv) for sc in SCENARIO_NAMES for sv in servos for c in controllers for r in ratios]
    jobs = [(p.name, sc, r, c, sv == "unlimited", s, Uf) for sc, r, c, sv in keys for s in range(seeds)]
    log(f"Envelope study for '{p.name}': {len(jobs)} flights on {workers} processes "
        f"(U_F = {Uf:.2f} m/s, ratios {ratios[0]}-{ratios[-1]}, {seeds} seeds)")
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(_run, jobs, chunksize=8))
    res: dict = {
        "metadata": {"version": __version__, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "seeds": list(range(seeds)), "note": "Every value in this file is produced by flutter_ai.envelope."},
        "preset": p.name, "flutter_speed": Uf, "divergence_speed": fa.divergence_speed, "ratios": list(ratios),
        "controllers": controllers, "servo_rate_limit_deg_s": None if p.u_rate_max is None
        else [float(v) for v in np.degrees(p.u_rate_max)], "neural_training_range": training,
        "scenarios": {}, "highest_clean_ratio": {},
    }
    for i, (sc, r, c, sv) in enumerate(keys):
        ms = runs[i * seeds:(i + 1) * seeds]
        ok = [m for m in ms if not m["failed"]]
        row = {"ratio": r, "velocity": r * Uf, "n": seeds, "failures": seeds - len(ok),
               "failure_times_s": [m["failure_time_s"] for m in ms if m["failed"]],
               "rms_h_mm_median": float(np.median([m["rms_h_mm"] for m in ok])) if ok else None,
               "cost_J_median": float(np.median([m["cost_J"] for m in ok])) if ok else None}
        for k in ("rate_rms_deg_s", "rate_p99_deg_s", "rate_above_limit"):
            vals = [m[k] for m in ms if m.get(k) is not None]
            row[k] = float(np.mean(vals)) if vals else None
        res["scenarios"].setdefault(sc, {}).setdefault(sv, {}).setdefault(c, []).append(row)
    for sc, by_servo in res["scenarios"].items():
        for sv, by_ctrl in by_servo.items():
            res["highest_clean_ratio"].setdefault(sc, {})[sv] = {c: _highest_clean(rows) for c, rows in by_ctrl.items()}
            for c, rows in by_ctrl.items():
                log(f"  {sc:17s} servo {sv:9s} {c:7s} failures: " + " ".join(f"{row['failures']:2d}" for row in rows))
    res["small_signal"] = small_signal(p, Uf, ratios)
    path = out_dir / "envelope.json"
    path.write_text(json.dumps(res, indent=2))
    log(f"Saved {path}")
    if make_figure:
        from . import plots

        log(f"Saved {plots.envelope_figure(res, out_dir / 'fig16_envelope.png')}")
    return res
