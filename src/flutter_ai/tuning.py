"""
Design-knob tuning for the classical baselines ("retuned LQR / LQG"), on tuning episodes only.

    python -m flutter_ai tune --preset uav_flap      # -> models/classical_tuning_<preset>.json

The neural policy is optimised through the true plant (including the actuator's position and slew-rate limits)
on randomised episodes. A control engineer would likewise tune the classical designs' free knobs for the real
actuator, so both get the same chance. The objective J is not changed; only the design knobs are
(controllers/design.py):

    LQR  rate_multiplier                        extra weight on command changes in the design cost
    LQG  rate_multiplier, measurement_noise     and the sensor noise the Kalman filter assumes

The best setting has the fewest failures, then the lowest mean J over the surviving episodes. The tuning episodes
are the benchmark scenarios with seeds 1000-1009, which never appear in the benchmark itself (seeds 0-19).
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone

import numpy as np

from . import __version__
from .controllers import LQGController, LQRController
from .controllers.design import tuning_path
from .experiments import draw_scenario
from .flutter import flutter_speed
from .params import WingParams
from .scenarios import scenarios_for
from .simulation import simulate

RATE_MULTIPLIERS = (1.0, 3.0, 10.0, 30.0, 100.0)
NOISE_FRACTIONS = (1e-3, 1e-2, 3e-2, 1e-1)
TUNING_SEEDS = tuple(range(1000, 1010))


def _score(p: WingParams, Uf: float, make) -> dict:
    metrics = [simulate(p, make(), draw_scenario(spec["scenario"], s), Uf).metrics
               for spec in scenarios_for(p.name).values() for s in TUNING_SEEDS]
    ok = [m["cost_J"] for m in metrics if not m["exceeded_limits"]]
    return {"failures": len(metrics) - len(ok), "episodes": len(metrics),
            "mean_J": float(np.mean(ok)) if ok else None}


def _best(rows: list[dict]) -> dict:
    return min(rows, key=lambda r: (r["failures"], r["mean_J"] if r["mean_J"] is not None else np.inf))


def tune_classical(p: WingParams, log=print) -> dict:
    Uf = flutter_speed(p)
    log(f"Tuning classical design knobs for '{p.name}' on {len(TUNING_SEEDS)} seeds x 4 scenarios")
    lqr_rows = []
    for rho in RATE_MULTIPLIERS:
        row = {"rate_multiplier": rho, **_score(p, Uf, lambda rho=rho: LQRController(rate_multiplier=rho))}
        lqr_rows.append(row)
        log(f"  LQR rho {rho:6.1f}: failures {row['failures']:2d}/{row['episodes']}  mean J {row['mean_J']}")
    lqg_rows = []
    for rho, noise in itertools.product(RATE_MULTIPLIERS, NOISE_FRACTIONS):
        row = {"rate_multiplier": rho, "measurement_noise": noise,
               **_score(p, Uf, lambda rho=rho, noise=noise: LQGController(rate_multiplier=rho, measurement_noise=noise))}
        lqg_rows.append(row)
        log(f"  LQG rho {rho:6.1f} noise {noise:.0e}: failures {row['failures']:2d}/{row['episodes']}  "
            f"mean J {row['mean_J']}")
    lqr, lqg = _best(lqr_rows), _best(lqg_rows)
    out = {
        "metadata": {"version": __version__, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "tuning_seeds": list(TUNING_SEEDS), "note": "Produced by flutter_ai.tuning; see its docstring."},
        "params_fingerprint": p.fingerprint(),
        "lqr": {"rate_multiplier": lqr["rate_multiplier"]},
        "lqg": {"rate_multiplier": lqg["rate_multiplier"], "measurement_noise": lqg["measurement_noise"]},
        "grid": {"lqr": lqr_rows, "lqg": lqg_rows},
    }
    path = tuning_path(p.name)
    path.write_text(json.dumps(out, indent=2) + "\n")
    log(f"Chosen: LQR {out['lqr']}  LQG {out['lqg']}\nSaved {path}")
    return out
