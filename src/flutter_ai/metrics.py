"""
Performance metrics. No NaN-masking: a run that exceeded its structural limits
is reported as a failure, and time-based metrics are None when undefined.
"""

from __future__ import annotations

import numpy as np

from .controllers.costs import input_weight, rate_weight, state_weight
from .params import WingParams

SETTLE_BAND_FRACTION = 0.02  # settled = within 2 % of the structural limits
COST_AGGRESSIVENESS = 0.5  # effort weight used to score every run with the same J


def _settling_time(t: np.ndarray, outside: np.ndarray) -> float | None:
    """First time after which the response stays inside the band for the rest of the run."""
    if outside[-1]:
        return None
    idx = np.where(outside)[0]
    return 0.0 if len(idx) == 0 else float(t[idx[-1] + 1])


def compute_metrics(result, params: WingParams, scenario) -> dict:
    t, x, u = result.t, result.x, result.u
    dt = scenario.dt
    u_max = np.asarray(params.u_max, dtype=float)
    h_mm = x[:, 0] * 1000.0
    a_deg = np.degrees(x[:, 1])

    band_h = SETTLE_BAND_FRACTION * params.h_limit * 1000.0
    band_a = SETTLE_BAND_FRACTION * params.alpha_limit_deg
    outside = (np.abs(h_mm) > band_h) | (np.abs(a_deg) > band_a)
    settle = None if result.exceeded_limits else _settling_time(t, outside)

    gust = scenario.gust
    gust_active = gust.kind != "none" and gust.amplitude != 0.0
    gust_end = gust.start + gust.duration
    after_gust = t >= gust.start
    peak_h_gust = float(np.max(np.abs(h_mm[after_gust]))) if gust_active and after_gust.any() else None
    if settle is not None and gust_active:
        recovery = max(0.0, settle - gust_end) if t[-1] >= gust_end else None
    else:
        recovery = None

    u_norm = u / u_max
    du = np.diff(u_norm, axis=0)
    # RMS per-step change of the normalised command: high values = chatter / actuator wear
    rate_rms = float(np.sqrt(np.mean(np.sum(du**2, axis=1)))) if len(du) else 0.0
    # The quadratic cost every controller is designed or trained to minimise (controllers/costs.py), with one
    # fixed effort weight so runs are comparable. None for failed runs: a truncated run would score too well.
    xq = x[:, :state_weight(params).shape[0]]
    du_all = np.diff(np.vstack([np.zeros((1, u.shape[1])), u]), axis=0)  # command changes, from rest
    cost = float(dt * (np.einsum("ti,ij,tj->", xq, state_weight(params), xq)
                       + np.einsum("ti,ij,tj->", u, input_weight(params, COST_AGGRESSIVENESS), u)
                       + np.einsum("ti,ij,tj->", du_all, rate_weight(params), du_all)))
    switches = sum(1 for a, b in zip(result.mode, result.mode[1:]) if a != b)
    llm_calls = sum(1 for d in result.decisions if d.source.startswith("llm"))
    llm_fallbacks = sum(1 for d in result.decisions if d.source == "llm_fallback")

    return {
        "exceeded_limits": bool(result.exceeded_limits),
        "failure_time_s": result.failure_time,
        "peak_h_mm": float(np.max(np.abs(h_mm))),
        "peak_alpha_deg": float(np.max(np.abs(a_deg))),
        "rms_h_mm": float(np.sqrt(np.mean(h_mm**2))),
        "rms_alpha_deg": float(np.sqrt(np.mean(a_deg**2))),
        "peak_beta_deg": float(np.max(np.abs(np.degrees(x[:, 2])))) if params.n_dof == 3 else None,
        "settling_time_s": settle,
        "gust_peak_h_mm": peak_h_gust,
        "gust_recovery_s": recovery,
        "cost_J": None if result.exceeded_limits else cost,
        "control_energy": float(np.sum(u_norm**2) * dt),
        "peak_control_fraction": float(np.max(np.abs(u_norm))) if len(u_norm) else 0.0,
        "saturation_fraction": float(np.mean(np.any(np.abs(u_norm) >= 0.999, axis=1))) if len(u_norm) else 0.0,
        "control_rate_rms": rate_rms,
        "mode_switches": switches,
        "supervisor_decisions": len(result.decisions),
        "llm_calls": llm_calls,
        "llm_fallbacks": llm_fallbacks,
        "duration_s": float(t[-1]),
    }
