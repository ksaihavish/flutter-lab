"""
Physics-based parameter identification from measured time histories (Phase 4.0).

    python -m flutter_ai identify data/<dataset> [--params f_h,f_alpha,zeta_h,zeta_alpha] [--delay 1]

Output-error method with multiple shooting: each run is cut into short segments (default 0.25 s); every segment
starts from the measured state (the unmeasured aerodynamic lag states are rebuilt from the measured downwash and
recorded gust, as the on-board observer does) and is simulated with the recorded airspeed, command and gust. The
chosen physical parameters are adjusted by bounded nonlinear least squares until the simulated plunge / pitch
(/ flap) match the measurements. Segments are needed above the flutter speed: replaying a whole run through an
unstable model amplifies rounding errors exponentially. The model structure
(Theodorsen/Wagner aerodynamics, typical section) is fixed and only physical constants are identified, so the
result is always a valid aeroelastic model: this is the legitimate "physics-informed" part of the project.
Uncertainty: 1-sigma from the Gauss-Newton covariance at the optimum (valid for small, white residuals).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .aero import JONES_EPS
from .controllers.costs import reference_scales
from .data import params_from_metadata, read_run_csv
from .model import discrete_state_space, downwash_rows
from .params import WingParams
from .paths import OUTPUTS_DIR

IDENTIFIABLE = ("f_h", "f_alpha", "zeta_h", "zeta_alpha", "x_alpha", "r_alpha_sq", "f_beta", "zeta_beta", "mass")
DEFAULT_PARAMS = ("f_h", "f_alpha", "zeta_h", "zeta_alpha")
SIGNED = {"x_alpha": 0.2}  # additive bound (semichords) instead of a factor for parameters that may be ~0


def load_runs(folder: str | Path) -> list[dict[str, np.ndarray]]:
    runs = [read_run_csv(f) for f in sorted((Path(folder) / "runs").glob("*.csv"))]
    if not runs:
        raise FileNotFoundError(f"No runs/*.csv in {folder} (see data/README.md)")
    return runs


def _measured_state(p: WingParams, run: dict[str, np.ndarray], U: float, dt: float) -> np.ndarray:
    """(T, N) [q, q', z]: measured q and q' (finite differences if rates are not recorded); lag states z rebuilt
    by integrating dz/dt = downwash + w_gust - lambda z with the measured motion (first-order hold)."""
    T, n = len(run["t"]), p.n_dof
    q = np.column_stack([run[name] if name in run else np.zeros(T) for name in p.dof_names])
    qd = np.column_stack([run[f"{name}_dot"] if f"{name}_dot" in run else np.gradient(q[:, j], dt)
                          for j, name in enumerate(p.dof_names)])
    dw_dq, dw_dqdot = downwash_rows(p, U)
    Q = q @ dw_dq + qd @ dw_dqdot  # downwash from the measured motion
    w = np.nan_to_num(run.get("w_gust", np.zeros(T)))  # recorded gust, held over each sample (as in the model)
    lam = np.array(JONES_EPS) * U / p.b
    # First-order hold (downwash linear between samples): the fast lag state settles within ~2 samples, so a
    # zero-order hold is not accurate enough here.
    decay = np.exp(-lam * dt)
    b0 = (1.0 - decay) / lam
    b1 = (dt / lam - b0 / lam) / dt
    z = np.zeros((T, 2))
    for k in range(T - 1):
        z[k + 1] = decay * z[k] + (b0 - b1) * Q[k] + b1 * Q[k + 1] + b0 * w[k]
    return np.hstack([q, qd, z])[:, :2 * n + 2]


def simulate_recorded(p: WingParams, run: dict[str, np.ndarray], input_delay_steps: int = 1,
                      segment_steps: int | None = None) -> np.ndarray:
    """
    Model response (T, n_dof) to a recorded run: its airspeed, commands (after the loop delay) and gust.
    segment_steps: None = one simulation from the initial state; otherwise restart from the measured state
    every segment_steps samples (multiple shooting).
    """
    t = run["t"]
    dt = float(np.median(np.diff(t)))
    U = float(np.nanmean(run["U"]))
    sys = discrete_state_space(p, U, dt)
    n, T = p.n_dof, len(t)
    measured = _measured_state(p, run, U, dt)
    x = measured[0].copy()
    x[2 * n:] = 0.0  # the run starts from rest aerodynamically
    u = np.column_stack([run.get(name, np.zeros(T)) for name in p.input_names])
    d = int(input_delay_steps)
    u_applied = np.vstack([np.zeros((d, u.shape[1])), u[:T - d]]) if d else u
    if p.u_rate_max is not None:
        rate, act = np.asarray(p.u_rate_max) * dt, np.zeros(u.shape[1])
        for k in range(T):
            act = act + np.clip(u_applied[k] - act, -rate, rate)
            u_applied[k] = act
    w = np.nan_to_num(run.get("w_gust", np.zeros(T)))
    out = np.zeros((T, n))
    for k in range(T):
        if segment_steps and k and k % segment_steps == 0:
            x = measured[k].copy()
        out[k] = x[:n]
        x = sys.Ad @ x + sys.Bd @ u_applied[k] + sys.Gd[:, 0] * w[k]
    return out


def _measured(p: WingParams, run: dict) -> tuple[np.ndarray, list[int]]:
    cols = [j for j, name in enumerate(p.dof_names) if name in run]
    return np.column_stack([run[p.dof_names[j]] for j in cols]), cols


def identify(nominal: WingParams, runs: list[dict], names=DEFAULT_PARAMS, input_delay_steps: int = 1,
             segment_s: float = 0.25) -> dict:
    names = tuple(names)
    bad = [n for n in names if n not in IDENTIFIABLE]
    if bad:
        raise ValueError(f"Cannot identify {bad}; choose from {IDENTIFIABLE}")
    if nominal.n_dof == 2 and any(n in ("f_beta", "zeta_beta") for n in names):
        raise ValueError("f_beta / zeta_beta need a flap (3-DOF) model")
    ref = reference_scales(nominal)[:nominal.n_dof]
    x0 = np.array([getattr(nominal, n) for n in names], dtype=float)
    lo = np.array([x - SIGNED[n] if n in SIGNED else 0.5 * x for n, x in zip(names, x0)])
    hi = np.array([x + SIGNED[n] if n in SIGNED else 2.0 * x for n, x in zip(names, x0)])
    scale = np.array([SIGNED.get(n, abs(x) or 1.0) for n, x in zip(names, x0)])

    def residual(theta: np.ndarray) -> np.ndarray:
        p = nominal.with_overrides(**dict(zip(names, map(float, theta))))
        parts = []
        for run in runs:
            meas, cols = _measured(p, run)
            steps = max(1, int(round(segment_s / float(np.median(np.diff(run["t"]))))))
            sim = simulate_recorded(p, run, input_delay_steps, steps)[:, cols]
            parts.append(((sim - meas) / ref[cols]).ravel() / np.sqrt(meas.size))
        return np.concatenate(parts)

    r0 = residual(x0)
    sol = least_squares(residual, x0, bounds=(lo, hi), x_scale=scale, method="trf")
    dof = max(1, sol.fun.size - len(names))
    sigma2 = 2.0 * sol.cost / dof
    try:
        cov = sigma2 * np.linalg.inv(sol.jac.T @ sol.jac)
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        std = np.full(len(names), np.nan)
    rows = [{"parameter": n, "nominal": float(x), "identified": float(v), "std": float(s),
             "change_pct": float(100 * (v - x) / x) if x else None, "at_bound": bool(np.isclose(v, a) or np.isclose(v, b))}
            for n, x, v, s, a, b in zip(names, x0, sol.x, std, lo, hi)]
    return {"parameters": rows, "rms_error_before": float(np.sqrt(np.sum(r0**2))),
            "rms_error_after": float(np.sqrt(np.sum(sol.fun**2))), "converged": bool(sol.success),
            "message": sol.message, "runs": len(runs), "input_delay_steps": input_delay_steps,
            "segment_s": segment_s}


def identify_dataset(folder: str | Path, names=DEFAULT_PARAMS, input_delay_steps: int = 1, log=print) -> dict:
    folder = Path(folder)
    meta = json.loads((folder / "metadata.json").read_text())
    nominal = params_from_metadata(meta, name=folder.name)
    res = identify(nominal, load_runs(folder), names, input_delay_steps)
    log(f"Identification on {folder.name} ({res['runs']} runs, delay {input_delay_steps} step(s)); "
        f"normalised RMS error {res['rms_error_before']:.4f} -> {res['rms_error_after']:.4f}")
    for r in res["parameters"]:
        log(f"  {r['parameter']:11s} nominal {r['nominal']:.5g}  identified {r['identified']:.5g} ± {r['std']:.2g}"
            f"  ({r['change_pct']:+.2f} %){'  AT BOUND' if r['at_bound'] else ''}")
    out = OUTPUTS_DIR / "identification" / f"{folder.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    log(f"Saved {out}")
    return res
