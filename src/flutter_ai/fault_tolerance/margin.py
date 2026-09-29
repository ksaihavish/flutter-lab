"""
On-board flutter-margin estimator: the physics-informed part of the supervisor's report.

Every call fits a few physical constants of the typical-section model (default: plunge and pitch stiffness, flap
effectiveness) to the last seconds of measurements by minimising the nominal model's one-step prediction error on
the rate channels (grey-box prediction-error identification; the aero-lag states are rebuilt from the measured
motion, as the on-board observer does). The turbulence is not measured on board, so it acts as process noise and
biases the fit slightly; the bias is removed by expressing every estimate relative to the same fit on the healthy
wing at the start of the flight (self-calibration).

From the identified wing it reports:
  - its open-loop flutter speed and the margin to the current airspeed (with a 1-sigma band from the Gauss-Newton
    covariance, propagated by finite differences);
  - for every supervisor mode, the largest spectral radius of the closed loop formed by that mode's controller
    (designed and gain-scheduled for the healthy wing) and the identified wing, over the airspeeds between the current
    one and the mode's target (the autopilot passes through all of them): < 1 = predicted stable all the way.
The prediction is linear (no saturation or rate limit) and covers only faults that change the wing's physics
(stiffness, flap effectiveness); actuator, sensor, timing and turbulence faults leave the estimate unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import least_squares

from ..flutter import flutter_analysis
from ..params import WingParams
from .diagnostics import one_step_residual
from .simulate import FT_MODES, SPEED_STEP
from .stability import mode_matrices

PARAMS = ("f_alpha", "flap_span_fraction")
BOUNDS = (0.3, 2.0)  # identified value / nominal
PATH_POINTS = 9  # airspeeds checked between the current one and a mode's target


@dataclass
class Fit:
    theta: np.ndarray  # identified values (not yet calibrated)
    std: np.ndarray
    cost: float
    converged: bool


@dataclass
class MarginEstimate:
    time: float
    airspeed: float
    params: dict[str, float]  # calibrated estimate / nominal (1 = healthy)
    params_std: dict[str, float]
    flutter_speed: float | None  # open-loop flutter speed of the identified wing [m/s]
    flutter_speed_std: float | None
    margin: float | None  # flutter_speed / airspeed - 1 (negative: above open-loop flutter, the controller must hold)
    mode_spectral_radius: dict[str, float] = field(default_factory=dict)  # < 1: predicted stable in that mode
    converged: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def applied_commands(p: WingParams, us: np.ndarray, k: int, dt: float, delay: int = 1) -> np.ndarray:
    """Commands 0..k-1 as the flap servo receives them on the healthy aircraft: shifted by the nominal loop delay and
    passed through the known slew-rate limit (as the LQG filter's input model). Without the limit, the fit blames the
    unmodelled rate saturation on the structure (plunge stiffness drifted by up to +40 % in trials)."""
    out = np.vstack([np.zeros((delay, us.shape[1])), us[:k - delay]]) if delay else us[:k].copy()
    if p.u_rate_max is not None:
        step, act = np.asarray(p.u_rate_max) * dt, np.zeros(us.shape[1])
        for j in range(len(out)):
            act = act + np.clip(out[j] - act, -step, step)
            out[j] = act
    return out


def fit_window(p: WingParams, U: float, dt: float, y: np.ndarray, u_applied: np.ndarray,
               names: tuple[str, ...] = PARAMS, start: np.ndarray | None = None) -> Fit:
    """Fit `names` (as factors of the nominal values) to one measurement window."""
    nominal = np.array([getattr(p, n) for n in names], dtype=float)

    def residual(k: np.ndarray) -> np.ndarray:
        q = p.with_overrides(**{n: float(v) for n, v in zip(names, k * nominal)})
        r = one_step_residual(q, U, dt, y, u_applied)
        return r.ravel() / np.sqrt(r.size)

    k0 = np.ones(len(names)) if start is None else np.clip(start, *BOUNDS)
    sol = least_squares(residual, k0, bounds=BOUNDS, x_scale=0.1, diff_step=1e-4, method="trf")
    dof = max(1, sol.fun.size - len(names))
    try:  # Gauss-Newton; the 1/sqrt(N) scaling cancels. Optimistic: gust-driven residuals are not white.
        cov = (2.0 * sol.cost / dof) * np.linalg.inv(sol.jac.T @ sol.jac)
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        std = np.full(len(names), np.nan)
    return Fit(theta=sol.x, std=std, cost=float(sol.cost), converged=bool(sol.success))


def _flutter_speed(p: WingParams) -> float | None:
    return flutter_analysis(p).flutter_speed


def estimate(p: WingParams, flutter_speed_nominal: float, t: float, U: float, dt: float, y: np.ndarray,
             u_applied: np.ndarray, baseline: Fit | None, names: tuple[str, ...] = PARAMS,
             modes: tuple[str, ...] = tuple(FT_MODES), cruise: float | None = None) -> tuple[MarginEstimate, Fit]:
    """One estimate from a measurement window; `baseline` is the fit on the healthy wing (None: no calibration)."""
    fit = fit_window(p, U, dt, y, u_applied, names, start=None if baseline is None else baseline.theta)
    k = fit.theta / baseline.theta if baseline is not None else fit.theta
    k_std = fit.std / baseline.theta if baseline is not None else fit.std
    p_hat = p.with_overrides(**{n: float(v * getattr(p, n)) for n, v in zip(names, k)})
    Uf = _flutter_speed(p_hat)
    Uf_std = None
    if Uf is not None:
        grad = []
        for j in range(len(names)):  # d U_F / d k_j by finite differences
            step = 0.01
            kj = k.copy()
            kj[j] += step
            Uf_j = _flutter_speed(p.with_overrides(**{m: float(v * getattr(p, m)) for m, v in zip(names, kj)}))
            grad.append(0.0 if Uf_j is None else (Uf_j - Uf) / step)
        Uf_std = float(np.sqrt(np.sum((np.array(grad) * k_std) ** 2)))
    radius = {}
    cruise = cruise if cruise is not None else U
    for m in modes:  # worst case on the way: the autopilot passes every airspeed between here and the target
        ratio = FT_MODES[m]["speed_ratio"]
        Um = cruise if ratio is None else ratio * flutter_speed_nominal
        path = np.unique(np.round(np.linspace(U, Um, PATH_POINTS) / SPEED_STEP) * SPEED_STEP)
        radius[m] = max(float(np.max(np.abs(np.linalg.eigvals(
            mode_matrices(p, float(Us), dt, plant=p_hat)[FT_MODES[m]["controller"]])))) for Us in path)
    est = MarginEstimate(time=t, airspeed=U, params=dict(zip(names, map(float, k))),
                         params_std=dict(zip(names, map(float, k_std))), flutter_speed=Uf, flutter_speed_std=Uf_std,
                         margin=None if Uf is None else Uf / U - 1.0, mode_spectral_radius=radius,
                         converged=fit.converged)
    return est, fit
