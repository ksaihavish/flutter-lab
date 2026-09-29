"""
LQG baseline: LQR on the plant augmented with a first-order turbulence model, plus a steady-state
Kalman filter that estimates the aero-lag states and the gust velocity from the measurements.

Why: plain LQR is optimal for initial conditions but knows nothing about how gusts enter, and on
`uav_flap` it roughly triples the gust cost below flutter compared with no control. This design
gets the same turbulence knowledge the neural policy is trained with (AR(1) gusts, correlation
time 0.05 s, the benchmark's turbulence intensity), which makes it the fair classical baseline
(Phase 4.2). Both minimise the same cost J (controllers/costs.py).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

from ..model import discrete_state_space
from ..params import WingParams
from .base import Controller
from .costs import input_weight, rate_weight, state_weight
from .lqr import dlqr_with_rate
from .neural import state_scales

MEASUREMENT_NOISE_FRACTION = 1e-3  # assumed sensor noise std, as a fraction of the reference scales
MODEL_NOISE_FRACTION = 1e-3  # small process noise on every state (model error), same scaling


def gust_augmented(p: WingParams, U: float, dt: float, correlation_time: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Plant plus gust state, [x, w_g]:  x_{k+1} = Ad x_k + Bd u + Gd w_k,  w_{k+1} = phi w_k + noise.
    phi matches the AR(1) turbulence of gusts.GustSpec and of the training episodes.
    """
    sys = discrete_state_space(p, U, dt)
    N, m = sys.Bd.shape
    A = np.zeros((N + 1, N + 1))
    A[:N, :N] = sys.Ad
    A[:N, N] = sys.Gd[:, 0]
    A[N, N] = np.exp(-dt / correlation_time)
    B = np.vstack([sys.Bd, np.zeros((1, m))])
    return A, B


def lqg_regulator_gain(p: WingParams, U: float, dt: float, aggressiveness: float, delay: int,
                       correlation_time: float, rate_multiplier: float = 1.0) -> np.ndarray:
    """K for u_k = -K [x_k, w_k, u_{k-1}, ..., u_{k-D}] (same cost as LQR, incl. the rate penalty);
    the gust column is a feedforward term."""
    A, B = gust_augmented(p, U, dt, correlation_time)
    N = state_weight(p).shape[0]
    Q = np.zeros_like(A)
    Q[:N, :N] = state_weight(p) * dt
    return dlqr_with_rate(A, B, Q, input_weight(p, aggressiveness) * dt, rate_weight(p) * dt * rate_multiplier,
                          delay)


def kalman_gain(p: WingParams, U: float, dt: float, gust_rms: float, correlation_time: float,
                measurement_noise: float = MEASUREMENT_NOISE_FRACTION) -> np.ndarray:
    """Steady-state filter gain L for [x, w] from y = [q, q'] (update: x_hat += L (y - C x_hat))."""
    A, _ = gust_augmented(p, U, dt, correlation_time)
    N = A.shape[0] - 1
    n2 = 2 * p.n_dof
    C = np.zeros((n2, N + 1))
    C[:, :n2] = np.eye(n2)
    phi = A[N, N]
    scales = state_scales(p)
    W = np.diag(np.append((MODEL_NOISE_FRACTION * scales) ** 2, gust_rms**2 * (1.0 - phi**2)))
    V = np.diag((measurement_noise * scales[:n2]) ** 2)
    P = solve_discrete_are(A.T, C.T, W, V)
    return P @ C.T @ np.linalg.inv(C @ P @ C.T + V)


class LQGController(Controller):
    """
    gust_rms / correlation_time: turbulence model the design assumes. None = the benchmark's
    turbulence_patch scenario for this preset (the same statistics the neural policy trains on).
    design_delay: loop delay the regulator and filter assume. None = the true delay.
    rate_multiplier / measurement_noise: design knobs (controllers/design.py). None = tuned values, if any.
    """

    name = "lqg"
    label = "LQG (turbulence model)"

    def __init__(self, gust_rms: float | None = None, correlation_time: float | None = None,
                 design_delay: int | None = None, rate_multiplier: float | None = None,
                 measurement_noise: float | None = None):
        super().__init__()
        self.rate_multiplier = rate_multiplier
        self.measurement_noise = measurement_noise
        self.gust_rms = gust_rms
        self.correlation_time = correlation_time
        self.design_delay = design_delay

    def reset(self, params: WingParams, U: float, dt: float, delay: int = 1) -> None:
        super().reset(params, U, dt, delay)
        if self.gust_rms is None or self.correlation_time is None:
            from ..scenarios import scenarios_for  # local import: scenarios imports the simulator

            gust = scenarios_for(params.name)["turbulence_patch"]["scenario"].gust
            self._rms = self.gust_rms if self.gust_rms is not None else gust.amplitude
            self._tau = self.correlation_time if self.correlation_time is not None else gust.correlation_time
        else:
            self._rms, self._tau = self.gust_rms, self.correlation_time
        self._model_delay = self.delay if self.design_delay is None else int(self.design_delay)
        from .design import design_settings

        settings, warning = design_settings(params, "lqg")
        if warning:
            self.warnings.append(warning)
        self._rate = self.rate_multiplier if self.rate_multiplier is not None else settings["rate_multiplier"]
        noise = self.measurement_noise if self.measurement_noise is not None else settings["measurement_noise"]
        self._A, self._B = gust_augmented(params, U, dt, self._tau)
        self._L = kalman_gain(params, U, dt, self._rms, self._tau, noise)
        self._C = np.zeros((2 * params.n_dof, self._A.shape[0]))
        self._C[:, :2 * params.n_dof] = np.eye(2 * params.n_dof)
        self._gains: dict[float, np.ndarray] = {}
        self.x_pred = np.zeros(self._A.shape[0])
        self.gust_estimate = 0.0
        # The filter's input model includes the actuator's known slew-rate limit (a datasheet value), so a
        # rate-limited flap is not mistaken for a gust.
        self._rate_step = None if params.u_rate_max is None else np.asarray(params.u_rate_max, dtype=float) * dt
        self._actuator = np.zeros(len(self.u_max))

    def _gain(self) -> np.ndarray:
        key = round(self.aggressiveness, 3)
        if key not in self._gains:
            self._gains[key] = lqg_regulator_gain(self.params, self.U, self.dt, key, self._model_delay, self._tau,
                                                  self._rate)
        return self._gains[key]

    def compute(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        # The filter replaces the base-class aero-lag observer: only the measurements y are used.
        y = x_hat[:2 * self.params.n_dof]
        est = self.x_pred + self._L @ (y - self._C @ self.x_pred)
        self.gust_estimate = float(est[-1])
        d = self._model_delay
        K = self._gain()
        past = list(self.history)[:(K.shape[1] - len(est)) // len(self.u_max)]  # u_{k-1}.. (>= 1 block)
        u = -K @ np.concatenate([est, *past])
        u = np.clip(u, -self.u_max, self.u_max)
        applied = u if d == 0 else self.history[d - 1]  # the command reaching the actuator this step
        if self._rate_step is not None:
            self._actuator = self._actuator + np.clip(applied - self._actuator, -self._rate_step, self._rate_step)
            applied = self._actuator
        self.x_pred = self._A @ est + self._B @ applied
        return u
