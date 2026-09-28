"""Linear-quadratic regulator: the classical baseline every flutter-control paper needs."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

from ..model import discrete_state_space
from ..params import WingParams
from .base import Controller
from .costs import input_weight, state_weight


def delay_augmented(Ad: np.ndarray, Bd: np.ndarray, delay: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Plant with a `delay`-step input delay, state [x, u_{k-1}, ..., u_{k-d}]:
        x_{k+1} = Ad x_k + Bd u_{k-d},   new command u_k enters the history.
    """
    if delay == 0:
        return Ad, Bd
    n, m = Bd.shape
    N = n + m * delay
    A = np.zeros((N, N))
    A[:n, :n] = Ad
    A[:n, n + m * (delay - 1):] = Bd
    for j in range(1, delay):
        A[n + m * j:n + m * (j + 1), n + m * (j - 1):n + m * j] = np.eye(m)
    B = np.zeros((N, m))
    B[n:n + m, :] = np.eye(m)
    return A, B


def lqr_gain(p: WingParams, U: float, dt: float, aggressiveness: float = 0.5, delay: int = 0) -> np.ndarray:
    """
    Discrete-time LQR on the ZOH model at the controller's sample period, designed for
    the given input delay. Returns K for u_k = -K [x_k, u_{k-1}, ..., u_{k-d}].
    (A continuous-time design implemented at 200 Hz chatters at high gain.)
    """
    sys = discrete_state_space(p, U, dt)
    A, B = delay_augmented(sys.Ad, sys.Bd, delay)
    Q = np.zeros_like(A)
    n = sys.Ad.shape[0]
    Q[:n, :n] = state_weight(p) * dt
    R = input_weight(p, aggressiveness) * dt
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def closed_loop_radius(p: WingParams, U: float, dt: float, K: np.ndarray, actual_delay: int,
                       with_observer: bool = True) -> float:
    """
    Spectral radius (< 1 = stable) of the closed loop when the linear law
        u_k = -K [y_k, z_hat_k, u_{k-1}, ..., u_{k-d}]
    runs on a plant with `actual_delay` steps of input delay.

    with_observer=True models the controller exactly as implemented: the aero-lag states
    come from the on-board observer (which lags the true states by about half a step).
    with_observer=False feeds the true lag states (the idealisation the LQR design assumes).
    Used for delay-margin analysis of LQR and of the linearised neural policy.
    """
    from .base import AeroLagObserver

    sys = discrete_state_space(p, U, dt)
    Ad, Bd = sys.Ad, sys.Bd
    N, m = Bd.shape
    n2 = 2 * p.n_dof
    d_design = (K.shape[1] - N) // m
    D = max(actual_delay, d_design, 1)
    no = 2 if with_observer else 0
    hb = N + no  # start of the command history block
    S = hb + m * D  # closed-loop state [x, (z_hat), u_{k-1}, ..., u_{k-D}]

    law = np.zeros((m, S))  # u_k = law @ state
    if with_observer:
        law[:, :n2] = -K[:, :n2]
        law[:, N:N + 2] = -K[:, n2:N]
    else:
        law[:, :N] = -K[:, :N]
    law[:, hb:hb + m * d_design] = -K[:, N:]

    A = np.zeros((S, S))
    if actual_delay == 0:
        A[:N, :] = Bd @ law
        A[:N, :N] += Ad
    else:
        A[:N, :N] = Ad
        A[:N, hb + m * (actual_delay - 1):hb + m * actual_delay] = Bd
    if with_observer:
        obs = AeroLagObserver(p, U, dt)
        A[N:N + 2, :n2] = np.outer(obs.gain, obs.w_row)
        A[N:N + 2, N:N + 2] = np.diag(obs.decay)
    A[hb:hb + m, :] = law  # the new command becomes u_{k-1} at the next step
    for j in range(1, D):  # shift the history
        A[hb + m * j:hb + m * (j + 1), hb + m * (j - 1):hb + m * j] = np.eye(m)
    return float(np.max(np.abs(np.linalg.eigvals(A))))


class LQRController(Controller):
    """
    Full-state discrete LQR (aero-lag states from the on-board observer), designed for the
    loop's input delay using the controller's own command history.

    design_velocity: airspeed the gain is designed for. None = the actual airspeed (ideal
    gain scheduling). Set it to study robustness to off-design operation.
    """

    name = "lqr"
    label = "LQR (classical)"

    def __init__(self, design_velocity: float | None = None, design_delay: int | None = None):
        super().__init__()
        self.design_velocity = design_velocity
        self.design_delay = design_delay
        self._gains: dict[float, np.ndarray] = {}

    def reset(self, params: WingParams, U: float, dt: float, delay: int = 1) -> None:
        super().reset(params, U, dt, delay)
        self._gains = {}

    def _gain(self) -> np.ndarray:
        key = round(self.aggressiveness, 3)
        if key not in self._gains:
            U = self.design_velocity or self.U
            d = self.delay if self.design_delay is None else self.design_delay
            self._gains[key] = lqr_gain(self.params, U, self.dt, key, d)
        return self._gains[key]

    def compute(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        K = self._gain()
        d = (K.shape[1] - len(x_hat)) // len(self.u_max)
        past = list(self.history)[:d]
        return -K @ np.concatenate([x_hat, *past]) if d else -K @ x_hat
