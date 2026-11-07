"""
Controller interface.

Controllers only see *measurements* y = [q, q'] (h, alpha[, beta] and their rates) and
the airspeed. The two aerodynamic lag states are reconstructed on board by
AeroLagObserver, which integrates the known lag filter driven by the measured downwash
(exact without gusts; realistically imperfect with them).

Timing: a command computed at step k reaches the actuator `delay` steps later
(computation/transport delay; default 1 step = 5 ms). Controllers keep their own
command history so they can account for commands that are "in flight".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from ..aero import JONES_EPS
from ..model import downwash_rows, n_inputs
from ..params import WingParams

# Aggressiveness knob a in [0, 1] -> multiplier on the control-effort cost.
# a = 0 -> effort 10x more expensive (efficient), a = 1 -> 10x cheaper (aggressive).
DEFAULT_AGGRESSIVENESS = 0.5


def effort_cost_multiplier(aggressiveness: float) -> float:
    a = float(np.clip(aggressiveness, 0.0, 1.0))
    return 10.0 ** (1.0 - 2.0 * a)


class AeroLagObserver:
    def __init__(self, params: WingParams, U: float, dt: float):
        lam = np.array(JONES_EPS) * U / params.b
        self.decay = np.exp(-lam * dt)
        self.gain = (1.0 - self.decay) / lam
        dw_dq, dw_dqdot = downwash_rows(params, U)
        self.w_row = np.concatenate([dw_dq, dw_dqdot])
        self.z = np.zeros(2)

    def full_state(self, y: np.ndarray) -> np.ndarray:
        return np.concatenate([y, self.z])

    def update(self, y: np.ndarray) -> None:
        self.z = self.decay * self.z + self.gain * float(self.w_row @ y)


class Controller(ABC):
    name: str = "controller"
    label: str = "Controller"

    def __init__(self) -> None:
        self.aggressiveness = DEFAULT_AGGRESSIVENESS
        self.observer: AeroLagObserver | None = None
        self.warnings: list[str] = []

    def reset(self, params: WingParams, U: float, dt: float, delay: int = 1) -> None:
        self.params, self.U, self.dt, self.delay = params, U, dt, int(delay)
        self.observer = AeroLagObserver(params, U, dt)
        self.aggressiveness = DEFAULT_AGGRESSIVENESS
        self.u_max = np.asarray(params.u_max, dtype=float)
        m = n_inputs(params)
        # history[0] = last command sent (u_{k-1}), history[j] = u_{k-1-j}
        self.history: deque[np.ndarray] = deque([np.zeros(m) for _ in range(8)], maxlen=8)
        self.warnings = []

    def set_aggressiveness(self, a: float) -> None:
        self.aggressiveness = float(np.clip(a, 0.0, 1.0))

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        if self.observer is None:
            raise RuntimeError(f"{type(self).__name__} used before reset()")
        x_hat = self.observer.full_state(y)
        u = np.clip(self.compute(t, x_hat), -self.u_max, self.u_max)
        self.observer.update(y)
        self.history.appendleft(u)
        return u

    @abstractmethod
    def compute(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        """Return the (unsaturated) command from the estimated state (self.history holds past commands)."""


class OpenLoop(Controller):
    name = "open_loop"
    label = "Open loop (no control)"

    def compute(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        return np.zeros(n_inputs(self.params))
