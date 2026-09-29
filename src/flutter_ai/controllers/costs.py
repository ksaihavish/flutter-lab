"""
Quadratic cost weights shared by LQR/LQG design, neural-policy training and the reported
cost J, so every controller optimises the *same* objective (Bryson's rule normalisation):

    J = sum( x^T Q x + c(a) * u^T R u + (u_k - u_{k-1})^T W (u_k - u_{k-1}) )

The last term penalises fast command changes (actuator bandwidth / wear). It used to be in
the neural training only, which made the comparison with LQR/LQG unfair.
"""

from __future__ import annotations

import numpy as np

from ..model import n_states
from ..params import WingParams
from .base import effort_cost_multiplier


def reference_scales(p: WingParams) -> np.ndarray:
    """
    Acceptable magnitudes of [q, q']: one tenth of the structural limits for h and alpha.
    The flap angle is only lightly weighted (reference = twice the command limit) - it is
    an actuator, and its use is already charged through the effort term.
    """
    h_ref = p.h_limit / 10.0
    a_ref = np.radians(p.alpha_limit_deg) / 10.0
    q_ref = [h_ref, a_ref]
    w = [p.omega_h, p.omega_alpha]
    if p.n_dof == 3:
        beta_ref = 2.0 * (p.u_max[0] if p.actuation == "flap_servo" else np.radians(10.0))
        q_ref.append(beta_ref)
        w.append(p.omega_beta)
    q = np.array(q_ref)
    return np.concatenate([q, np.array(w) * q])


def state_weight(p: WingParams) -> np.ndarray:
    N = n_states(p)
    Q = np.zeros((N, N))
    k = 2 * p.n_dof
    Q[:k, :k] = np.diag(1.0 / reference_scales(p) ** 2)
    return Q


def input_weight(p: WingParams, aggressiveness: float) -> np.ndarray:
    u_max = np.asarray(p.u_max, dtype=float)
    return np.diag(1.0 / u_max**2) * effort_cost_multiplier(aggressiveness)


RATE_WEIGHT = 1.0  # weight on (du / u_max)^2 per step


def rate_weight(p: WingParams) -> np.ndarray:
    """W in the cost term (u_k - u_{k-1})^T W (u_k - u_{k-1})."""
    return np.diag(RATE_WEIGHT / np.asarray(p.u_max, dtype=float) ** 2)
