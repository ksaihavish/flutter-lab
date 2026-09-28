"""
Linear state-space model of the typical section (2-DOF or 3-DOF with a trailing-edge
flap) with unsteady Theodorsen aerodynamics in the time domain.

Generalised coordinates q = [h, alpha] or [h, alpha, beta]. Equations of motion
(per unit span, h down, alpha nose-up, beta trailing-edge down):

    M_s q'' + C_s q' + K_s q = f_aero + B_q u

    M_s = [[m,   S_a,               S_b              ],
           [S_a, I_a,               I_b + b(c-a) S_b ],
           [S_b, I_b + b(c-a) S_b,  I_b              ]]

Aerodynamics (Theodorsen 1935; T1..T14 in aero.theodorsen_T), moved to the left-hand
side as   M_a q'' + C_nc q' + K_nc q + e * w_eff:

  - non-circulatory (apparent mass) terms: M_a, C_nc, K_nc
  - circulatory terms through the 3/4-chord downwash
        Q = h' + U alpha + b(1/2 - a) alpha' + (U/pi) T10 beta + (b/2pi) T11 beta'
    passed through the Wagner function (R.T. Jones two-lag approximation):
        w_eff = phi(0) (Q + w_gust) + psi1 lam1 z1 + psi2 lam2 z2
        z_i'  = -lam_i z_i + Q + w_gust,    lam_i = eps_i U / b
  - e = [2 pi rho U b,  -2 pi rho U b^2 (a + 1/2),  rho U b^2 T12]

State  x = [q, q', z1, z2]          (2n + 2 states)
Input  u = [F_h, M_alpha]  (force_moment)   or   [beta_cmd] (flap_servo: k_beta * beta_cmd on the flap)
Gust   w_g  vertical gust velocity [m/s, positive up]

    x' = A x + B u + G w_g

Simplification (documented): gusts use the Wagner lag instead of the Kussner function.
Validated in tests/test_physics.py against textbook values, independent p-k analysis
with the exact Theodorsen function, and independent thin-airfoil theory for the flap.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import expm

from .aero import JONES_EPS, JONES_PSI, PHI0, theodorsen_T
from .params import WingParams


def n_states(p: WingParams) -> int:
    return 2 * p.n_dof + 2


def n_inputs(p: WingParams) -> int:
    return len(p.input_names)


def state_names(p: WingParams) -> tuple[str, ...]:
    return tuple(p.dof_names) + tuple(f"{n}_dot" for n in p.dof_names) + ("z1", "z2")


@dataclass(frozen=True)
class StateSpace:
    A: np.ndarray
    B: np.ndarray
    G: np.ndarray
    velocity: float


@dataclass(frozen=True)
class DiscreteStateSpace:
    Ad: np.ndarray
    Bd: np.ndarray
    Gd: np.ndarray
    dt: float
    velocity: float


def _T(p: WingParams) -> dict[int, float] | None:
    return None if p.flap_hinge is None else theodorsen_T(p.flap_hinge, p.a)


def structural_matrices(p: WingParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if p.n_dof == 2:
        Ms = np.array([[p.mass, p.S_alpha], [p.S_alpha, p.I_alpha]])
        return Ms, np.diag([p.c_h, p.c_alpha]), np.diag([p.k_h, p.k_alpha])
    coupling = p.I_beta + p.b * (p.flap_hinge - p.a) * p.S_beta
    Ms = np.array([
        [p.mass, p.S_alpha, p.S_beta],
        [p.S_alpha, p.I_alpha, coupling],
        [p.S_beta, coupling, p.I_beta],
    ])
    return Ms, np.diag([p.c_h, p.c_alpha, p.c_beta]), np.diag([p.k_h, p.k_alpha, p.k_beta])


def aero_force_vector(p: WingParams, U: float) -> np.ndarray:
    """
    e: maps the circulatory effective downwash w_eff to generalised forces on the LHS.

    Modified strip theory (Yates 1966): the circulatory lift uses the section lift-curve
    slope `cl_alpha` (thin airfoil: 2 pi) acting at the aerodynamic centre `x_ac`
    (thin airfoil: quarter chord, x_ac = -1/2). Defaults reproduce Theodorsen exactly.
    """
    kappa = p.cl_alpha / (2.0 * np.pi)
    e = [kappa * 2.0 * np.pi * p.rho * U * p.b, -kappa * 2.0 * np.pi * p.rho * U * p.b**2 * (p.a - p.x_ac)]
    if p.n_dof == 3:
        e.append(kappa * p.flap_span_fraction * p.rho * U * p.b**2 * _T(p)[12])
    return np.array(e)


def downwash_rows(p: WingParams, U: float) -> tuple[np.ndarray, np.ndarray]:
    """Q = dw_dq . q + dw_dqdot . q'   (+ w_gust)."""
    dw_dq = [0.0, U]
    dw_dqdot = [1.0, p.b * (0.5 - p.a)]
    if p.n_dof == 3:  # a partial-span flap contributes in proportion to the span it covers
        T, f = _T(p), p.flap_span_fraction
        dw_dq.append(f * U * T[10] / np.pi)
        dw_dqdot.append(f * p.b * T[11] / (2.0 * np.pi))
    return np.array(dw_dq), np.array(dw_dqdot)


def noncirculatory_matrices(p: WingParams, U: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apparent-mass, non-circulatory damping and stiffness, moved to the LHS."""
    b, a, rho = p.b, p.a, p.rho
    if p.n_dof == 2:
        k = np.pi * rho * b**2
        Ma = k * np.array([[1.0, -b * a], [-b * a, b**2 * (1.0 / 8.0 + a**2)]])
        Cnc = k * U * np.array([[0.0, 1.0], [0.0, b * (0.5 - a)]])
        return Ma, Cnc, np.zeros((2, 2))

    c, T = p.flap_hinge, _T(p)
    r = rho * b**2
    Ma = r * np.array([
        [np.pi, -np.pi * b * a, -b * T[1]],
        [-np.pi * b * a, np.pi * b**2 * (1.0 / 8.0 + a**2), -b**2 * (T[7] + (c - a) * T[1])],
        [-b * T[1], 2.0 * b**2 * T[13], -b**2 * T[3] / np.pi],
    ])
    Cnc = r * U * np.array([
        [0.0, np.pi, -T[4]],
        [0.0, np.pi * b * (0.5 - a), -b * (-T[1] + T[8] + (c - a) * T[4] - 0.5 * T[11])],
        [0.0, -b * (2.0 * T[9] + T[1] - (a - 0.5) * T[4]), -b * T[4] * T[11] / (2.0 * np.pi)],
    ])
    Knc = r * U**2 * np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, T[4] + T[10]],
        [0.0, 0.0, (T[5] - T[4] * T[10]) / np.pi],
    ])
    # Partial-span flap: flap rows/columns act over a fraction f of the span (the flap's own
    # beta-beta terms scale by f, not f^2). f = 1 leaves the 2D Theodorsen terms unchanged.
    # Note: the circulatory beta-beta hinge-moment term (via e x downwash) scales by f^2; exact
    # treatment needs separate lag states for flapped strips - negligible for an actuator-driven flap.
    f = p.flap_span_fraction
    if f != 1.0:
        scale = np.array([[1.0, 1.0, f], [1.0, 1.0, f], [f, f, f]])
        Ma, Cnc, Knc = Ma * scale, Cnc * scale, Knc * scale
    return Ma, Cnc, Knc


def input_matrix_q(p: WingParams) -> np.ndarray:
    """Generalised forces per unit input (n_dof x n_inputs)."""
    if p.actuation == "force_moment":
        Bq = np.zeros((p.n_dof, 2))
        Bq[0, 0] = Bq[1, 1] = 1.0
        return Bq
    Bq = np.zeros((p.n_dof, 1))
    Bq[2, 0] = p.k_beta  # servo spring pulls the flap towards beta_cmd
    return Bq


def continuous_state_space(p: WingParams, U: float) -> StateSpace:
    n = p.n_dof
    N = 2 * n + 2
    Ms, Cs, Ks = structural_matrices(p)
    Ma, Cnc, Knc = noncirculatory_matrices(p, U)
    e = aero_force_vector(p, U)
    dw_dq, dw_dqdot = downwash_rows(p, U)
    lam = np.array(JONES_EPS) * U / p.b
    psi = np.array(JONES_PSI)

    M = Ms + Ma
    C = Cs + Cnc + PHI0 * np.outer(e, dw_dqdot)
    K = Ks + Knc + PHI0 * np.outer(e, dw_dq)
    Ez = np.outer(e, psi * lam)
    Minv = np.linalg.inv(M)

    A = np.zeros((N, N))
    A[0:n, n:2 * n] = np.eye(n)
    A[n:2 * n, 0:n] = -Minv @ K
    A[n:2 * n, n:2 * n] = -Minv @ C
    A[n:2 * n, 2 * n:] = -Minv @ Ez
    for i in range(2):
        A[2 * n + i, 0:n] = dw_dq
        A[2 * n + i, n:2 * n] = dw_dqdot
        A[2 * n + i, 2 * n + i] = -lam[i]

    Bq = input_matrix_q(p)
    B = np.zeros((N, Bq.shape[1]))
    B[n:2 * n, :] = Minv @ Bq

    G = np.zeros((N, 1))
    G[n:2 * n, 0] = -Minv @ (PHI0 * e)
    G[2 * n:, 0] = 1.0

    return StateSpace(A=A, B=B, G=G, velocity=U)


def discretize(ss: StateSpace, dt: float) -> DiscreteStateSpace:
    """Exact zero-order-hold discretisation (inputs held constant over dt)."""
    n, m = ss.A.shape[0], ss.B.shape[1]
    aug = np.zeros((n + m + 1, n + m + 1))
    aug[:n, :n] = ss.A
    aug[:n, n:n + m] = ss.B
    aug[:n, n + m:] = ss.G
    phi = expm(aug * dt)
    return DiscreteStateSpace(Ad=phi[:n, :n], Bd=phi[:n, n:n + m], Gd=phi[:n, n + m:], dt=dt, velocity=ss.velocity)


@lru_cache(maxsize=256)
def _cached_discrete(p: WingParams, U: float, dt: float) -> DiscreteStateSpace:
    return discretize(continuous_state_space(p, U), dt)


def discrete_state_space(p: WingParams, U: float, dt: float) -> DiscreteStateSpace:
    return _cached_discrete(p, round(float(U), 9), round(float(dt), 12))


def structural_energy(p: WingParams, x: np.ndarray) -> np.ndarray:
    """Kinetic + strain energy of the structure [J/m]. Works on (N,) or (T, N) states."""
    n = p.n_dof
    Ms, _, Ks = structural_matrices(p)
    x = np.atleast_2d(x)
    q, qd = x[:, 0:n], x[:, n:2 * n]
    return 0.5 * np.einsum("ti,ij,tj->t", qd, Ms, qd) + 0.5 * np.einsum("ti,ij,tj->t", q, Ks, q)


def initial_state(p: WingParams, h0: float = 0.0, alpha0: float = 0.0, beta0: float = 0.0) -> np.ndarray:
    """Displacement initial condition, at rest; aero-lag states start at zero."""
    x = np.zeros(n_states(p))
    x[0], x[1] = h0, alpha0
    if p.n_dof == 3:
        x[2] = beta0
    return x
