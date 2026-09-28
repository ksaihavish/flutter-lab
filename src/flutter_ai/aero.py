"""
Unsteady thin-airfoil aerodynamics.

- Exact Theodorsen function C(k) via Hankel functions (frequency domain).
- R.T. Jones two-lag approximation of the Wagner function (time domain):
      phi(s) = 1 - psi1*exp(-eps1*s) - psi2*exp(-eps2*s),   s = U t / b
  whose Fourier transform is the matching approximation of C(k).
"""

from __future__ import annotations

import numpy as np
from scipy.special import hankel2

# R.T. Jones (1938) Wagner-function coefficients
JONES_PSI = (0.165, 0.335)
JONES_EPS = (0.0455, 0.3)
PHI0 = 1.0 - sum(JONES_PSI)  # phi(0) = 0.5


def theodorsen_exact(k: float | np.ndarray) -> complex | np.ndarray:
    """Theodorsen's function C(k) = H1(k) / (H1(k) + i H0(k)), Hankel 2nd kind."""
    k = np.asarray(k, dtype=float)
    k_safe = np.where(k < 1e-8, 1e-8, k)
    h1 = hankel2(1, k_safe)
    h0 = hankel2(0, k_safe)
    c = h1 / (h1 + 1j * h0)
    c = np.where(k < 1e-8, 1.0 + 0j, c)
    return c.item() if c.ndim == 0 else c


def theodorsen_jones(k: float | np.ndarray) -> complex | np.ndarray:
    """C(k) implied by the Jones Wagner approximation (used by the state-space model)."""
    k = np.asarray(k, dtype=float)
    c = PHI0 + sum(p * e / (1j * k + e) for p, e in zip(JONES_PSI, JONES_EPS))
    return c.item() if np.ndim(c) == 0 else c


def theodorsen_T(c: float, a: float) -> dict[int, float]:
    """
    Theodorsen's (1935, NACA Rep. 496) geometric flap coefficients T1..T14 for a
    trailing-edge flap hinged at c semichords aft of mid-chord, elastic axis at a.
    Validated in tests/test_physics.py (added-mass symmetry/positive-definiteness,
    steady loads vs independent thin-airfoil theory).
    """
    if not -1.0 < c < 1.0:
        raise ValueError("flap hinge c must lie strictly inside the chord (-1 < c < 1)")
    s = np.sqrt(1.0 - c * c)
    ac = np.arccos(c)
    T = {}
    T[1] = -s * (2.0 + c * c) / 3.0 + c * ac
    T[2] = c * (1.0 - c * c) - s * (1.0 + c * c) * ac + c * ac * ac
    T[3] = -(1.0 / 8.0 + c * c) * ac * ac + 0.25 * c * s * ac * (7.0 + 2.0 * c * c) - (1.0 - c * c) * (5.0 * c * c + 4.0) / 8.0
    T[4] = -ac + c * s
    T[5] = -(1.0 - c * c) - ac * ac + 2.0 * c * s * ac
    T[6] = T[2]
    T[7] = -(1.0 / 8.0 + c * c) * ac + c * s * (7.0 + 2.0 * c * c) / 8.0
    T[8] = -s * (2.0 * c * c + 1.0) / 3.0 + c * ac
    T[9] = 0.5 * (s**3 / 3.0 + a * T[4])
    T[10] = s + ac
    T[11] = ac * (1.0 - 2.0 * c) + s * (2.0 - c)
    T[12] = s * (2.0 + c) - ac * (2.0 * c + 1.0)
    T[13] = 0.5 * (-T[7] - (c - a) * T[1])
    T[14] = 1.0 / 16.0 + 0.5 * a * c
    return T
