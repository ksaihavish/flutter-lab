"""
Flutter / divergence analysis (any number of structural DOF).

Primary method: eigenvalues of the time-domain state-space model (Jones/Wagner),
which is exact for that model - no reduced-frequency iteration needed.

Independent check: classical p-k method with the *exact* Theodorsen function.
The two agree to within ~1-2 % (the Jones approximation error).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, linear_sum_assignment

from .aero import theodorsen_exact
from .model import (
    aero_force_vector,
    continuous_state_space,
    downwash_rows,
    noncirculatory_matrices,
    structural_matrices,
)
from .params import WingParams


@dataclass
class FlutterResult:
    flutter_speed: float | None
    flutter_frequency_hz: float | None
    divergence_speed: float | None
    velocities: list[float] = field(default_factory=list)
    damping: list[list[float]] = field(default_factory=list)  # [mode][velocity]
    frequency_hz: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        def clean(v):
            return None if v is None or not np.isfinite(v) else float(v)

        return {
            "flutter_speed": clean(self.flutter_speed),
            "flutter_frequency_hz": clean(self.flutter_frequency_hz),
            "divergence_speed": clean(self.divergence_speed),
            "velocities": [float(v) for v in self.velocities],
            "damping": [[clean(z) for z in mode] for mode in self.damping],
            "frequency_hz": [[clean(f) for f in mode] for mode in self.frequency_hz],
        }


def _oscillatory_modes(eigs: np.ndarray, min_imag: float = 1e-6) -> np.ndarray:
    osc = eigs[np.imag(eigs) > min_imag]
    return osc[np.argsort(np.imag(osc))]


def max_oscillatory_real(p: WingParams, U: float) -> float:
    osc = _oscillatory_modes(np.linalg.eigvals(continuous_state_space(p, U).A))
    return float(np.max(np.real(osc))) if len(osc) else -np.inf


def max_real_nonoscillatory(p: WingParams, U: float) -> float:
    eigs = np.linalg.eigvals(continuous_state_space(p, U).A)
    real = eigs[np.abs(np.imag(eigs)) <= 1e-6]
    return float(np.max(np.real(real))) if len(real) else -np.inf


def _first_crossing(f, velocities: np.ndarray) -> float | None:
    vals = np.array([f(U) for U in velocities])
    idx = np.where(vals > 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return float(velocities[0])
    return float(brentq(f, velocities[i - 1], velocities[i], xtol=1e-6))


def default_velocities(p: WingParams, n: int = 300) -> np.ndarray:
    u_ref = p.b * p.omega_alpha  # reference speed b * omega_alpha
    return np.linspace(0.05 * u_ref, 6.0 * u_ref, n)


def flutter_analysis(p: WingParams, velocities: np.ndarray | None = None) -> FlutterResult:
    """V-g / V-f sweep plus refined flutter and divergence speeds."""
    velocities = default_velocities(p) if velocities is None else np.asarray(velocities, dtype=float)
    n = p.n_dof
    damping: list[list[float]] = [[] for _ in range(n)]
    freq: list[list[float]] = [[] for _ in range(n)]
    prev: np.ndarray | None = None  # eigenvalue of each tracked mode at the previous airspeed
    for U in velocities:
        eigs = np.linalg.eigvals(continuous_state_space(p, U).A)
        cand = eigs[np.imag(eigs) >= -1e-9]  # upper half plane, incl. real roots
        if prev is None:  # identify the structural modes at the lowest speed: the oscillatory roots
            osc = _oscillatory_modes(eigs)
            ordered = osc[:n] if len(osc) >= n else np.pad(osc, (0, n - len(osc)))
        else:  # follow each mode by continuity (no label swaps where frequencies cross)
            rows, cols = linear_sum_assignment(np.abs(prev[:, None] - cand[None, :]))
            ordered = np.empty(n, dtype=complex)
            ordered[rows] = cand[cols]
        prev = ordered
        for m in range(n):
            lam = ordered[m]
            if abs(np.imag(lam)) > 1e-6:
                damping[m].append(float(-np.real(lam) / abs(lam)))
                freq[m].append(float(np.imag(lam) / (2 * np.pi)))
            else:  # overdamped (non-oscillatory) - not shown on a V-g plot
                damping[m].append(float("nan"))
                freq[m].append(float("nan"))

    Uf = _first_crossing(lambda U: max_oscillatory_real(p, U), velocities)
    Ud = _first_crossing(lambda U: max_real_nonoscillatory(p, U), velocities)
    ff = None
    if Uf is not None:
        osc = _oscillatory_modes(np.linalg.eigvals(continuous_state_space(p, Uf).A))
        ff = float(np.imag(osc[np.argmax(np.real(osc))]) / (2 * np.pi))
    return FlutterResult(Uf, ff, Ud, list(velocities), damping, freq)


def flutter_speed(p: WingParams) -> float:
    result = flutter_analysis(p)
    if result.flutter_speed is None:
        raise ValueError(f"No flutter found for preset '{p.name}' in the sweep range.")
    return result.flutter_speed


# --------------------------------------------------------------------------- #
# Independent validation: p-k method with the exact Theodorsen function
# --------------------------------------------------------------------------- #
def _pk_eigs(p: WingParams, U: float, k: float) -> np.ndarray:
    n = p.n_dof
    Ms, Cs, Ks = structural_matrices(p)
    Ma, Cnc, Knc = noncirculatory_matrices(p, U)
    e = aero_force_vector(p, U)
    dw_dq, dw_dqdot = downwash_rows(p, U)
    Ck = theodorsen_exact(k)
    M = Ms + Ma
    C = Cs + Cnc + Ck * np.outer(e, dw_dqdot)
    K = Ks + Knc + Ck * np.outer(e, dw_dq)
    Minv = np.linalg.inv(M)
    A = np.zeros((2 * n, 2 * n), dtype=complex)
    A[0:n, n:] = np.eye(n)
    A[n:, 0:n] = -Minv @ K
    A[n:, n:] = -Minv @ C
    return np.linalg.eigvals(A)


def pk_damping(p: WingParams, velocities: np.ndarray, tol: float = 1e-8, max_iter: int = 200) -> np.ndarray:
    """Real part of p for each structural mode, tracked with the p-k iteration."""
    guesses = [p.omega_h, p.omega_alpha] + ([p.omega_beta] if p.n_dof == 3 else [])
    out = np.zeros((p.n_dof, len(velocities)))
    for j, U in enumerate(velocities):
        for m in range(p.n_dof):
            omega, root = guesses[m], None
            for _ in range(max_iter):
                eigs = _pk_eigs(p, U, max(omega * p.b / U, 1e-6))
                upper = eigs[np.imag(eigs) > 0]
                if len(upper) == 0:
                    break
                root = upper[np.argmin(np.abs(np.imag(upper) - omega))]
                if abs(np.imag(root) - omega) < tol * max(1.0, omega):
                    break
                omega = float(np.imag(root))
            guesses[m] = float(np.imag(root)) if root is not None else omega
            out[m, j] = float(np.real(root)) if root is not None else np.nan
    return out


def pk_flutter_speed(p: WingParams, velocities: np.ndarray | None = None) -> float | None:
    velocities = default_velocities(p, 400) if velocities is None else velocities
    re = pk_damping(p, velocities)
    unstable = np.where(np.nanmax(re, axis=0) > 0)[0]
    if len(unstable) == 0:
        return None
    i = unstable[0]
    if i == 0:
        return float(velocities[0])
    v0, v1 = velocities[i - 1], velocities[i]
    r0, r1 = np.nanmax(re[:, i - 1]), np.nanmax(re[:, i])
    return float(v0 + (v1 - v0) * (-r0) / (r1 - r0))
