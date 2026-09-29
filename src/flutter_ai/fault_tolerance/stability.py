"""
Dwell-time stability of supervisor switching (docs/research/fault-tolerant-plan.md §7).

    python -m flutter_ai ft-stability --preset uav_flap      # -> outputs/<preset>/ft_stability.json

The controller bank runs every controller on every step, so all modes share one state:
    xi = [x (plant), x_pred (LQG filter), z (neural's aero-lag observer), u_{k-1} (last applied command)].
Each mode is a linear map xi_{k+1} = A_mode xi_k (small-signal: saturation and rate limit inactive; nominal
wing; 1-sample loop delay; turbulence is an input and does not affect stability). For switching among
Schur-stable modes, Hespanha & Morse (1999): with V_i = xi' P_i xi, V_i(A_i xi) <= lam V_i(xi) and
V_j <= mu V_i, every switching signal whose average dwell time exceeds ln(mu) / (-ln lam) samples is
exponentially stable. We report that bound for the controllers that can be switched at each airspeed of the
schedule. Speed changes themselves (gain scheduling) are slow and outside this argument.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, solve_discrete_lyapunov

from ..controllers import LQGController, NeuralController
from ..controllers.neural import state_scales
from ..flutter import flutter_speed
from ..model import discrete_state_space
from ..params import WingParams
from ..paths import OUTPUTS_DIR

SPEEDS = (0.7, 0.8, 0.9, 1.0, 1.1)  # x nominal flutter speed: the autopilot's schedule
DT = 0.005


@functools.lru_cache(maxsize=512)
def _designs(p: WingParams, U: float, dt: float):
    """Controller-side matrices at airspeed U (cached: the margin estimator asks for the same speeds repeatedly)."""
    lqg = LQGController()
    lqg.reset(p, U, dt, 1)
    K = lqg._gain()  # u = -K [est (N+1), u_{k-1}]
    nn = NeuralController()
    nn.reset(p, U, dt, 1)
    pol = nn._policy
    if pol is None:
        raise RuntimeError("neural policy not loaded")
    Knn = pol.linearize(U / nn.flutter_speed, 0.5)  # u = -Knn [y, z, u_{k-1}]
    if nn.observer is None:
        raise RuntimeError("observer not initialised")
    return K, lqg._L, lqg._A, lqg._B, lqg._C, Knn, nn.observer


def mode_matrices(p: WingParams, U: float, dt: float = DT, plant: WingParams | None = None,
                  delay: int = 1) -> dict[str, np.ndarray]:
    """Closed-loop matrices on the shared bank state for the LQG-active and neural-active modes at airspeed U.
    The controllers are designed for p; `plant` (default p) is the wing they act on, e.g. an identified damaged wing.
    `delay`: the plant's loop delay in samples (>= 1). The controllers keep their 1-sample design (as in ft_simulate);
    for delay d the state carries the commands still in flight: xi = [x, x_pred, z, u_{k-1}, ..., u_{k-d}]."""
    return {k: v[0] for k, v in mode_systems(p, U, dt, plant, delay).items()}


def mode_systems(p: WingParams, U: float, dt: float = DT, plant: WingParams | None = None,
                 delay: int = 1) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(A, G, V) per active controller: xi_{k+1} = A xi_k + G w_k + V v_k, with w_k the vertical gust velocity [m/s]
    entering the plant through its gust column (as in ft_simulate) and v_k the measurement noise added to y = [q, q']
    (every controller and estimator in the bank sees the same noisy y). G and V drive the authority prediction
    (margin.command_statistics); stability only needs A."""
    if delay < 1:
        raise ValueError(f"delay must be >= 1 sample (the bank's design delay), got {delay}")
    n2 = 2 * p.n_dof
    sys = discrete_state_space(plant or p, U, dt)
    N, m = sys.Bd.shape
    K, L, Al, Bl, Cl, Knn, obs = _designs(p, U, dt)

    ix = slice(0, N)
    ip = slice(N, 2 * N + 1)
    iz = slice(2 * N + 1, 2 * N + 3)
    iu = slice(2 * N + 3, 2 * N + 3 + m)  # u_{k-1}: the last command, which every controller feeds back
    iud = slice(2 * N + 3 + (delay - 1) * m, 2 * N + 3 + delay * m)  # u_{k-delay}: the one reaching the servo now
    S = 2 * N + 3 + delay * m
    C = np.zeros((n2, N))
    C[:, :n2] = np.eye(n2)
    base = np.zeros((S, S))
    base[ix, ix] = sys.Ad
    base[ix, iud] = sys.Bd  # applied input = the command sent `delay` samples ago
    for j in range(1, delay):  # shift register of the commands in flight
        base[2 * N + 3 + j * m:2 * N + 3 + (j + 1) * m, 2 * N + 3 + (j - 1) * m:2 * N + 3 + j * m] = np.eye(m)
    # LQG filter: est = (I - L Cl) x_pred + L C x ;  x_pred' = Al est + Bl u_{k-1}
    E_pred = np.eye(N + 1) - L @ Cl
    E_x = L @ C
    base[ip, ip] = Al @ E_pred
    base[ip, ix] = Al @ E_x
    base[ip, iu] = Bl
    # neural observer: z' = decay z + gain (w_row . y)
    base[iz, iz] = np.diag(obs.decay)
    base[iz, ix] = np.outer(obs.gain, obs.w_row) @ C
    G = np.zeros((S, 1))
    G[ix, 0] = sys.Gd[:, 0]
    V = np.zeros((S, n2))
    V[ip] = Al @ L
    V[iz] = np.outer(obs.gain, obs.w_row)
    out = {}
    # LQG active: u_k = -K_est est - K_u u_{k-1}
    A = base.copy()
    A[iu, ip] = -K[:, :N + 1] @ E_pred
    A[iu, ix] = -K[:, :N + 1] @ E_x
    A[iu, iu] = -K[:, N + 1:]
    Vl = V.copy()
    Vl[iu] = -K[:, :N + 1] @ L
    out["lqg"] = (A, G, Vl)
    # neural active: u_k = -Knn [C x, z, u_{k-1}]
    A = base.copy()
    A[iu, ix] = -Knn[:, :n2] @ C
    A[iu, iz] = -Knn[:, n2:n2 + 2]
    A[iu, iu] = -Knn[:, n2 + 2:]
    Vn = V.copy()
    Vn[iu] = -Knn[:, :n2]
    out["neural"] = (A, G, Vn)
    return out


def bank_scales(p: WingParams, gust_scale: float = 0.5) -> np.ndarray:
    """Reference magnitudes of the bank state (plant, LQG filter incl. gust [m/s], observer, command). The bound is
    computed in these units: without scaling, mixing metres, radians and m/s made it absurdly conservative."""
    xs = state_scales(p)
    return np.concatenate([xs, xs, [gust_scale], xs[-2:], np.asarray(p.u_max, dtype=float)])


def _lyapunov(A: np.ndarray) -> tuple[np.ndarray, float]:
    """P with A'PA - P = -I, and lam = max V(A xi) / V(xi)."""
    P = solve_discrete_lyapunov(A.T, np.eye(A.shape[0]))
    P = 0.5 * (P + P.T)
    w, V = eigh(P)
    Ph = V @ np.diag(np.sqrt(w)) @ V.T
    Pmh = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    lam = float(np.linalg.norm(Ph @ A @ Pmh, 2) ** 2)
    return P, lam


def dwell_time_analysis(p: WingParams, speeds=SPEEDS, dt: float = DT) -> dict:
    Uf = flutter_speed(p)
    rows = []
    for r in speeds:
        s = bank_scales(p)
        mats = {k: (A * s[None, :]) / s[:, None] for k, A in mode_matrices(p, r * Uf, dt).items()}  # T A T^-1, T = diag(1/s)
        rho = {k: float(np.max(np.abs(np.linalg.eigvals(A)))) for k, A in mats.items()}
        row: dict = {"speed_ratio": r, "spectral_radius": rho, "all_stable": all(v < 1 for v in rho.values())}
        if row["all_stable"]:
            lyap = {k: _lyapunov(A) for k, A in mats.items()}
            lam = max(v[1] for v in lyap.values())
            mu = max(float(np.max(eigh(lyap[j][0], lyap[i][0], eigvals_only=True)))
                     for i in lyap for j in lyap if i != j)
            steps = math.log(mu) / -math.log(lam) if lam < 1 else math.inf
            row.update({"lambda": lam, "mu": mu, "dwell_steps": steps, "dwell_s": steps * dt})
        rows.append(row)
    worst = max((r_["dwell_s"] for r_ in rows if "dwell_s" in r_), default=math.inf)
    return {"preset": p.name, "flutter_speed": Uf, "speeds": rows, "max_dwell_s": worst,
            "note": "Small-signal, nominal wing, 1-sample delay; Hespanha & Morse (1999) average dwell-time bound."}


def run_stability(p: WingParams, out_dir: Path | None = None, log=print) -> dict:
    res = dwell_time_analysis(p)
    for r in res["speeds"]:
        rho = ", ".join(f"{k} {v:.4f}" for k, v in r["spectral_radius"].items())
        tail = f"  dwell bound {r['dwell_s']:.2f} s" if "dwell_s" in r else "  NOT all modes stable"
        log(f"  U = {r['speed_ratio']:.1f} U_F: spectral radius {rho}{tail}")
    log(f"Worst-case dwell-time bound over the schedule: {res['max_dwell_s']:.2f} s")
    out = Path(out_dir or OUTPUTS_DIR / p.name) / "ft_stability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    log(f"Saved {out}")
    return res
