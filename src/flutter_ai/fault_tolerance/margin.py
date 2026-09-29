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

v2 (`estimate_v2`, used by FTPhysicsRuleSupervisorV2; `estimate` above is unchanged) adds what v1 cannot see:
  - control authority: flap damage kills the wing through saturation, not linear instability. For every mode, the
    stationary RMS of the flap command and of its rate at the mode's airspeed under the estimated turbulence and
    sensor noise (discrete Lyapunov equation of the closed loop with the simulator's AR(1) gust), relative to the
    flap's limits;
  - the turbulence intensity and the sensor-noise level, estimated on board from the residual's gust component and
    the measurements' high-frequency content, calibrated on the take-off window;
  - the measured loop delay, in the fit and in every closed-loop prediction (stability.mode_systems);
  - robust margins: worst case over the identification's confidence region (covariance inflated for the correlated,
    gust-driven residuals), in the spirit of NASA's Flutterometer (see estimate_v2).
Still linear: a saturating or linearly unstable loop is predicted as such, not what the nonlinear flight then does.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import least_squares

from ..flutter import flutter_analysis
from ..params import WingParams
from .diagnostics import one_step_residual
from .simulate import FT_MODES, SPEED_STEP
from .stability import mode_matrices, mode_systems

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


def applied_commands(p: WingParams, us: np.ndarray, k: int, dt: float, delay: int | np.ndarray = 1) -> np.ndarray:
    """Commands 0..k-1 as the flap servo receives them on the healthy aircraft: shifted by the nominal loop delay and
    passed through the known slew-rate limit (as the LQG filter's input model). Without the limit, the fit blames the
    unmodelled rate saturation on the structure (plunge stiffness drifted by up to +40 % in trials).
    `delay` may be the per-sample loop delay the flight computer measured (length >= k; v2): sample j then receives
    command j - delay[j]. A single value for a window that spans a change of delay biased the fit."""
    if isinstance(delay, np.ndarray):
        idx = np.arange(k) - delay[:k].astype(int)
        out = np.where((idx >= 0)[:, None], us[np.clip(idx, 0, None)], 0.0)
    else:
        out = np.vstack([np.zeros((delay, us.shape[1])), us[:k - delay]]) if delay else us[:k].copy()
    if p.u_rate_max is not None:
        step, act = np.asarray(p.u_rate_max) * dt, np.zeros(us.shape[1])
        for j in range(len(out)):
            act = act + np.clip(out[j] - act, -step, step)
            out[j] = act
    return out


def _least_squares(p: WingParams, U: float, dt: float, y: np.ndarray, u_applied: np.ndarray,
                   names: tuple[str, ...], start: np.ndarray | None, bounds: tuple[float, float] = BOUNDS):
    """The prediction-error fit shared by v1 and v2 (identical numerics for the same bounds)."""
    nominal = np.array([getattr(p, n) for n in names], dtype=float)

    def residual(k: np.ndarray) -> np.ndarray:
        q = p.with_overrides(**{n: float(v) for n, v in zip(names, k * nominal)})
        r = one_step_residual(q, U, dt, y, u_applied)
        return r.ravel() / np.sqrt(r.size)

    k0 = np.ones(len(names)) if start is None else np.clip(start, *bounds)
    return least_squares(residual, k0, bounds=bounds, x_scale=0.1, diff_step=1e-4, method="trf")


def fit_window(p: WingParams, U: float, dt: float, y: np.ndarray, u_applied: np.ndarray,
               names: tuple[str, ...] = PARAMS, start: np.ndarray | None = None) -> Fit:
    """Fit `names` (as factors of the nominal values) to one measurement window."""
    sol = _least_squares(p, U, dt, y, u_applied, names, start)
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


# --------------------------------------------------------------------------------------------------------------------
# v2: control authority, measured loop delay, turbulence and sensor-noise estimates, robust (worst-case) margins
# --------------------------------------------------------------------------------------------------------------------
# The raw flap factor of the healthy wing fits at ~0.75 (the unmeasured gust biases it; calibration removes that), so
# v1's floor of 0.3 capped flap damage at -60 %; v2 fits down to 0.05 (the fault model's own floor).
V2_BOUNDS = (0.05, 2.0)
DESIGN_TURBULENCE_RMS = 0.25  # [m/s] turbulence of the healthy flight (ft_simulate default), assumed during take-off
GUST_CORRELATION_S = 0.05  # [s] AR(1) correlation time of the turbulence (ft_simulate and the LQG design)
SIGMA_LEVEL = 3.0  # authority ratio = 3 sigma of the command (or of its rate) / the flap's limit
CONFIDENCE = 0.95  # robust margins: joint confidence region of the identified factors
RING_POINTS = 12  # points on the boundary of that region (2 parameters: 30 deg apart)
HAC_LAGS = 50  # [samples] residual autocovariance lags in the sandwich covariance (~5 gust correlation times)
NOISE_PROBE_S = 20.0  # [s] fixed synthetic white-noise record that measures the residual's response to sensor noise
NOISE_DIFF_ORDER = 4  # the 4th difference of h, alpha: the motion's share is ~1e-3 of the 2nd difference's


@dataclass
class FitV2(Fit):
    cov: np.ndarray  # covariance of theta: Gauss-Newton, inflated to cover correlated residuals (see fit_window_v2)
    inflation: float  # variance factor applied to the Gauss-Newton covariance (>= 1; nan if singular)
    gust_rms: float  # RMS of the residual's gust component (not calibrated)
    noise_level: float  # noise-equivalent std of the measured h, alpha (hf_noise_level; fraction of reference scales)
    noise_gain: float  # variance of the gust component per unit sensor-noise variance (std / reference scale)^2


@dataclass
class MarginEstimateV2(MarginEstimate):
    """v1 fields (params_std from the inflated covariance; flutter_speed_std from the confidence ring; mode_spectral_
    radius as in v1 but with the measured loop delay) plus the v2 predictions. The confidence region is sampled by the
    point estimate and RING_POINTS wings on its boundary.

    Admissibility of a mode. Its *transit* is the path from the current airspeed to its target without the current
    airspeed itself (every mode starts there, so it cannot tell apart modes that share a controller; a mode that holds
    speed is judged at the airspeed it holds). A mode is
      stable on a wing      if its closed loop has spectral radius < 1 at every transit airspeed where that wing needs
                            control (at or above the wing's open-loop flutter speed). Where the open-loop wing is
                            stable, a saturating or linearly unstable controller is bounded by the flap limits: it
                            chatters, the wing does not flutter;
      authority_ok          if the identified wing does not need control at the target, or the authority ratio there
                            is < 1 (point estimate, as the brief for v2 asks: the flap factor is weakly identified in
                            healthy flight, so a worst case over the region would flag saturation on healthy wings);
      admissible            stable on every sampled wing (robust) and authority_ok;
      admissible_point      stable on the point estimate and authority_ok.
    `modes[m]` also holds: target_airspeed [m/s]; spectral_radius_worst (whole path, every sampled wing: v1's
    definition made robust); control_needed and spectral_radius_needed (point estimate, transit airspeeds needing
    control; None if none do); command_rms [rad] and command_rate_rms [rad/s] (stationary, at the target, point
    estimate with the estimated turbulence and noise and the measured delay; None if the loop is unstable there);
    authority_ratio = SIGMA_LEVEL x max(command RMS / u_max, rate RMS / u_rate_max), >= 1: saturation expected;
    stable_share (share of the sampled wings on which the mode is stable)."""

    version: int = 2
    delay_samples: int = 1  # measured loop delay used in the fit and in every closed-loop prediction
    turbulence_rms: float | None = None  # estimated vertical gust RMS [m/s]
    noise_std: float | None = None  # estimated sensor noise std, fraction of the reference scales
    params_cov_inflation: float | None = None
    confidence: float = CONFIDENCE
    robust: bool = False  # False: no covariance (singular fit); the worst-case fields then equal the point estimate
    flutter_speed_worst: float | None = None  # lowest flutter speed over the confidence region [m/s]
    margin_worst: float | None = None
    modes: dict[str, dict] = field(default_factory=dict)


def flutter_speed_fast(p: WingParams) -> float | None:
    """flutter_analysis(p).flutter_speed without the V-g sweep and mode tracking: the first oscillatory crossing is
    bracketed on the 80-point grid flutter_analysis uses to choose its range, then refined by brentq. The robust margin
    needs a dozen flutter speeds per estimate; tests check it against flutter_analysis."""
    from scipy.optimize import brentq

    from ..flutter import max_oscillatory_real
    from ..model import continuous_state_space

    u_ref = p.b * p.omega_alpha
    for top in (6.0, 30.0, 150.0):
        v = np.linspace(0.05 * u_ref, top * u_ref, 80)
        osc, div = [], []
        for U in v:
            eigs = np.linalg.eigvals(continuous_state_space(p, U).A)
            o, r = eigs[np.imag(eigs) > 1e-6], eigs[np.abs(np.imag(eigs)) <= 1e-6]
            osc.append(float(np.max(np.real(o))) if len(o) else -np.inf)
            div.append(float(np.max(np.real(r))) if len(r) else -np.inf)
        if max(osc) > 0 or max(div) > 0:
            break
    idx = np.where(np.array(osc) > 0)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(v[0])
    return float(brentq(lambda U: max_oscillatory_real(p, U), v[i - 1], v[i], xtol=1e-6))


def _gust_component(q: WingParams, U: float, dt: float, r: np.ndarray) -> np.ndarray:
    """Per-sample gust velocity [m/s] that best explains the one-step residual r (rate channels, reference-scaled),
    fitted jointly with an actuator term (the model's input direction), so that a flap which does not follow its
    modelled command (servo fault, unmodelled saturation) is not read as turbulence."""
    from ..controllers.costs import reference_scales
    from ..model import discrete_state_space

    n = q.n_dof
    sys = discrete_state_space(q, U, dt)
    ref = reference_scales(q)[n:2 * n]
    M = np.column_stack([sys.Gd[n:2 * n, 0] / ref, sys.Bd[n:2 * n, :] / ref[:, None]])
    return np.linalg.lstsq(M, r.T, rcond=None)[0][0]


def fit_window_v2(p: WingParams, U: float, dt: float, y: np.ndarray, u_applied: np.ndarray,
                  names: tuple[str, ...] = PARAMS, start: np.ndarray | None = None) -> FitV2:
    """v1's prediction-error fit (wider bounds) plus what v2 needs from the same window:
    - a covariance that is honest about correlated residuals. The gust (correlation time 0.05 s = 10 samples) makes
      the residuals coloured, so the Gauss-Newton covariance, which assumes white residuals, is optimistic. The
      sandwich covariance H^-1 S H^-1 (S: Bartlett-weighted autocovariances of the per-sample score up to HAC_LAGS)
      is valid for correlated residuals; the Gauss-Newton ellipse is scaled up by the smallest factor that contains
      it (never shrunk: with few effective samples the sandwich estimate is itself noisy);
    - the residual's gust component and the measurements' high-frequency content (turbulence and noise estimates)."""
    from ..controllers.costs import reference_scales

    sol = _least_squares(p, U, dt, y, u_applied, names, start, V2_BOUNDS)
    n, size = p.n_dof, sol.fun.size
    r = (sol.fun * np.sqrt(size)).reshape(-1, n)  # residual per sample and rate channel (the fit scales by 1/sqrt(N))
    J = (sol.jac * np.sqrt(size)).reshape(len(r), n, -1)
    try:
        Hinv = np.linalg.inv(np.einsum("tci,tcj->ij", J, J))
        white = float(np.sum(r**2)) / max(1, size - len(names)) * Hinv
        s = np.einsum("tci,tc->ti", J, r)  # per-sample score
        S = s.T @ s
        for lag in range(1, min(HAC_LAGS, len(s) - 1) + 1):
            G = s[lag:].T @ s[:-lag]
            S += (1.0 - lag / (HAC_LAGS + 1)) * (G + G.T)
        inflation = max(1.0, float(np.max(np.real(np.linalg.eigvals(np.linalg.solve(white, Hinv @ S @ Hinv))))))
        cov = inflation * white
    except np.linalg.LinAlgError:
        cov, inflation = np.full((len(names), len(names)), np.nan), float("nan")
    q = p.with_overrides(**{nm: float(v * getattr(p, nm)) for nm, v in zip(names, sol.x)})
    gust = _gust_component(q, U, dt, r)
    probe = np.random.default_rng(0).standard_normal((int(NOISE_PROBE_S / dt), 2 * n)) * reference_scales(q)[:2 * n]
    noise_gain = float(np.mean(_gust_component(q, U, dt, one_step_residual(q, U, dt, probe,
                                                                              np.zeros((len(probe), len(q.u_max)))))**2))
    return FitV2(theta=sol.x, std=np.sqrt(np.clip(np.diag(cov), 0.0, None)), cost=float(sol.cost),
                 converged=bool(sol.success), cov=cov, inflation=inflation,
                 gust_rms=float(np.sqrt(np.mean(gust**2))), noise_level=hf_noise_level(p, y), noise_gain=noise_gain)


def hf_noise_level(p: WingParams, y: np.ndarray, order: int = NOISE_DIFF_ORDER) -> float:
    """Noise-equivalent std of the measured h and alpha [fraction of the reference scales]: the mean square of their
    `order`-th difference divided by C(2 order, order), which is exactly s^2 for white noise of std s. The structure's
    own motion (~10 Hz at 200 Hz sampling) is almost removed by the 4th difference; the diagnostics' 2nd difference
    left a motion share that wandered by +/-20 % between healthy windows and read as sensor noise of ~1 %."""
    from math import comb

    from ..controllers.costs import reference_scales

    d = np.diff(y[:, :2], n=order, axis=0) / reference_scales(p)[:2]
    return float(np.sqrt(np.mean(d**2) / comb(2 * order, order))) if len(d) else 0.0


def disturbance_estimate(fit: FitV2, base: FitV2, base_turbulence_rms: float = DESIGN_TURBULENCE_RMS
                         ) -> tuple[float, float]:
    """(turbulence RMS [m/s], sensor-noise std [fraction of the reference scales]) from two statistics of the window,
    each relative to the take-off window (design turbulence, no sensor noise):
        gust component of the residual            g^2 / g_b^2 = tau + noise_gain / g_b^2 * s^2
        noise-equivalent level of h, alpha        N^2 / N_b^2 = tau + s^2 / N_b^2
    with tau = (sigma_w / sigma_b)^2; the motion's own share of both statistics is assumed to scale with the
    turbulence. Sensor noise alone would otherwise read as a turbulence rise (the residual cannot tell them apart)."""
    rg = (fit.gust_rms / base.gust_rms) ** 2
    rn = (fit.noise_level / base.noise_level) ** 2
    bg, bn = fit.noise_gain / base.gust_rms**2, 1.0 / base.noise_level**2
    s2 = (rn - rg) / (bn - bg) if bn > bg else 0.0  # bn <= bg: the two statistics cannot separate noise
    if s2 <= 0.0:
        return base_turbulence_rms * float(np.sqrt(rg)), 0.0
    return base_turbulence_rms * float(np.sqrt(max(0.0, rg - bg * s2))), float(np.sqrt(s2))


def command_statistics(p: WingParams, system: tuple[np.ndarray, np.ndarray, np.ndarray], dt: float,
                       turbulence_rms: float, noise_std: float = 0.0,
                       correlation_time: float = GUST_CORRELATION_S) -> tuple[np.ndarray, np.ndarray] | None:
    """Stationary RMS of the flap command u_k [rad] and of its rate (u_k - u_{k-1}) / dt [rad/s], per input, for a
    closed loop (A, G, V) from stability.mode_systems driven by AR(1) turbulence (RMS turbulence_rms [m/s], the
    simulator's model) and white sensor noise (std noise_std x reference scales), from the discrete Lyapunov equation
    of the loop augmented with the gust state. None if the loop is unstable (no stationary statistics)."""
    from scipy.linalg import solve_discrete_lyapunov

    from ..controllers.costs import reference_scales

    A, G, V = system
    if np.max(np.abs(np.linalg.eigvals(A))) >= 1.0:
        return None
    S, m, n2 = A.shape[0], len(p.u_max), 2 * p.n_dof
    iu = slice(2 * (2 * p.n_dof + 2) + 3, 2 * (2 * p.n_dof + 2) + 3 + m)  # u_{k-1} slot of the bank state
    phi = float(np.exp(-dt / correlation_time))
    Aa = np.zeros((S + 1, S + 1))
    Aa[:S, :S], Aa[:S, S], Aa[S, S] = A, G[:, 0], phi
    Bw = np.zeros(S + 1)
    Bw[S] = turbulence_rms * np.sqrt(1.0 - phi**2)
    Bv = np.zeros((S + 1, n2))
    Bv[:S] = V * (noise_std * reference_scales(p)[:n2])[None, :]
    P = solve_discrete_lyapunov(Aa, np.outer(Bw, Bw) + Bv @ Bv.T)
    D = Aa[iu].copy()  # u_k - u_{k-1} = (A[iu] - E) xi_k + V[iu] v_k
    D[:, iu] -= np.eye(m)
    u_var = np.diag(P[iu, iu])
    du_var = np.einsum("ij,jk,ik->i", D, P, D) + np.sum(Bv[iu] ** 2, axis=1)
    return np.sqrt(u_var), np.sqrt(du_var) / dt


def _authority_ratio(p: WingParams, stats: tuple[np.ndarray, np.ndarray] | None) -> float | None:
    if stats is None:
        return None
    ratio = np.max(stats[0] / np.asarray(p.u_max, dtype=float))
    if p.u_rate_max is not None:
        ratio = max(ratio, np.max(stats[1] / np.asarray(p.u_rate_max, dtype=float)))
    return float(SIGMA_LEVEL * ratio)


def confidence_ring(k: np.ndarray, cov: np.ndarray, confidence: float = CONFIDENCE,
                    points: int = RING_POINTS) -> list[np.ndarray]:
    """Points on the boundary of the joint confidence ellipsoid {dk' cov^-1 dk <= chi2_n(confidence)} (a smooth,
    near-monotone function of k takes its extremes there; 2 parameters: a ring, otherwise +/- the principal axes),
    intersected with the fit's physical range. Empty if the covariance is not available."""
    from scipy.stats import chi2

    if not np.all(np.isfinite(cov)):
        return []
    r = float(np.sqrt(chi2.ppf(confidence, len(k))))
    w, vec = np.linalg.eigh(0.5 * (cov + cov.T))
    L = vec @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
    if len(k) == 2:
        dirs = [np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0.0, 2 * np.pi, points, endpoint=False)]
    else:
        dirs = [s * e for e in np.eye(len(k)) for s in (1.0, -1.0)]
    return [np.clip(k + r * L @ d, *V2_BOUNDS) for d in dirs]


def _mode_path(U: float, target: float) -> np.ndarray:
    return np.unique(np.round(np.linspace(U, target, PATH_POINTS) / SPEED_STEP) * SPEED_STEP)


def estimate_v2(p: WingParams, flutter_speed_nominal: float, t: float, U: float, dt: float, y: np.ndarray,
                u_applied: np.ndarray, baseline: Fit | None, *, delay: int = 1, names: tuple[str, ...] = PARAMS,
                modes: tuple[str, ...] = tuple(FT_MODES), cruise: float | None = None,
                base_turbulence_rms: float = DESIGN_TURBULENCE_RMS,
                confidence: float = CONFIDENCE) -> tuple[MarginEstimateV2, FitV2]:
    """
    One v2 estimate from a measurement window. `u_applied` must use the measured loop delay (applied_commands with the
    per-sample delay history); `delay` is the current measured delay [samples], used in every closed-loop prediction.
    `baseline`: fit_window_v2 on the healthy take-off window (None: no calibration, design turbulence, no noise).

    Robust margins, in the spirit of NASA's Flutterometer (Lind & Brenner, J. Aircraft 37(6):1105-1112, 2000,
    doi:10.2514/2.2719; NASA/TM-1997-207056, NTRS 19980017615): the flutter speed and each mode's spectral radius are
    reported as the worst case over the set of wings consistent with the flight data, here the identification's joint
    `confidence` region (inflated covariance, including the take-off calibration's own uncertainty), evaluated on its
    boundary. This is NOT mu-analysis: the uncertainty is parametric (two physical factors), the worst case is found by
    sampling, not by a structured-singular-value bound, and unmodelled dynamics are not covered.
    """
    if baseline is not None and not isinstance(baseline, FitV2):
        raise TypeError("estimate_v2 needs a take-off baseline from fit_window_v2 (it carries the disturbance levels)")
    fit = fit_window_v2(p, U, dt, y, u_applied, names, start=None if baseline is None else baseline.theta)
    if baseline is not None:  # ratio of two independent estimates: the take-off fit's uncertainty adds (delta method)
        k = fit.theta / baseline.theta
        D1, D2 = np.diag(1.0 / baseline.theta), np.diag(fit.theta / baseline.theta**2)
        cov = D1 @ fit.cov @ D1 + D2 @ baseline.cov @ D2
        turb, noise = disturbance_estimate(fit, baseline, base_turbulence_rms)
    else:
        k, cov, turb, noise = fit.theta, fit.cov, base_turbulence_rms, 0.0

    def wing(kk: np.ndarray) -> WingParams:
        return p.with_overrides(**{n: float(v * getattr(p, n)) for n, v in zip(names, kk)})

    from scipy.stats import chi2

    p_hat = wing(k)
    ring = confidence_ring(k, cov, confidence)
    wings = [p_hat] + [wing(kk) for kk in ring]
    Ufs = [flutter_speed_fast(q) for q in wings]  # None: no flutter in the analysis range
    Uf = Ufs[0]
    found = [u for u in Ufs if u is not None]
    Uf_worst = min(found) if found else None
    # linearised 1-sigma: over the ring, a linear U_F(k) spans 2 r sqrt(g' cov g) (v1 used finite differences)
    on_ring = [u for u in Ufs[1:] if u is not None]
    Uf_std = (None if Uf is None or not ring or len(on_ring) < len(ring) else
              float((max(on_ring) - min(on_ring)) / (2.0 * np.sqrt(chi2.ppf(confidence, len(k))))))

    cruise = cruise if cruise is not None else U
    targets = {m: float(np.round((cruise if FT_MODES[m]["speed_ratio"] is None else
                                  FT_MODES[m]["speed_ratio"] * flutter_speed_nominal) / SPEED_STEP) * SPEED_STEP)
               for m in modes}
    paths = {m: _mode_path(U, targets[m]) for m in modes}
    speeds = sorted({float(s) for path in paths.values() for s in path})
    col = {s: j for j, s in enumerate(speeds)}
    radius: dict[str, np.ndarray] = {}  # controller -> (wing, airspeed)
    for i, q in enumerate(wings):
        for j, Us in enumerate(speeds):
            for c, (A, _, _) in mode_systems(p, Us, dt, plant=q, delay=delay).items():
                radius.setdefault(c, np.zeros((len(wings), len(speeds))))[i, j] = np.max(np.abs(np.linalg.eigvals(A)))
    point_radius, per_mode = {}, {}
    for m in modes:
        c = FT_MODES[m]["controller"]
        full = [col[float(s)] for s in paths[m]]
        point_radius[m] = float(np.max(radius[c][0, full]))  # v1's definition (current airspeed included)
        # every mode starts at the current airspeed, so it cannot tell apart modes that share a controller: a mode that
        # leaves it is judged by where it goes, a mode that holds it by where it stays
        transit = [s for s in paths[m] if len(paths[m]) == 1 or abs(s - U) > 1e-9]
        stable = []
        for i, Uf_i in enumerate(Ufs):
            need = [col[float(s)] for s in transit if Uf_i is not None and s >= Uf_i]
            stable.append(not need or float(np.max(radius[c][i, need])) < 1.0)
        need0 = [col[float(s)] for s in transit if Uf is not None and s >= Uf]
        needed = Uf is not None and targets[m] >= Uf
        stats = command_statistics(p, mode_systems(p, targets[m], dt, plant=p_hat, delay=delay)[c], dt, turb, noise)
        auth = _authority_ratio(p, stats)
        authority_ok = not needed or (auth is not None and auth < 1.0)
        per_mode[m] = {
            "target_airspeed": targets[m],
            "spectral_radius_worst": float(np.max(radius[c][:, full])),
            "control_needed": bool(needed),
            "spectral_radius_needed": float(np.max(radius[c][0, need0])) if need0 else None,
            "command_rms": None if stats is None else float(np.max(stats[0])),
            "command_rate_rms": None if stats is None else float(np.max(stats[1])),
            "authority_ratio": auth,
            "authority_ok": bool(authority_ok),
            "stable": bool(stable[0]),
            "stable_share": float(np.mean(stable)),
            "admissible": bool(all(stable) and authority_ok),
            "admissible_point": bool(stable[0] and authority_ok),
        }
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None)) if np.all(np.isfinite(cov)) else np.full(len(names), np.nan)
    est = MarginEstimateV2(
        time=t, airspeed=U, params=dict(zip(names, map(float, k))), params_std=dict(zip(names, map(float, std))),
        flutter_speed=Uf, flutter_speed_std=Uf_std, margin=None if Uf is None else Uf / U - 1.0,
        mode_spectral_radius=point_radius, converged=fit.converged, delay_samples=int(delay), turbulence_rms=turb,
        noise_std=noise, params_cov_inflation=fit.inflation, confidence=confidence, robust=bool(ring),
        flutter_speed_worst=Uf_worst,
        margin_worst=None if Uf_worst is None else Uf_worst / U - 1.0, modes=per_mode)
    return est, fit
