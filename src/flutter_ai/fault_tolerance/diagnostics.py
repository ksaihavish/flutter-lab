"""
On-board diagnostics for fault-tolerant supervision. Only information available on the aircraft is used:
measurements y = [q, q'], the commands the flight computer sent, the airspeed, and its own loop timing.

Main signal: the one-step prediction residual of the *nominal* model, per channel. The flight computer predicts
the next measurement from the current one (aero-lag states rebuilt from the measured downwash, as the observer
does) and the command it applied; faults leave different signatures:
    servo slower          -> flap channel;   stiffness / flap damage -> pitch / plunge channels;
    sensor noise          -> every channel, plus high-frequency content;   turbulence -> plunge / pitch.
Residuals are reported as multiples of a healthy baseline measured at the start of the flight.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..aero import JONES_EPS
from ..controllers.costs import reference_scales
from ..model import discrete_state_space, downwash_rows
from ..params import WingParams

LOOKBACK_S = 4.0  # features use the last 4 s, whatever the supervisor's interval
# the rebuilt aero-lag states start at zero in each window; skip their transient (slowest lag time constant
# b / (eps_1 U) is about 0.1 s at 0.7 U_F on uav_flap, so 0.5 s leaves < 1 %)
RESIDUAL_WARMUP_S = 0.5


@dataclass
class FTFeatures:
    time: float
    airspeed: float
    airspeed_ratio: float  # U / nominal flutter speed
    mode: str
    risk: float  # window peak |h|, |alpha| as a fraction of the structural limits (max of the two)
    energy_growth: float  # mean structural energy, last half of the window / first half
    saturation_fraction: float  # share of samples with the command at its limit
    pitch_freq_hz: float | None  # dominant pitch oscillation frequency (None if the motion is too small)
    nominal_freq_hz: float | None  # the same, measured on the healthy wing at the start of the flight
    residual_ratio: dict[str, float]  # one-step model residual per channel / healthy baseline
    noise_ratio: float  # high-frequency measurement content / healthy baseline
    loop_delay_ms: float  # measured by the flight computer
    margin: dict | None = None  # latest on-board flutter-margin estimate (margin.MarginEstimate.to_dict()), if any

    def to_dict(self) -> dict:
        return asdict(self)


def one_step_residual(p: WingParams, U: float, dt: float, y: np.ndarray, u_applied: np.ndarray) -> np.ndarray:
    """(T-1, n_dof) RMS-normalised one-step prediction error of the nominal model on the rate channels q'."""
    n = p.n_dof
    sys = discrete_state_space(p, U, dt)
    dw_dq, dw_dqdot = downwash_rows(p, U)
    lam = np.array(JONES_EPS) * U / p.b
    decay = np.exp(-lam * dt)
    b0 = (1.0 - decay) / lam
    b1 = (dt / lam - b0 / lam) / dt
    Q = y[:, :n] @ dw_dq + y[:, n:2 * n] @ dw_dqdot
    z = np.zeros((len(y), 2))
    for k in range(len(y) - 1):
        z[k + 1] = decay * z[k] + (b0 - b1) * Q[k] + b1 * Q[k + 1]
    x = np.hstack([y, z])
    pred = x[:-1] @ sys.Ad.T + u_applied[:-1] @ sys.Bd.T
    rate_ref = reference_scales(p)[n:2 * n]
    r = (y[1:, n:2 * n] - pred[:, n:2 * n]) / rate_ref
    skip = int(round(RESIDUAL_WARMUP_S / dt))
    return r[skip:] if len(r) > 2 * skip else r


def _noise_level(p: WingParams, y: np.ndarray) -> float:
    """RMS of the second difference of h and alpha (high-frequency content), normalised by the reference scales."""
    ref = reference_scales(p)[:2]
    d2 = np.diff(y[:, :2], n=2, axis=0) / ref
    return float(np.sqrt(np.mean(d2**2))) if len(d2) else 0.0


def _dominant_freq(signal: np.ndarray, dt: float, min_amp: float) -> float | None:
    """Peak of the Welch spectrum (1 s segments, 50 % overlap) in 3-30 Hz: a single FFT of small random motion is
    too noisy (healthy estimates wandered by +/-1.5 Hz)."""
    from scipy.signal import welch

    s = signal - np.mean(signal)
    seg = int(round(1.0 / dt))
    if len(s) < seg or np.max(np.abs(s)) < min_amp:
        return None
    freqs, power = welch(s, fs=1.0 / dt, nperseg=seg, noverlap=seg // 2, nfft=8 * seg)
    band = (freqs > 3.0) & (freqs < 30.0)
    return float(freqs[band][np.argmax(power[band])])


@dataclass
class Baseline:
    residual_rms: np.ndarray  # per channel
    noise: float
    freq_hz: float | None = None  # dominant pitch frequency of the healthy closed loop


def raw_features(p: WingParams, U: float, dt: float, y: np.ndarray, u_applied: np.ndarray) -> tuple[np.ndarray, float]:
    r = one_step_residual(p, U, dt, y, u_applied)
    return np.sqrt(np.mean(r**2, axis=0)), _noise_level(p, y)


def healthy_freq(p: WingParams, y: np.ndarray, dt: float) -> float | None:
    return _dominant_freq(y[:, 1], dt, min_amp=0.002 * np.radians(p.alpha_limit_deg))


def extract(p: WingParams, t: float, U: float, flutter_speed: float, flutter_freq_hz: float, dt: float,
            y: np.ndarray, u_cmd: np.ndarray, u_applied_nominal: np.ndarray, mode: str, baseline: Baseline,
            loop_delay_ms: float) -> FTFeatures:
    """Features over the look-back window. u_applied_nominal = commands shifted by the *nominal* loop delay."""
    n = p.n_dof
    h_lim, a_lim = p.h_limit, np.radians(p.alpha_limit_deg)
    risk = float(max(np.max(np.abs(y[:, 0])) / h_lim, np.max(np.abs(y[:, 1])) / a_lim))
    from ..model import structural_energy

    e = structural_energy(p, y)
    half = len(e) // 2
    growth = float(np.mean(e[half:]) / max(np.mean(e[:half]), 1e-15)) if half else 1.0
    u_max = np.asarray(p.u_max)
    sat = float(np.mean(np.any(np.abs(u_cmd) >= 0.999 * u_max, axis=1))) if len(u_cmd) else 0.0
    res, noise = raw_features(p, U, dt, y, u_applied_nominal)
    names = p.dof_names
    ratios = {names[j]: float(res[j] / max(baseline.residual_rms[j], 1e-12)) for j in range(n)}
    freq = _dominant_freq(y[:, 1], dt, min_amp=0.002 * a_lim)
    return FTFeatures(time=t, airspeed=U, airspeed_ratio=U / flutter_speed, mode=mode, risk=risk, energy_growth=growth,
                      saturation_fraction=sat, pitch_freq_hz=freq, nominal_freq_hz=baseline.freq_hz,
                      residual_ratio=ratios, noise_ratio=float(noise / max(baseline.noise, 1e-12)),
                      loop_delay_ms=loop_delay_ms)


def _level(x: float) -> str:
    return "NORMAL" if x < 2.0 else "ELEVATED" if x < 5.0 else "HIGH"


def format_ft_report(f: FTFeatures) -> str:
    if f.pitch_freq_hz is None:
        freq = "no clear oscillation"
    elif f.nominal_freq_hz is None:
        freq = f"{f.pitch_freq_hz:.1f} Hz"
    else:
        freq = (f"{f.pitch_freq_hz:.1f} Hz (healthy: {f.nominal_freq_hz:.1f} Hz, "
                f"{100 * (f.pitch_freq_hz - f.nominal_freq_hz) / f.nominal_freq_hz:+.0f} %)")
    res = "\n".join(f"- {name}: x{r:.1f} of healthy baseline ({_level(r)})" for name, r in f.residual_ratio.items())
    return f"""FLUTTER SUPPRESSION STATUS REPORT
Time: {f.time:.1f} s
Airspeed: {f.airspeed:.1f} m/s ({100 * f.airspeed_ratio:.0f}% of the healthy wing's flutter speed)
Current mode: {f.mode}

RESPONSE (last {LOOKBACK_S:.0f} s):
- Peak motion: {100 * f.risk:.0f}% of structural limit
- Energy trend: x{f.energy_growth:.2f}
- Actuator at its limit: {100 * f.saturation_fraction:.0f}% of the time
- Dominant pitch oscillation: {freq}

MODEL CONSISTENCY (nominal model's one-step prediction error, per channel):
{res}
- Measurement high-frequency content: x{f.noise_ratio:.1f} of healthy baseline ({_level(f.noise_ratio)})

SYSTEM:
- Control loop delay: {f.loop_delay_ms:.0f} ms""" + format_physics(f)


def format_physics(f: FTFeatures) -> str:
    """The physics block of the report (only when the on-board margin estimator runs)."""
    m = f.margin
    if not m:
        return ""
    names = {"f_alpha": "pitch stiffness", "f_h": "plunge stiffness", "flap_span_fraction": "flap effectiveness"}
    params = ", ".join(f"{names.get(k, k)} x{v:.2f} of healthy" for k, v in m["params"].items())
    healthy = f.airspeed / f.airspeed_ratio
    if m["flutter_speed"] is None:
        flutter = "no flutter speed found in the analysis range"
    else:
        flutter = (f"{m['flutter_speed']:.1f} m/s (healthy wing {healthy:.1f} m/s); margin at the current airspeed "
                   f"{100 * m['margin']:+.0f} % (negative: above the open-loop flutter speed, the controller holds it)")
    modes = ", ".join(f"{mode} {'stable' if r < 1.0 else 'UNSTABLE'}" for mode, r in m["mode_spectral_radius"].items())
    return f"""

PHYSICS ESTIMATE (on-board identification of the wing, last {LOOKBACK_S:.0f} s, at {m['time']:.1f} s):
- Identified: {params}
- Estimated flutter speed: {flutter}
- Predicted stability of each mode on the identified wing (worst case on the way to its airspeed): {modes}
- Blind spots: this estimate cannot see actuator, sensor, loop-timing or turbulence problems."""
