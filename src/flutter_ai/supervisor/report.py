"""
Situation reports: compress the recent closed-loop history into features and a
human/LLM-readable text. Only information available on board is used
(measurements, commanded control, airspeed) - never the true gust.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..model import structural_energy
from ..params import WingParams


@dataclass
class SituationFeatures:
    time: float
    airspeed: float
    airspeed_ratio: float  # U / U_flutter
    h_mm: float
    alpha_deg: float
    peak_h_ratio: float  # window peak |h| / h_limit
    peak_alpha_ratio: float  # window peak |alpha| / alpha_limit
    risk: float  # max of the two ratios
    energy: float  # current structural energy [J/m]
    window_energy: float  # mean structural energy over this window [J/m]
    energy_growth: float  # window energy / previous window energy, >1 = growing
    saturation_fraction: float  # share of samples with any actuator at its limit
    control_effort: float  # RMS of u / u_max over the window
    current_mode: str

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in asdict(self).items()}


def extract_features(
    p: WingParams,
    t: float,
    U: float,
    flutter_speed: float,
    y_window: np.ndarray,  # (T, 2n) measured [q, q'] over the window
    u_window: np.ndarray,  # (T, m) commanded (saturated) inputs
    prev_energy: float | None,
    current_mode: str,
) -> SituationFeatures:
    energy_series = structural_energy(p, y_window)  # uses only [q, q']
    energy = float(energy_series[-1])
    window_energy = float(np.mean(energy_series))
    growth = window_energy / prev_energy if prev_energy and prev_energy > 1e-12 else 1.0

    peak_h = float(np.max(np.abs(y_window[:, 0]))) / p.h_limit
    peak_a = float(np.max(np.abs(np.degrees(y_window[:, 1])))) / p.alpha_limit_deg
    u_max = np.asarray(p.u_max)
    sat = float(np.mean(np.any(np.abs(u_window) >= 0.999 * u_max, axis=1))) if len(u_window) else 0.0
    effort = float(np.sqrt(np.mean(np.sum((u_window / u_max) ** 2, axis=1)))) if len(u_window) else 0.0

    return SituationFeatures(
        time=t,
        airspeed=U,
        airspeed_ratio=U / flutter_speed,
        h_mm=float(y_window[-1, 0] * 1000.0),
        alpha_deg=float(np.degrees(y_window[-1, 1])),
        peak_h_ratio=peak_h,
        peak_alpha_ratio=peak_a,
        risk=max(peak_h, peak_a),
        energy=energy,
        window_energy=window_energy,
        energy_growth=growth,
        saturation_fraction=sat,
        control_effort=effort,
        current_mode=current_mode,
    )


def _trend(growth: float) -> str:
    if growth > 1.25:
        return "INCREASING"
    if growth < 0.8:
        return "DECREASING"
    return "STEADY"


def _risk_level(risk: float) -> str:
    if risk > 0.6:
        return "HIGH - approaching structural limits"
    if risk > 0.25:
        return "ELEVATED - significant oscillation"
    if risk > 0.05:
        return "MODERATE - small oscillation"
    return "LOW - quiescent"


def format_report(f: SituationFeatures) -> str:
    return f"""AEROELASTIC SYSTEM STATUS REPORT
Time: {f.time:.2f} s
Airspeed: {f.airspeed:.1f} m/s ({100 * f.airspeed_ratio:.0f}% of open-loop flutter speed)

STRUCTURAL STATE (latest sample):
- Plunge: {f.h_mm:+.2f} mm
- Pitch: {f.alpha_deg:+.2f} deg
- Window peak plunge: {100 * f.peak_h_ratio:.0f}% of structural limit
- Window peak pitch: {100 * f.peak_alpha_ratio:.0f}% of structural limit

ENERGY:
- Structural energy: {f.energy:.4g} J/m
- Trend vs previous window: {_trend(f.energy_growth)} (x{f.energy_growth:.2f})

ACTUATORS:
- RMS effort: {100 * f.control_effort:.0f}% of actuator authority
- Time at saturation: {100 * f.saturation_fraction:.0f}%

CURRENT MODE: {f.current_mode}
RISK LEVEL: {_risk_level(f.risk)}"""
