"""
Validation against published wind-tunnel data.

    python -m flutter_ai validate          # -> outputs/validation/bact.json + printed table

Every "measured" number below is quoted from its source; every "predicted" number is computed.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .flutter import flutter_analysis, pk_flutter_speed
from .params import FT, PSF, WingParams, get_preset
from .paths import OUTPUTS_DIR

BACT_REFERENCE: dict[str, Any] = {
    "measured": {"flutter_q_psf": 148.0, "flutter_freq_hz": 4.0, "mach": 0.77, "medium": "R-12 heavy gas",
                 "source": "Waszak, NASA/TP-1998-206270, 'Flutter Properties' (approximate values)"},
    "nasa_model": {"flutter_q_psf": 150.8, "flutter_freq_hz": 4.16,
                   "source": "Waszak, NASA/TP-1998-206270: derivative-based model with measured + CFD aero"},
    "measured_flap_cl_delta": 0.63,  # Table 2, per rad
}


def _predict(p: WingParams, velocities: np.ndarray) -> dict:
    fa = flutter_analysis(p, velocities)
    if fa.flutter_speed is None:
        return {"flutter_speed_mps": None}
    pk = pk_flutter_speed(p, velocities)
    return {
        "flutter_speed_mps": fa.flutter_speed,
        "flutter_q_psf": 0.5 * p.rho * fa.flutter_speed**2 / PSF,
        "flutter_freq_hz": fa.flutter_frequency_hz,
        "pk_flutter_speed_mps": pk,
        "divergence_speed_mps": fa.divergence_speed,
    }


def flap_cl_delta(p: WingParams) -> float:
    """Steady lift-curve slope due to flap deflection predicted by the model (per rad, whole wing)."""
    from .aero import theodorsen_T

    if p.flap_hinge is None:
        raise ValueError(f"Preset '{p.name}' has no flap")
    T10 = theodorsen_T(p.flap_hinge, p.a)[10]
    return (p.cl_alpha / (2 * np.pi)) * 2.0 * T10 * p.flap_span_fraction


def validate_bact(out_dir: Path | None = None, log=print) -> dict[str, Any]:
    p = get_preset("bact")
    v = np.linspace(20.0, 400.0, 400)
    ref = BACT_REFERENCE["measured"]
    variants = {
        "preset: modified strip theory (measured CL_a, AC 17.5% c, 30%-span flap)": p,
        "plain 2D Theodorsen (thin-airfoil CL_a = 2 pi, AC 25% c, full-span flap)": replace(
            p, cl_alpha=2 * np.pi, x_ac=-0.5, flap_span_fraction=1.0),
        "sensitivity: AC at 32.5% c (report text) instead of 17.5% c": replace(p, x_ac=-0.35),
        "sensitivity: density from U = 0.77 x 152 m/s (R-12 speed of sound)": replace(
            p, rho=2 * ref["flutter_q_psf"] * PSF / (0.77 * 152.0) ** 2),
    }
    rows: dict[str, dict[str, Any]] = {}
    for label, params in variants.items():
        pred = _predict(params, v)
        if pred.get("flutter_q_psf"):
            pred["q_error_pct"] = 100 * (pred["flutter_q_psf"] - ref["flutter_q_psf"]) / ref["flutter_q_psf"]
            pred["freq_error_pct"] = 100 * (pred["flutter_freq_hz"] - ref["flutter_freq_hz"]) / ref["flutter_freq_hz"]
        rows[label] = pred
    result = {
        "reference": BACT_REFERENCE,
        "assumptions": {
            "density_kg_m3": p.rho,
            "density_basis": "q = 148 psf at U = 400 ft/s (report's reference speed)",
            "reference_speed_mps": 400 * FT,
            "estimated_not_in_source": ["flap inertia about hinge", "structural limits h_limit, alpha_limit"],
            "model_limits": "incompressible unsteady terms (Theodorsen/Wagner) at Mach 0.77; strip theory",
        },
        "predictions": rows,
        "flap_effectiveness": {"predicted_cl_delta": flap_cl_delta(p),
                               "measured_cl_delta": BACT_REFERENCE["measured_flap_cl_delta"]},
    }

    log(f"BACT flutter validation (measured: ~{ref['flutter_q_psf']:.0f} psf at ~{ref['flutter_freq_hz']:.0f} Hz, "
        f"Mach {ref['mach']}, {ref['medium']}; NASA model: "
        f"{BACT_REFERENCE['nasa_model']['flutter_q_psf']} psf / {BACT_REFERENCE['nasa_model']['flutter_freq_hz']} Hz)")
    for label, r in rows.items():
        if r.get("flutter_q_psf"):
            log(f"  {r['flutter_q_psf']:6.1f} psf ({r['q_error_pct']:+5.1f}%)  {r['flutter_freq_hz']:5.2f} Hz "
                f"({r['freq_error_pct']:+5.1f}%)   {label}")
        else:
            log(f"  no flutter found   {label}")
    fe: dict[str, Any] = result["flap_effectiveness"]
    log(f"  flap CL_delta: predicted {fe['predicted_cl_delta']:.3f} /rad vs measured {fe['measured_cl_delta']:.2f} /rad")

    out_dir = Path(out_dir or OUTPUTS_DIR / "validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bact.json").write_text(json.dumps(result, indent=2))
    log(f"Saved {out_dir / 'bact.json'}")
    return result
