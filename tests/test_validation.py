"""BACT preset and validation against NASA wind-tunnel data (NASA/TP-1998-206270)."""

import math

import numpy as np
import pytest

from flutter_ai.flutter import flutter_analysis
from flutter_ai.params import FT, LBF, PSF, get_preset
from flutter_ai.scenarios import scenarios_for
from flutter_ai.validation import BACT_REFERENCE, flap_cl_delta

SPAN = 32 * 0.0254


def test_bact_unit_conversions_reproduce_table_1_stiffness():
    """k = m*omega^2 from the per-span preset must match Table 1's measured K_h and K_theta."""
    p = get_preset("bact")
    K_h_table = 2686 * LBF / FT / SPAN  # lb/ft -> N/m, per metre of span
    K_theta_table = 3000 * LBF * FT / SPAN  # lb-ft/rad -> N m/rad, per metre of span
    assert p.k_h == pytest.approx(K_h_table, rel=2e-3)
    assert p.k_alpha == pytest.approx(K_theta_table, rel=2e-3)
    assert p.a == 0.0 and p.flap_hinge == 0.5 and p.flap_span_fraction == 0.3


def test_bact_aerodynamic_centre_reproduces_table_2_moment_slope():
    p = get_preset("bact")
    cm_alpha = p.cl_alpha * (p.a - p.x_ac) / 2
    assert cm_alpha == pytest.approx(1.490, rel=1e-6)


def test_bact_flutter_matches_wind_tunnel_within_documented_accuracy():
    """
    Regression guard for the current validation result: flutter frequency within 5 % of the measured
    ~4 Hz; flutter dynamic pressure within 30 % of the measured ~148 psf (currently +24 %: incompressible
    model at Mach 0.77). If a model change improves this, tighten the bounds.
    """
    p = get_preset("bact")
    fa = flutter_analysis(p, np.linspace(20, 400, 400))
    q_psf = 0.5 * p.rho * fa.flutter_speed**2 / PSF
    ref = BACT_REFERENCE["measured"]
    assert fa.flutter_frequency_hz == pytest.approx(ref["flutter_freq_hz"], rel=0.05)
    assert q_psf == pytest.approx(ref["flutter_q_psf"], rel=0.30)


def test_modified_strip_theory_defaults_leave_other_presets_unchanged():
    p = get_preset("uav_flap")
    assert p.cl_alpha == pytest.approx(2 * math.pi) and p.x_ac == -0.5 and p.flap_span_fraction == 1.0
    assert "cl_alpha" not in p.fingerprint()  # older trained policies stay valid


def test_partial_span_flap_scales_flap_effectiveness():
    full = get_preset("uav_flap")
    part = full.with_overrides(flap_span_fraction=0.3)
    assert flap_cl_delta(part) == pytest.approx(0.3 * flap_cl_delta(full))


def test_bact_scenarios_use_scaled_disturbances():
    std = scenarios_for("uav_flap")["discrete_gust"]["scenario"]
    bact = scenarios_for("bact")["discrete_gust"]["scenario"]
    assert bact.h0 == pytest.approx(0.25 * std.h0)
    assert bact.gust.amplitude == pytest.approx(0.25 * std.gust.amplitude)
    assert bact.velocity_ratio == std.velocity_ratio
