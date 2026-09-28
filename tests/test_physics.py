"""Physics validation: these tests are the evidence that the aeroelastic model is right."""

import math

import numpy as np
import pytest

from flutter_ai.aero import theodorsen_exact, theodorsen_jones
from flutter_ai.flutter import flutter_analysis, pk_flutter_speed
from flutter_ai.model import continuous_state_space, discretize, structural_energy
from flutter_ai.params import get_preset


def test_theodorsen_function_textbook_values():
    # Tabulated Theodorsen function: C(0.1) = 0.832 - 0.172i, C(infinity) -> 0.5
    c = theodorsen_exact(0.1)
    assert c.real == pytest.approx(0.832, abs=2e-3)
    assert c.imag == pytest.approx(-0.172, abs=2e-3)
    assert theodorsen_exact(50.0).real == pytest.approx(0.5, abs=0.01)
    assert theodorsen_exact(0.0) == 1.0


def test_jones_approximation_close_to_exact():
    k = np.linspace(0.01, 2.0, 50)
    err = np.abs(theodorsen_jones(k) - theodorsen_exact(k))
    assert err.max() < 0.02


def test_hodges_pierce_benchmark():
    """Classic textbook case: divergence at sqrt(8) b*omega_a exactly; flutter near 2.17 b*omega_a."""
    p = get_preset("hodges_pierce")
    u_ref = p.b * p.omega_alpha
    r = flutter_analysis(p)
    assert r.divergence_speed / u_ref == pytest.approx(math.sqrt(8.0), rel=1e-3)
    assert r.flutter_speed / u_ref == pytest.approx(2.17, abs=0.02)


@pytest.mark.parametrize("preset", ["hodges_pierce", "uav_section"])
def test_state_space_matches_independent_pk_method(preset):
    """Time-domain (Jones/Wagner) eigenvalues vs p-k with the exact Theodorsen function."""
    p = get_preset(preset)
    ss = flutter_analysis(p).flutter_speed
    pk = pk_flutter_speed(p)
    assert abs(ss - pk) / pk < 0.015


@pytest.mark.parametrize("preset", ["uav_section", "uav_flap"])
def test_stable_below_unstable_above_flutter(preset):
    p = get_preset(preset)
    Uf = flutter_analysis(p).flutter_speed
    below = np.linalg.eigvals(continuous_state_space(p, 0.95 * Uf).A)
    above = np.linalg.eigvals(continuous_state_space(p, 1.05 * Uf).A)
    assert np.max(below.real) < 0
    assert np.max(above.real) > 0


def test_structure_conserves_energy_in_vacuum():
    """No air, no structural damping -> purely imaginary eigenvalues (energy conserved)."""
    p = get_preset("uav_section", zeta_h=0.0, zeta_alpha=0.0).with_overrides(rho=1e-12)
    eig = np.linalg.eigvals(continuous_state_space(p, 1.0).A)
    osc = eig[np.abs(eig.imag) > 1]
    assert np.max(np.abs(osc.real)) < 1e-6


def test_zero_order_hold_discretisation_matches_matrix_exponential():
    p = get_preset()
    ss = continuous_state_space(p, 40.0)
    d = discretize(ss, 0.005)
    from scipy.linalg import expm

    assert np.allclose(d.Ad, expm(ss.A * 0.005), atol=1e-12)


def test_structural_energy_positive():
    p = get_preset("uav_section")
    x = np.array([[0.01, 0.02, 0.1, 0.5, 0.0, 0.0]])
    assert structural_energy(p, x)[0] > 0


# --------------------------------------------------------------------------- #
# 3-DOF (plunge, pitch, trailing-edge flap)
# --------------------------------------------------------------------------- #
def _thin_airfoil_loads(b, a, c, alpha, beta, V=1.0, rho=1.0, N=3000, G=200001):
    """Independent steady thin-airfoil theory (Glauert series) for a flapped plate."""
    th_h = np.arccos(-c)
    A0 = alpha + beta * (np.pi - th_h) / np.pi
    n = np.arange(1, N + 1)
    An = 2 * beta / (n * np.pi) * np.sin(n * th_h)
    L = np.pi * rho * V**2 * b * (2 * A0 + An[0])
    th = np.linspace(1e-9, np.pi - 1e-9, G)
    x = -b * np.cos(th)
    gamma = 2 * V * (A0 * (1 + np.cos(th)) / np.sin(th) + (An[:, None] * np.sin(np.outer(n, th))).sum(0))
    dp = rho * V * gamma * b * np.sin(th)  # lift per d(theta), positive up
    M_ea = np.trapezoid(dp * (a * b - x), th)  # nose-up about the elastic axis
    m = th >= th_h
    M_h = -np.trapezoid(dp[m] * (x[m] - c * b), th[m])  # trailing-edge-down about the hinge
    return np.array([L, M_ea, M_h])


@pytest.mark.parametrize("a,c", [(-0.3, 0.6), (-0.5, 0.5), (0.0, 0.3)])
def test_flap_steady_loads_match_independent_thin_airfoil_theory(a, c):
    from flutter_ai.model import aero_force_vector, downwash_rows, noncirculatory_matrices

    p = get_preset("uav_flap", a=a, flap_hinge=c).with_overrides(rho=1.0)
    _, _, Knc = noncirculatory_matrices(p, 1.0)
    dq, _ = downwash_rows(p, 1.0)
    K = Knc + np.outer(aero_force_vector(p, 1.0), dq)  # steady aero stiffness, C(k) = 1
    for q in (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])):
        f = K @ q
        theodorsen = np.array([f[0], -f[1], -f[2]])  # lift up, pitch moment, hinge moment
        reference = _thin_airfoil_loads(p.b, a, c, q[1], q[2])
        assert np.allclose(theodorsen, reference, rtol=2e-4, atol=1e-7)


def test_flap_added_mass_is_symmetric_positive_definite():
    """Potential-flow kinetic energy => apparent-mass matrix symmetric positive definite."""
    from flutter_ai.model import noncirculatory_matrices

    for c in (0.2, 0.5, 0.8):
        Ma, _, _ = noncirculatory_matrices(get_preset("uav_flap", flap_hinge=c), 30.0)
        assert np.allclose(Ma, Ma.T, rtol=1e-12, atol=1e-15)
        assert np.all(np.linalg.eigvalsh(Ma) > 0)


def test_stiff_flap_recovers_two_dof_flutter_speed():
    two = flutter_analysis(get_preset("uav_section")).flutter_speed
    three = flutter_analysis(get_preset("uav_flap", f_beta=1000.0)).flutter_speed
    assert three == pytest.approx(two, rel=2e-3)


def test_three_dof_state_space_matches_pk():
    p = get_preset("uav_flap")
    assert abs(flutter_analysis(p).flutter_speed - pk_flutter_speed(p)) / pk_flutter_speed(p) < 0.015
