"""Flutter-margin estimator: identifies the damaged wing from on-board data and predicts which modes stay stable.
Development seed 1000 (the fault-tolerant benchmark tests on seeds 0-19)."""

import numpy as np
import pytest

from flutter_ai.fault_tolerance.faults import benchmark_fault, plant_at
from flutter_ai.fault_tolerance.margin import applied_commands, estimate, fit_window
from flutter_ai.fault_tolerance.simulate import SPEED_STEP, ft_simulate
from flutter_ai.flutter import flutter_analysis
from flutter_ai.params import get_preset

DT = 0.005


@pytest.fixture(scope="module")
def wing():
    p = get_preset("uav_flap")
    fa = flutter_analysis(p)
    return p, fa.flutter_speed, fa.flutter_frequency_hz


def _estimate_at(p, Uf, fhz, kind, t_est, duration):
    fault = benchmark_fault(kind)
    r = ft_simulate(p, Uf, fhz, fault, None, duration=duration, seed=1000)
    U = round(r.airspeed[0] / SPEED_STEP) * SPEED_STEP
    ua = applied_commands(p, r.u, len(r.u), DT)
    i0, i1 = int(0.5 / DT), int(4.5 / DT)
    base = fit_window(p, U, DT, r.y[i0:i1], ua[i0:i1])
    k = int(round(t_est / DT))
    a = k - int(4.0 / DT)
    est, _ = estimate(p, Uf, t_est, U, DT, r.y[a:k], ua[a:k], base, cruise=U)
    return est, plant_at(p, fault, t_est)


def test_healthy_wing_is_recognised(wing):
    p, Uf, fhz = wing
    est, _ = _estimate_at(p, Uf, fhz, "none", 10.0, 10.0)
    assert est.flutter_speed == pytest.approx(Uf, rel=0.03)
    assert all(abs(k - 1.0) < 0.1 for k in est.params.values())
    assert all(r < 1.0 for r in est.mode_spectral_radius.values())


def test_structural_damage_is_identified_and_the_slow_down_trap_predicted(wing):
    p, Uf, fhz = wing
    est, true_p = _estimate_at(p, Uf, fhz, "stiffness_loss", 16.0, 16.0)
    assert est.params["f_alpha"] == pytest.approx(true_p.f_alpha / p.f_alpha, abs=0.03)
    assert est.flutter_speed == pytest.approx(flutter_analysis(true_p).flutter_speed, rel=0.07)
    rad = est.mode_spectral_radius
    # holding speed stays stable; the moderate slow-down (and the neural controller) do not
    assert rad["nominal"] < 1.0 and rad["cautious"] > 1.0 and rad["robust"] > 1.0 and rad["safe"] < 1.0


def test_applied_commands_respect_delay_and_slew_limit():
    p = get_preset("uav_flap")
    us = np.zeros((40, 1))
    us[5:] = np.radians(15.0)
    ua = applied_commands(p, us, 40, DT)
    assert ua[5, 0] == 0.0 and ua[6, 0] > 0.0  # one-sample delay
    assert np.max(np.diff(ua[:, 0])) <= p.u_rate_max[0] * DT + 1e-12
