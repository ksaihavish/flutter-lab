"""Robustness study, model mismatch (design vs true wing) and the actuator slew-rate limit."""

import math
from dataclasses import replace

import numpy as np

from flutter_ai.controllers import LQRController
from flutter_ai.flutter import flutter_speed
from flutter_ai.params import get_preset
from flutter_ai.robustness import perturbed_wing, robustness_study
from flutter_ai.scenarios import SCENARIOS
from flutter_ai.simulation import simulate


def test_perturbed_wing_stays_within_bounds_and_is_reproducible():
    p = get_preset("uav_flap")
    a = perturbed_wing(p, 0.2, np.random.default_rng(1))
    b = perturbed_wing(p, 0.2, np.random.default_rng(1))
    assert a == b
    assert 0.8 <= a.mass / p.mass <= 1.2
    assert 0.8 <= a.zeta_h / p.zeta_h <= 1.2 and 0.8 <= a.f_beta / p.f_beta <= 1.2
    # frequency = sqrt(stiffness factor / mass factor), both within +/-20 %
    assert math.sqrt(0.8 / 1.2) <= a.f_alpha / p.f_alpha <= math.sqrt(1.2 / 0.8)
    assert a.name == p.name  # controllers look up per-preset settings by name


def test_design_params_decouple_the_controller_model_from_the_true_wing():
    p = get_preset("uav_section")
    Uf = flutter_speed(p)
    true_wing = p.with_overrides(f_alpha=p.f_alpha * 1.1)
    sc = replace(SCENARIOS["discrete_gust"]["scenario"], duration=4.0)
    nominal_design = simulate(true_wing, LQRController(), sc, Uf, design_params=p)
    matched_design = simulate(true_wing, LQRController(), sc, Uf)
    assert not np.allclose(nominal_design.u, matched_design.u)


def test_actuator_rate_limit_bounds_the_applied_command():
    p = get_preset("uav_flap")
    Uf = flutter_speed(p)
    limited = replace(p, u_rate_max=(math.radians(200.0),))
    r = simulate(limited, LQRController(), replace(SCENARIOS["discrete_gust"]["scenario"], duration=4.0), Uf,
                 design_params=p)
    step = np.abs(np.diff(r.u_applied[:, 0]))
    assert step.max() <= math.radians(200.0) * 0.005 + 1e-12
    free = simulate(p, LQRController(), replace(SCENARIOS["discrete_gust"]["scenario"], duration=4.0), Uf)
    assert np.abs(np.diff(free.u_applied[:, 0])).max() > step.max()  # the limit was actually active


def test_robustness_study_writes_results_and_figure(tmp_path):
    res = robustness_study(get_preset("uav_section"), n_draws=1, controllers=["lqr"], out_dir=tmp_path,
                           log=lambda *a: None)
    assert (tmp_path / "robustness.json").exists() and (tmp_path / "fig10_robustness.png").exists()
    row = res["parameters"]["discrete_gust"]["0.05"]
    assert row["lqr"]["n"] == 1 and 0.0 <= row["lqr"]["failure_rate"] <= 1.0
    assert res["rate_limit"] == {}  # force/moment actuation has no flap to rate-limit
