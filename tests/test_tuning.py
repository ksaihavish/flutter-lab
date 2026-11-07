"""Shared rate penalty in the classical designs, and the tuned design knobs."""

import json
from dataclasses import replace

import numpy as np
from scipy.linalg import solve_discrete_are

from flutter_ai.controllers import LQRController, delay_augmented
from flutter_ai.controllers import design as design_mod
from flutter_ai.controllers.costs import input_weight, state_weight
from flutter_ai.controllers.lqr import dlqr_with_rate
from flutter_ai.flutter import flutter_speed
from flutter_ai.model import discrete_state_space
from flutter_ai.params import get_preset
from flutter_ai.scenarios import SCENARIOS
from flutter_ai.simulation import simulate


def test_zero_rate_weight_recovers_plain_lqr():
    p = get_preset("uav_section")
    sys = discrete_state_space(p, 1.1 * flutter_speed(p), 0.005)
    Q, R = state_weight(p) * 0.005, input_weight(p, 0.5) * 0.005
    K_rate = dlqr_with_rate(sys.Ad, sys.Bd, Q, R, np.zeros_like(R), delay=1)
    A, B = delay_augmented(sys.Ad, sys.Bd, 1)
    Qa = np.zeros_like(A)
    Qa[:Q.shape[0], :Q.shape[0]] = Q
    P = solve_discrete_are(A, B, Qa, R)
    K_plain = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    assert np.allclose(K_rate, K_plain, atol=1e-8)


def test_larger_rate_multiplier_gives_smoother_commands():
    p = get_preset("uav_flap")
    Uf = flutter_speed(p)
    sc = replace(SCENARIOS["discrete_gust"]["scenario"], duration=4.0)
    rates = [simulate(p, LQRController(rate_multiplier=rho), sc, Uf).metrics["control_rate_rms"] for rho in (1.0, 30.0)]
    assert rates[1] < rates[0]


def test_tuned_knobs_are_used_only_for_the_parameters_they_were_tuned_for(tmp_path, monkeypatch):
    monkeypatch.setattr(design_mod, "MODELS_DIR", tmp_path)
    p = get_preset("uav_section")
    assert design_mod.design_settings(p, "lqr") == ({"rate_multiplier": 1.0}, None)  # no file: defaults
    (tmp_path / "classical_tuning_uav_section.json").write_text(json.dumps(
        {"params_fingerprint": p.fingerprint(), "lqr": {"rate_multiplier": 30.0},
         "lqg": {"rate_multiplier": 10.0, "measurement_noise": 0.01}}))
    assert design_mod.design_settings(p, "lqr")[0]["rate_multiplier"] == 30.0
    assert design_mod.design_settings(p, "lqg")[0] == {"rate_multiplier": 10.0, "measurement_noise": 0.01}
    other = p.with_overrides(f_alpha=p.f_alpha * 1.1)
    settings, warning = design_mod.design_settings(other, "lqr")
    assert settings["rate_multiplier"] == 1.0 and "tuned for other" in warning
