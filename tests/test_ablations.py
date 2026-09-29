"""Ablation machinery: random supervisor, speed-blind policies, and the ablation runner."""

import numpy as np
import pytest

from flutter_ai.ablations import ABLATION_VARIANTS, run_ablations
from flutter_ai.controllers import NeuralPolicy, policy_path
from flutter_ai.params import get_preset
from flutter_ai.supervisor import MODES, RandomSupervisor, SituationFeatures


def features(time=1.0):
    return SituationFeatures(time=time, airspeed=54.0, airspeed_ratio=1.1, h_mm=1.0, alpha_deg=0.1,
                             peak_h_ratio=0.01, peak_alpha_ratio=0.01, risk=0.01, energy=0.01, window_energy=0.01,
                             energy_growth=1.0, saturation_fraction=0.0, control_effort=0.05, current_mode="balanced")


def test_random_supervisor_is_reproducible_and_uses_every_mode():
    a = [RandomSupervisor(seed=3).decide(features(time=t)).mode for t in range(1)]
    sup1, sup2 = RandomSupervisor(seed=3), RandomSupervisor(seed=3)
    seq1 = [sup1.decide(features(time=t)).mode for t in range(40)]
    seq2 = [sup2.decide(features(time=t)).mode for t in range(40)]
    assert seq1 == seq2 and seq1[0] == a[0]
    assert set(seq1) == set(MODES)


def test_speed_blind_policy_ignores_airspeed():
    rng = np.random.default_rng(0)
    nx, m, hidden = 8, 1, 4
    n_in = nx + m + 2
    weights = [(rng.standard_normal((hidden, n_in)), rng.standard_normal(hidden)),
               (rng.standard_normal((m, hidden)), rng.standard_normal(m))]
    scales, u_max = np.ones(nx), np.ones(m)
    blind = NeuralPolicy(weights, scales, u_max, {"config": {"use_airspeed": False}})
    aware = NeuralPolicy(weights, scales, u_max, {"config": {"use_airspeed": True}})
    x, u = rng.standard_normal(nx), np.zeros(m)
    assert np.allclose(blind.features(x, u, 1.3, 0.5), blind.features(x, u, 1.0, 0.5))
    assert not np.allclose(aware.features(x, u, 1.3, 0.5), aware.features(x, u, 1.0, 0.5))


def test_ablation_variants_change_exactly_one_setting():
    assert ABLATION_VARIANTS["full"] == {}
    assert all(len(v) == 1 for k, v in ABLATION_VARIANTS.items() if k != "full")


@pytest.mark.skipif(not policy_path("uav_section").exists(), reason="no neural policy for uav_section")
def test_ablation_runner_writes_results(tmp_path):
    res = run_ablations(get_preset("uav_section"), seeds=(0,), out_dir=tmp_path, log=lambda *a: None)
    assert (tmp_path / "ablations.json").exists()
    assert {"neural", "neural+rules", "neural+random"} <= set(res["cases"])
    assert "neural+random vs neural" in res["scenarios"]["turbulence_patch"]["statistics"]
