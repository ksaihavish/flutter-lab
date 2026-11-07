"""Parameter identification recovers a known wing from its simulated time histories."""

from dataclasses import replace

import pytest

from flutter_ai.controllers import LQRController, OpenLoop
from flutter_ai.data import export_run_csv, read_run_csv
from flutter_ai.flutter import flutter_speed
from flutter_ai.identification import identify, simulate_recorded
from flutter_ai.params import get_preset
from flutter_ai.scenarios import SCENARIOS
from flutter_ai.simulation import simulate


def _runs(true_wing, nominal, tmp_path):
    """Two synthetic 'experiments' on the true wing: a sub-critical gust and a controlled run above flutter."""
    Uf = flutter_speed(nominal)
    below = replace(SCENARIOS["below_flutter"]["scenario"], duration=4.0)
    above = replace(SCENARIOS["discrete_gust"]["scenario"], duration=4.0)
    runs = []
    for i, (ctrl, sc) in enumerate(((OpenLoop(), below), (LQRController(), above))):
        r = simulate(true_wing, ctrl, sc, Uf, design_params=nominal)
        runs.append(read_run_csv(export_run_csv(r, tmp_path / f"run{i}.csv")))
    return runs


def test_recorded_simulation_reproduces_the_simulator(tmp_path):
    p = replace(get_preset("uav_flap"), u_rate_max=None)
    below, above = _runs(p, p, tmp_path)
    whole = simulate_recorded(p, below, input_delay_steps=1)  # stable: one simulation is exact
    assert abs(whole[:, 1] - below["alpha"]).max() < 1e-6
    unstable = simulate_recorded(p, above, input_delay_steps=1)  # above flutter: rounding errors explode
    assert abs(unstable[:, 1] - above["alpha"]).max() > 1e-2
    shot = simulate_recorded(p, above, input_delay_steps=1, segment_steps=50)  # multiple shooting stays close
    assert abs(shot[:, 1] - above["alpha"]).max() < 1e-2 * abs(above["alpha"]).max()  # lag states rebuilt with FOH


def test_identification_recovers_perturbed_parameters(tmp_path):
    nominal = replace(get_preset("uav_flap"), u_rate_max=None)
    truth = nominal.with_overrides(f_h=nominal.f_h * 0.95, f_alpha=nominal.f_alpha * 1.06,
                                   zeta_alpha=nominal.zeta_alpha * 2.0)
    res = identify(nominal, _runs(truth, nominal, tmp_path), ("f_h", "f_alpha", "zeta_h", "zeta_alpha"))
    got = {r["parameter"]: r["identified"] for r in res["parameters"]}
    assert got["f_h"] == pytest.approx(truth.f_h, rel=1e-3)
    assert got["f_alpha"] == pytest.approx(truth.f_alpha, rel=1e-3)
    assert got["zeta_alpha"] == pytest.approx(truth.zeta_alpha, rel=0.05)
    assert res["rms_error_after"] < 0.01 * res["rms_error_before"]


def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError):
        identify(get_preset("uav_section"), [], ("f_beta",))
