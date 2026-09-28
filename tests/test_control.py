"""Controllers, simulator honesty (no NaN masking, no clipping), delay handling and metrics."""

from dataclasses import replace

import numpy as np
import pytest

from flutter_ai.controllers import (
    LQRController,
    NeuralController,
    NeuralPolicy,
    OpenLoop,
    closed_loop_radius,
    delay_augmented,
    lqr_gain,
    policy_path,
)
from flutter_ai.flutter import flutter_speed
from flutter_ai.gusts import GustSpec
from flutter_ai.model import discrete_state_space
from flutter_ai.params import get_preset
from flutter_ai.scenarios import SCENARIOS
from flutter_ai.simulation import Scenario, simulate
from flutter_ai.supervisor import RuleSupervisor

PRESETS = ["uav_flap", "uav_section"]
QUIET = Scenario(velocity_ratio=1.1, duration=3.0, gust=GustSpec(kind="none"))


def setup(preset):
    p = get_preset(preset)
    return p, flutter_speed(p)


def needs_policy(preset):
    return pytest.mark.skipif(not policy_path(preset).exists(), reason=f"no neural policy for {preset}")


@pytest.mark.parametrize("preset", PRESETS)
def test_open_loop_above_flutter_is_flagged_not_hidden(preset):
    p, Uf = setup(preset)
    r = simulate(p, OpenLoop(), QUIET, Uf)
    assert r.exceeded_limits
    assert r.failure_time is not None and r.failure_time < 3.0
    assert np.all(np.isfinite(r.x))
    assert r.metrics["settling_time_s"] is None


@pytest.mark.parametrize("preset", PRESETS)
def test_open_loop_below_flutter_decays(preset):
    p, Uf = setup(preset)
    r = simulate(p, OpenLoop(), replace(QUIET, velocity_ratio=0.8), Uf)
    assert not r.exceeded_limits
    assert abs(r.x[-1, 0]) < 1e-4


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("ratio", [1.1, 1.3])
@pytest.mark.parametrize("delay", [0, 1, 2])
def test_delay_aware_lqr_is_stable_at_its_design_delay(preset, ratio, delay):
    p, Uf = setup(preset)
    d = discrete_state_space(p, ratio * Uf, 0.005)
    K = lqr_gain(p, ratio * Uf, 0.005, 0.5, delay)
    A, B = delay_augmented(d.Ad, d.Bd, delay)
    rho = np.max(np.abs(np.linalg.eigvals(A - B @ K)))
    assert rho < 1.0
    assert closed_loop_radius(p, ratio * Uf, 0.005, K, delay, with_observer=False) == pytest.approx(rho)
    assert closed_loop_radius(p, ratio * Uf, 0.005, K, delay) < 1.0  # as implemented, with the observer


def test_delay_margin_analysis_agrees_with_simulation():
    """The linear analysis must predict what the simulator does (it once did not: missing observer)."""
    p, Uf = setup("uav_flap")
    U = 1.1 * Uf
    K = lqr_gain(p, U, 0.005, 1.0, 1)  # aggressive LQR designed for 1 step of delay
    tiny = Scenario(velocity_ratio=1.1, duration=2.0, h0=1e-4, alpha0_deg=0.01, gust=GustSpec(kind="none"))
    for delay in (0, 1):
        rho = closed_loop_radius(p, U, 0.005, K, delay)
        r = simulate(p, LQRController(design_delay=1), replace(tiny, input_delay_steps=delay), Uf, aggressiveness=1.0)
        grew = r.exceeded_limits or np.max(np.abs(r.x[-40:, 0])) > 1e-4
        assert grew == (rho > 1.0), (delay, rho)


def test_input_delay_is_applied():
    """A 1-step delay must change the trajectory versus no delay (the command reaches the plant later)."""
    p, Uf = setup("uav_flap")
    sc = SCENARIOS["discrete_gust"]["scenario"]
    r0 = simulate(p, LQRController(design_delay=1), replace(sc, input_delay_steps=0), Uf)
    r1 = simulate(p, LQRController(design_delay=1), replace(sc, input_delay_steps=1), Uf)
    assert not np.allclose(r0.x[:, 0], r1.x[:, 0])


@pytest.mark.parametrize("preset", PRESETS)
def test_lqr_suppresses_flutter_with_saturated_actuators(preset):
    p, Uf = setup(preset)
    r = simulate(p, LQRController(), SCENARIOS["discrete_gust"]["scenario"], Uf)
    assert not r.exceeded_limits
    assert np.all(np.abs(r.u) <= np.asarray(p.u_max) + 1e-12)


def test_aggressiveness_trades_effort_for_response():
    p, Uf = setup("uav_section")
    sc = SCENARIOS["turbulence_patch"]["scenario"]
    lo = simulate(p, LQRController(), sc, Uf, aggressiveness=0.0).metrics
    hi = simulate(p, LQRController(), sc, Uf, aggressiveness=1.0).metrics
    assert hi["control_energy"] > lo["control_energy"]
    assert hi["rms_h_mm"] < lo["rms_h_mm"]


def test_supervisor_decisions_change_the_control():
    """Regression test for the original bug: strategy changes had no effect on control."""
    p, Uf = setup("uav_flap")
    sc = SCENARIOS["turbulence_patch"]["scenario"]
    fixed = simulate(p, LQRController(), sc, Uf)
    supervised = simulate(p, LQRController(), sc, Uf, RuleSupervisor())
    assert supervised.metrics["mode_switches"] > 0
    assert len(set(supervised.aggressiveness)) > 1
    assert not np.allclose(fixed.u, supervised.u)


@pytest.mark.parametrize("preset", PRESETS)
def test_neural_policy_stabilises_above_flutter(preset):
    if not policy_path(preset).exists():
        pytest.skip(f"no neural policy for {preset}")
    p, Uf = setup(preset)
    r = simulate(p, NeuralController(), SCENARIOS["discrete_gust"]["scenario"], Uf)
    assert not r.exceeded_limits
    assert not r.warnings


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("a", [0.0, 0.5, 1.0])
def test_neural_policy_is_quiet_at_rest(preset, a):
    """Regression test: an earlier policy chattered and kept a force offset at rest."""
    if not policy_path(preset).exists():
        pytest.skip(f"no neural policy for {preset}")
    p, Uf = setup(preset)
    calm = replace(QUIET, duration=2.0)
    r = simulate(p, NeuralController(), calm, Uf, aggressiveness=a)
    assert np.max(np.abs(r.u[-50:] / np.asarray(p.u_max))) < 0.01
    lqr = simulate(p, LQRController(), calm, Uf, aggressiveness=a).metrics["control_rate_rms"]
    assert r.metrics["control_rate_rms"] < 1.5 * lqr + 0.01


@pytest.mark.parametrize("preset", PRESETS)
def test_neural_policy_delay_margin(preset):
    """
    Regression test: a policy trained with perfect timing went unstable with one extra
    sample of delay. The linearised policy must be stable for every delay in its training
    range at balanced aggressiveness.
    """
    if not policy_path(preset).exists():
        pytest.skip(f"no neural policy for {preset}")
    p, Uf = setup(preset)
    pol = NeuralPolicy.load(policy_path(preset))
    lo, hi = pol.meta["delay_range"]
    for ratio in (0.9, 1.1):
        K = pol.linearize(ratio, 0.5)
        for delay in range(lo, hi + 1):
            assert closed_loop_radius(p, ratio * Uf, 0.005, K, delay) < 1.0, (ratio, delay)


def test_metrics_have_no_nan():
    p, Uf = setup("uav_flap")
    r = simulate(p, LQRController(), SCENARIOS["discrete_gust"]["scenario"], Uf)
    for k, v in r.metrics.items():
        if isinstance(v, float):
            assert np.isfinite(v), k
