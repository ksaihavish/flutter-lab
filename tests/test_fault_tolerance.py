"""Fault-tolerant supervision: faults, diagnostics, the controller bank, dwell time, stability analysis, LLM wiring."""

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from flutter_ai.controllers import LQGController
from flutter_ai.fault_tolerance.faults import FAULTS, Fault, benchmark_fault, delay_at, plant_at
from flutter_ai.fault_tolerance.simulate import (
    FT_MODES, MIN_DWELL_S, MODE_ORDER, SPEED_STEP, FTDecision, controller_of, ft_simulate,
)
from flutter_ai.fault_tolerance.stability import dwell_time_analysis, mode_matrices
from flutter_ai.fault_tolerance.supervisors import FTRuleSupervisor
from flutter_ai.flutter import flutter_analysis
from flutter_ai.gusts import GustSpec
from flutter_ai.params import get_preset
from flutter_ai.simulation import Scenario, simulate


@pytest.fixture(scope="module")
def wing():
    p = get_preset("uav_flap")
    fa = flutter_analysis(p)
    return p, fa.flutter_speed, fa.flutter_frequency_hz


def test_fault_ramps_and_effects():
    f = Fault("stiffness_loss", onset=5.0, ramp=5.0, severity=1.0)
    assert f.level(4.9) == 0.0 and f.level(7.5) == pytest.approx(0.5) and f.level(20.0) == 1.0
    p = get_preset("uav_flap")
    assert plant_at(p, f, 20.0).f_alpha == pytest.approx(p.f_alpha * math.sqrt(0.84))
    assert plant_at(p, f, 1.0) is p  # untouched before onset
    assert delay_at(Fault("loop_delay"), 20.0) == 3 and delay_at(Fault("loop_delay"), 0.0) == 1
    assert {k for k, v in FAULTS.items() if v["held_out"]} == {"loop_delay", "flap_effectiveness"}


def test_healthy_flight_survives_without_false_alarms(wing):
    p, Uf, fhz = wing
    r = ft_simulate(p, Uf, fhz, benchmark_fault("none"), FTRuleSupervisor(), duration=12.0, seed=1)
    assert not r.failed and r.metrics["slowed_s"] == 0.0


def test_forced_mode_slows_the_aircraft_and_uses_its_controller(wing):
    p, Uf, fhz = wing
    r = ft_simulate(p, Uf, fhz, benchmark_fault("none"), None, duration=12.0, forced_mode=(2.0, "safe"))
    assert r.airspeed[-1] < 0.75 * Uf and r.mode[-1] == "safe"
    assert FT_MODES["safe"]["controller"] == "lqg"


def test_dwell_time_is_enforced_between_switches(wing):
    p, Uf, fhz = wing

    class Flipper:
        name = "flipper"

        def decide(self, f):
            other = "robust" if f.mode == "nominal" else "nominal"
            return FTDecision(time=f.time, mode=other, reasoning="flip", source="test")

    r = ft_simulate(p, Uf, fhz, benchmark_fault("none"), Flipper(), duration=25.0, seed=0)
    switch_times = [r.t[k] for k in range(1, len(r.mode)) if r.mode[k] != r.mode[k - 1]]
    assert len(switch_times) >= 2
    assert np.min(np.diff(switch_times)) >= MIN_DWELL_S - 1e-9
    # answers inside the dwell time are dropped (and the supervisor asked again), never applied late
    assert any(d.outcome == "blocked" for d in r.decisions)
    assert all(d.applied_at == pytest.approx(d.time) for d in r.decisions if d.outcome == "switched")


def test_speed_only_mode_changes_are_not_delayed(wing):
    """The dwell time protects controller switches; nominal -> cautious -> safe all keep LQG."""
    p, Uf, fhz = wing
    assert controller_of("nominal") == controller_of("cautious") == controller_of("safe")

    class Escalate:
        name = "escalate"

        def decide(self, f):
            nxt = {"nominal": "cautious", "cautious": "safe"}.get(f.mode, f.mode)
            return FTDecision(time=f.time, mode=nxt, reasoning="escalate", source="test")

    r = ft_simulate(p, Uf, fhz, benchmark_fault("none"), Escalate(), duration=8.0, seed=0)
    assert [m for _, m, _ in r.switch_log] == ["cautious", "safe"]
    assert r.switch_log[1][0] - r.switch_log[0][0] < 1.0


def test_reaction_counts_only_changes_decided_after_the_onset(wing):
    p, Uf, fhz = wing

    class Early:  # a false alarm before the fault, then "keep" forever
        name = "early"

        def decide(self, f):
            return FTDecision(time=f.time, mode="cautious", reasoning="x", source="test")

    fault = benchmark_fault("none")
    r = ft_simulate(p, Uf, fhz, fault, Early(), duration=10.0, seed=0)
    assert r.switch_log and r.switch_log[0][2] < fault.onset
    assert r.metrics["reaction_s"] is None and r.metrics["first_response"] is None

    class Late:
        name = "late"

        def decide(self, f):
            return FTDecision(time=f.time, mode="safe" if f.time >= 7.0 else f.mode, reasoning="x", source="test")

    r = ft_simulate(p, Uf, fhz, fault, Late(), duration=10.0, seed=0)
    assert r.metrics["first_response"] == "safe"
    assert r.metrics["reaction_s"] == pytest.approx(7.0 - fault.onset, abs=0.01)


def test_servo_fault_needs_a_rate_limit():
    p = replace(get_preset("uav_flap"), u_rate_max=None)
    with pytest.raises(ValueError, match="slew-rate limit"):
        plant_at(p, benchmark_fault("servo_slew"), 20.0)


def test_lqg_mode_matrix_reproduces_the_simulator():
    """The stability analysis is only valid if its matrices are the implemented controller."""
    p = replace(get_preset("uav_flap"), u_rate_max=None)  # small-signal: no rate limit
    Uf = flutter_analysis(p).flutter_speed
    U = 1.1 * Uf
    sc = Scenario(velocity_ratio=1.1, duration=1.0, h0=0.001, alpha0_deg=0.1, gust=GustSpec(kind="none"))
    r = simulate(p, LQGController(), sc, Uf)
    A = mode_matrices(p, U)["lqg"]
    N = r.x.shape[1]
    xi = np.zeros(A.shape[0])
    xi[:N] = r.x[0]
    for k in range(1, len(r.t)):
        xi = A @ xi
        assert np.allclose(xi[:N], r.x[k], rtol=1e-6, atol=1e-10), f"diverged at step {k}"


def test_mode_matrices_reproduce_the_fault_tolerance_simulator():
    """Both analysed modes against ft_simulate itself (bank, bumpless history, gain scheduling state)."""
    p = replace(get_preset("uav_flap"), u_rate_max=None)
    fa = flutter_analysis(p)
    Uf, fhz = fa.flutter_speed, fa.flutter_frequency_hz
    U_level = round(1.1 * Uf / SPEED_STEP) * SPEED_STEP

    def max_rel_error(mode, key, amp):
        r = ft_simulate(p, Uf, fhz, benchmark_fault("none"), None, duration=1.0, turbulence_rms=0.0,
                        initial_mode=mode, x0=(amp, 0.0))
        A = mode_matrices(p, U_level)[key]
        N = r.x.shape[1]
        xi, err = np.zeros(A.shape[0]), 0.0
        xi[:N] = r.x[0]
        for k in range(1, len(r.t)):
            xi = A @ xi
            err = max(err, float(np.max(np.abs(xi[:N] - r.x[k]) / (np.max(np.abs(r.x), axis=0) + 1e-15))))
        return err

    assert max_rel_error("nominal", "lqg", 1e-3) < 1e-9  # linear controller: exact
    # neural policy: the matrix is its linearisation, so the error must shrink with the amplitude
    big, small = max_rel_error("robust", "neural", 1e-3), max_rel_error("robust", "neural", 1e-4)
    assert small < 2e-3 and small < 0.2 * big


def test_enforced_dwell_covers_the_stability_bound():
    res = dwell_time_analysis(get_preset("uav_flap"))
    assert all(row["all_stable"] for row in res["speeds"])
    assert res["max_dwell_s"] <= MIN_DWELL_S


def test_llm_ft_supervisor_uses_ft_modes_and_falls_back_to_ft_rules(wing):
    from flutter_ai.fault_tolerance.llm import make_ft_llm_supervisor
    from flutter_ai.supervisor import OllamaSupervisor

    seen = []

    def transport(body):
        seen.append(body)
        return {"message": {"content": json.dumps({"mode": "cautious", "reasoning": "flap residual high"})}}

    p, Uf, fhz = wing
    sup = make_ft_llm_supervisor("ollama")
    assert isinstance(sup, OllamaSupervisor)
    sup._transport = transport
    r = ft_simulate(p, Uf, fhz, benchmark_fault("none"), sup, duration=8.0)
    body = [b for b in seen if b.get("options", {}).get("num_predict") is None][0]
    assert body["format"]["properties"]["mode"]["enum"] == list(MODE_ORDER)
    assert "MODEL CONSISTENCY" in body["messages"][1]["content"]
    assert r.decisions and r.decisions[0].mode == "cautious" and r.decisions[0].source == "llm"

    sup_bad = make_ft_llm_supervisor("ollama")
    sup_bad._transport = lambda body: {"message": {"content": json.dumps({"mode": "eject", "reasoning": "x"})}}
    r2 = ft_simulate(p, Uf, fhz, benchmark_fault("none"), sup_bad, duration=6.0)
    assert r2.decisions and all(d.source == "llm_fallback" for d in r2.decisions)


def test_paired_statistics_count_discordant_flights():
    from flutter_ai.fault_tolerance.benchmark import _summary, paired

    def rec(failed, slowed, final="nominal"):
        return {"failed": failed, "slowed_s": slowed, "cost_J": None if failed else 1.0, "reaction_s": None,
                "final_mode": final, "llm_fallbacks": 0}

    a = [rec(False, 5.0, "cautious"), rec(False, 5.0, "cautious"), rec(True, 0.0), rec(False, 0.0)]
    b = [rec(True, 0.0), rec(True, 0.0), rec(True, 0.0), rec(False, 0.0)]
    st = paired(a, b)
    assert st["survived_only_candidate"] == 2 and st["survived_only_reference"] == 0
    assert st["survival_diff"] == 0.5 and 0.0 < st["mcnemar_p"] <= 1.0
    # time slowed is compared only where both flights survived: a failure must not count as "not slowed"
    assert st["n_both_survived"] == 1 and st["slowed_s_mean_diff"] == 0.0
    summ = _summary(a, ["cautious", "cautious", "cautious", "nominal"])
    assert summ["survival"] == 0.75 and summ["oracle_agreement"] == 0.75
    assert summ["slowed_s_mean"] == pytest.approx(10.0 / 3.0)  # survivors only
    assert paired([rec(True, 0.0)], [rec(True, 0.0)])["slowed_s_mean_diff"] is None
