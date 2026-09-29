"""Physics-informed supervision: the report's physics block, the physics rule veto, latency injection, the time-scale
study and the LLM variant's wiring."""

import json

import pytest

from flutter_ai.fault_tolerance.diagnostics import FTFeatures, format_ft_report
from flutter_ai.fault_tolerance.faults import benchmark_fault
from flutter_ai.fault_tolerance.simulate import FTDecision, ft_simulate
from flutter_ai.fault_tolerance.supervisors import Delayed, FTPhysicsRuleSupervisor
from flutter_ai.flutter import flutter_analysis
from flutter_ai.params import get_preset


@pytest.fixture(scope="module")
def wing():
    p = get_preset("uav_flap")
    fa = flutter_analysis(p)
    return p, fa.flutter_speed, fa.flutter_frequency_hz


def _features(margin=None, mode="nominal", **kw):
    base = dict(time=10.0, airspeed=51.0, airspeed_ratio=1.1, mode=mode, risk=0.1, energy_growth=1.0,
                saturation_fraction=0.0, pitch_freq_hz=9.0, nominal_freq_hz=10.0,
                residual_ratio={"h": 1.0, "alpha": 1.0, "beta": 1.0}, noise_ratio=1.0, loop_delay_ms=5.0)
    base.update(kw)
    return FTFeatures(**base, margin=margin)


def _margin(unstable=()):
    return {"time": 9.0, "airspeed": 51.0, "params": {"f_alpha": 0.85, "flap_span_fraction": 1.0},
            "params_std": {}, "flutter_speed": 37.0, "flutter_speed_std": 0.2, "margin": 37.0 / 51.0 - 1.0,
            "mode_spectral_radius": {m: (1.02 if m in unstable else 0.95) for m in ("nominal", "robust", "cautious", "safe")},
            "converged": True}


def test_physics_rules_never_choose_a_mode_predicted_unstable():
    sup = FTPhysicsRuleSupervisor()
    # symptom rules alone would slow down moderately (noise high); physics says that is unstable -> hold speed
    d = sup.decide(_features(_margin(unstable=("cautious", "robust")), noise_ratio=4.0))
    assert d.mode == "nominal" and "unstable" in d.reasoning
    # current mode predicted unstable and no rule fires -> least disruptive predicted-stable mode
    assert sup.decide(_features(_margin(unstable=("nominal", "robust", "cautious")))).mode == "safe"
    # no physics estimate -> exactly the symptom rules
    assert sup.decide(_features(None, noise_ratio=4.0)).mode == "cautious"


def test_report_has_a_physics_block_only_with_an_estimate():
    assert "PHYSICS ESTIMATE" not in format_ft_report(_features())
    text = format_ft_report(_features(_margin(unstable=("cautious",))))
    assert "PHYSICS ESTIMATE" in text and "cautious UNSTABLE" in text and "37.0 m/s" in text


def test_on_board_estimator_feeds_the_supervisor(wing):
    p, Uf, fhz = wing
    seen = []

    class Probe:
        name = "probe"

        def decide(self, f):
            seen.append(f.margin)
            return None

    ft_simulate(p, Uf, fhz, benchmark_fault("none"), Probe(), duration=7.0, seed=1000, margin_every_s=1.0,
                supervisor_interval=1.0)
    assert seen and seen[-1] is not None
    assert seen[-1]["flutter_speed"] == pytest.approx(Uf, rel=0.05)


def test_delayed_answers_arrive_late(wing):
    p, Uf, fhz = wing

    class Once:
        name = "once"

        def decide(self, f):
            return FTDecision(time=f.time, mode="safe", reasoning="x", source="test")

    r = ft_simulate(p, Uf, fhz, benchmark_fault("none"), Delayed(Once(), 3.0), duration=10.0, seed=0)
    t_switch, _, decided = r.switch_log[0]
    assert t_switch - decided == pytest.approx(3.0, abs=0.01)


def test_timescale_study_writes_a_grid(tmp_path):
    from flutter_ai.fault_tolerance.timescale import run_timescale

    res = run_timescale(get_preset("uav_flap"), faults=("stiffness_loss",), ramps=(4.0,), latencies=(0.0, 3.0),
                        seeds=(200,), supervisors=("rules",), out_dir=tmp_path, workers=2, log=lambda m: None)
    cell = res["survival"]["stiffness_loss"]["4"]
    assert set(cell) == {"oracle", "rules@0", "rules@3"}
    assert json.loads((tmp_path / "ft_timescale.json").read_text())["metadata"]["seeds"] == [200]
    assert (tmp_path / "fig13_timescale.png").exists()


def test_llm_physics_variant_gets_the_guidance_and_the_physics_fallback():
    from flutter_ai.fault_tolerance.llm import make_ft_llm_supervisor

    sup = make_ft_llm_supervisor("ollama", physics=True)
    assert sup.name == "llm+physics"
    assert "PHYSICS ESTIMATE" in sup.system_prompt and sup.system_prompt.rstrip().endswith("cite the report.")
    assert isinstance(sup.fallback, FTPhysicsRuleSupervisor)
    assert "PHYSICS ESTIMATE" not in make_ft_llm_supervisor("ollama").system_prompt
