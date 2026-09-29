"""Supervisor safety: the LLM can only pick a valid mode, and failures fall back to rules."""

import json
from types import SimpleNamespace

from flutter_ai.supervisor import (
    MODES,
    ClaudeSupervisor,
    OllamaSupervisor,
    RuleSupervisor,
    SituationFeatures,
    format_report,
)


def features(**kw):
    base = dict(time=1.0, airspeed=54.0, airspeed_ratio=1.1, h_mm=1.0, alpha_deg=0.1, peak_h_ratio=0.01,
                peak_alpha_ratio=0.01, risk=0.01, energy=0.01, window_energy=0.01, energy_growth=1.0,
                saturation_fraction=0.0, control_effort=0.05, current_mode="balanced")
    base.update(kw)
    return SituationFeatures(**base)


def fake_client(text=None, stop_reason="end_turn", exc=None):
    def create(**kwargs):
        if exc:
            raise exc
        return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


def test_rules_escalate_on_high_risk_immediately():
    d = RuleSupervisor().decide(features(risk=0.7, peak_h_ratio=0.7))
    assert d.mode == "emergency"


def test_rules_deescalate_when_calm():
    sup = RuleSupervisor()
    d = sup.decide(features(time=5.0))
    assert d.mode == "efficient"


def test_report_contains_key_information():
    text = format_report(features(risk=0.3, energy_growth=2.0))
    assert "INCREASING" in text and "RISK LEVEL" in text


def test_llm_valid_response_is_used():
    sup = ClaudeSupervisor(client=fake_client(json.dumps({"mode": "aggressive", "reasoning": "energy growing"})))
    d = sup.decide(features())
    assert d.mode == "aggressive" and d.source == "llm"
    assert d.aggressiveness == MODES["aggressive"]["aggressiveness"]


def test_llm_invalid_mode_falls_back_to_rules():
    sup = ClaudeSupervisor(client=fake_client(json.dumps({"mode": "max_power", "reasoning": "x"})))
    d = sup.decide(features())
    assert d.source == "llm_fallback" and d.mode in MODES


def test_llm_refusal_and_bad_json_fall_back():
    for client in (fake_client("not json"), fake_client("{}", stop_reason="refusal")):
        d = ClaudeSupervisor(client=client).decide(features())
        assert d.source == "llm_fallback"


def test_llm_call_budget_is_enforced():
    sup = ClaudeSupervisor(client=fake_client(json.dumps({"mode": "balanced", "reasoning": "ok"})), max_calls=2)
    sources = [sup.decide(features(time=t)).source for t in range(4)]
    assert sources == ["llm", "llm", "llm_fallback", "llm_fallback"]


def fake_ollama(text=None, exc=None, seen=None):
    def transport(body):
        if seen is not None:
            seen.append(body)
        if exc:
            raise exc
        return {"message": {"role": "assistant", "content": text}}

    return transport


def test_ollama_valid_response_uses_schema_and_is_used():
    seen = []
    sup = OllamaSupervisor(model="test-model", transport=fake_ollama(
        json.dumps({"mode": "emergency", "reasoning": "near limit"}), seen=seen))
    d = sup.decide(features(risk=0.7))
    assert d.mode == "emergency" and d.source == "llm"
    assert seen[0]["format"]["properties"]["mode"]["enum"] == list(MODES)  # constrained decoding to the enum
    assert seen[0]["options"]["temperature"] == 0


def test_ollama_failures_fall_back_to_rules():
    import urllib.error

    for transport in (fake_ollama(exc=urllib.error.URLError("connection refused")), fake_ollama(exc=TimeoutError()),
                      fake_ollama("not json"), fake_ollama(json.dumps({"mode": "max_power", "reasoning": "x"}))):
        d = OllamaSupervisor(transport=transport).decide(features())
        assert d.source == "llm_fallback" and d.mode in MODES


def test_slow_supervisor_is_asked_one_question_at_a_time():
    """With latency longer than the interval, answers must arrive in order and never overlap."""
    from flutter_ai.controllers import LQRController
    from flutter_ai.flutter import flutter_speed
    from flutter_ai.gusts import GustSpec
    from flutter_ai.params import get_preset
    from flutter_ai.simulation import Scenario, simulate

    class Slow(RuleSupervisor):
        name = "llm"

        def decide(self, f):
            d = super().decide(f)
            d.source, d.latency_s = "llm", 1.0  # 1 s answers, asked every 0.25 s
            return d

    p = get_preset("uav_section")
    sc = Scenario(velocity_ratio=1.1, duration=4.0, gust=GustSpec(kind="none"), supervisor_interval=0.25)
    r = simulate(p, LQRController(), sc, flutter_speed(p), Slow())
    asked = [d.time for d in r.decisions]
    landed = [d.time + d.latency_s for d in r.decisions]
    assert all(nxt >= prev_landed - 1e-9 for nxt, prev_landed in zip(asked[1:], landed))
    assert len(r.decisions) <= 4


def test_slow_answer_counts_as_timeout_and_costs_exactly_the_timeout():
    """Socket timeouts only bound silences; an answer slower than timeout_s overall must still fall back."""
    import time as _time

    def slow(body):
        _time.sleep(0.3)
        return {"message": {"content": json.dumps({"mode": "emergency", "reasoning": "late"})}}

    d = OllamaSupervisor(transport=slow, timeout_s=0.1).decide(features())
    assert d.source == "llm_fallback" and "TimeoutError" in d.reasoning
    assert d.latency_s == 0.1
