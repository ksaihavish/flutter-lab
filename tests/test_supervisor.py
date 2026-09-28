"""Supervisor safety: the LLM can only pick a valid mode, and failures fall back to rules."""

import json
from types import SimpleNamespace

from flutter_ai.supervisor import MODES, LLMSupervisor, RuleSupervisor, SituationFeatures, format_report


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
    sup = LLMSupervisor(client=fake_client(json.dumps({"mode": "aggressive", "reasoning": "energy growing"})))
    d = sup.decide(features())
    assert d.mode == "aggressive" and d.source == "llm"
    assert d.aggressiveness == MODES["aggressive"]["aggressiveness"]


def test_llm_invalid_mode_falls_back_to_rules():
    sup = LLMSupervisor(client=fake_client(json.dumps({"mode": "max_power", "reasoning": "x"})))
    d = sup.decide(features())
    assert d.source == "llm_fallback" and d.mode in MODES


def test_llm_refusal_and_bad_json_fall_back():
    for client in (fake_client("not json"), fake_client("{}", stop_reason="refusal")):
        d = LLMSupervisor(client=client).decide(features())
        assert d.source == "llm_fallback"


def test_llm_call_budget_is_enforced():
    sup = LLMSupervisor(client=fake_client(json.dumps({"mode": "balanced", "reasoning": "ok"})), max_calls=2)
    sources = [sup.decide(features(time=t)).source for t in range(4)]
    assert sources == ["llm", "llm", "llm_fallback", "llm_fallback"]
