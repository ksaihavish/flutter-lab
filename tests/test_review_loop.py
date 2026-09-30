"""The LLM's off-control-path roles (explainer, post-flight reviewer) and the verification gate; LLM transport mocked."""

import json

import numpy as np
import pytest

from flutter_ai.fault_tolerance.diagnostics import FTFeatures
from flutter_ai.fault_tolerance.explainer import Explainer, expected_faults, faithfulness
from flutter_ai.fault_tolerance.review import (EXPLAIN_CLASS_MIN, F1_TOL, FALSE_ALARM_TOL, alarms, check_hypothesis,
                                               explanation_scores, gate_decision, hypothesis_specs, review_flight,
                                               reviewer_schema)
from flutter_ai.fault_tolerance.simulate import FTDecision
from flutter_ai.supervisor.llm import LLMSupervisor, check_schema


class FakeLLM(LLMSupervisor):
    """The wrapper's real `ask` with canned model output instead of a backend."""

    backend = "fake"

    def __init__(self, answers, **kw):
        super().__init__(model="fake", **kw)
        self.answers = list(answers)
        self.prompts: list[tuple[str, str]] = []

    def _complete_with(self, system_prompt, user, schema):
        self.prompts.append((system_prompt, user))
        a = self.answers.pop(0)
        if isinstance(a, BaseException):
            raise a
        return a if isinstance(a, str) else json.dumps(a)


def test_check_schema_rejects_what_constrained_decoding_might_let_through():
    s = {"type": "object", "properties": {"mode": {"type": "string", "enum": ["a", "b"]}, "x": {"type": "number"}},
         "required": ["mode"], "additionalProperties": False}
    check_schema({"mode": "a", "x": 1.5}, s)
    for bad in ({"mode": "c"}, {"x": 1.0}, {"mode": "a", "y": 1}, {"mode": "a", "x": True}, {"mode": "a", "x": "1"}):
        with pytest.raises(ValueError):
            check_schema(bad, s)


def test_ask_returns_failures_with_their_reason_never_the_answer():
    schema = {"type": "object", "properties": {"k": {"type": "string", "enum": ["ok"]}}, "required": ["k"]}

    def task_check(d):
        raise ValueError("task check failed")

    llm = FakeLLM([{"k": "ok"}, "not json", {"k": "nope"}, {"k": "ok"}, TimeoutError("slow")], max_calls=5)
    assert llm.ask("s", "u", schema).data == {"k": "ok"}
    for check in (None, None, task_check, None):
        ans = llm.ask("s", "u", schema, check)
        assert ans.data is None and ans.error
    assert llm.ask("s", "u", schema).error.startswith("ValueError: LLM call budget")  # 6th call: over the budget


def _features(diagnosis, mode="nominal"):
    return FTFeatures(time=12.0, airspeed=51.0, airspeed_ratio=1.1, mode=mode, risk=0.1, energy_growth=1.0,
                      saturation_fraction=0.0, pitch_freq_hz=10.4, nominal_freq_hz=10.4,
                      residual_ratio={"h": 1.0, "alpha": 1.0, "beta": 7.0}, noise_ratio=1.0, loop_delay_ms=5.0,
                      diagnosis=diagnosis)


DIAG = {"time": 11.5, "window_s": 2.5, "probabilities": {"servo_slew": 0.9, "stiffness_loss": 0.01, "sensor_noise": 0.02,
                                        "turbulence_rise": 0.1},
        "prob_threshold": {c: 0.5 for c in ("servo_slew", "stiffness_loss", "sensor_noise", "turbulence_rise")},
        "confirmed": ["servo_slew"], "ood_score": 0.1, "ood_threshold": 0.2, "unknown": False}


def test_explainer_uses_a_faithful_answer_and_replaces_an_unfaithful_one():
    f = _features(DIAG)
    d = FTDecision(time=12.0, mode="cautious", reasoning="servo", source="guardian")
    assert expected_faults(DIAG) == {"servo_slew"}
    good = {"explanation": "The flap servo has slowed down, so the aircraft slows to 90 % of flutter speed.",
            "fault": "servo_slew", "mode": "cautious"}
    wrong_mode = dict(good, mode="safe")
    wrong_fault = dict(good, fault="sensor_noise")
    names_unflagged = dict(good, explanation="Looks like loop_delay to me, slowing down now.")
    ex = Explainer(llm=FakeLLM([good, wrong_mode, wrong_fault, names_unflagged, "garbage"]))
    e = ex.explain(f, d)
    assert e.source == "llm" and e.mode == "cautious" and e.fault == "servo_slew"
    for _ in range(4):
        e = ex.explain(f, d)
        assert e.source == "template" and e.rejected and e.mode == "cautious" and "cautious" in e.text
    assert faithfulness(good, f, d) == []


HYP = {"kind": "flap_effectiveness", "severity_min": 0.8, "severity_max": 1.2, "growth_min_s": 2.0,
       "growth_max_s": 8.0, "evidence": "flap residual x7 while the flap saturates; flap effectiveness 0.4"}


def test_hypothesis_ranges_are_checked_and_drive_the_training_flights():
    check_hypothesis(HYP)
    for bad in (dict(HYP, severity_min=1.3), dict(HYP, growth_max_s=40.0), dict(HYP, evidence="x")):
        with pytest.raises(ValueError):
            check_hypothesis(bad)
    specs = hypothesis_specs(HYP, 50000, 40, "review_train")
    faults = [f for s in specs for f in s["faults"]]
    assert faults and all(f["kind"] == "flap_effectiveness" for f in faults)
    assert all(2.0 <= f["ramp"] <= 8.0 and 0.8 * 1.5 <= f["severity"] <= 1.2 * 1.5 for f in faults)
    assert any(not s["faults"] for s in specs)  # healthy flights in every batch
    assert set(reviewer_schema()["properties"]["kind"]["enum"]) >= {"flap_effectiveness", "loop_delay", "servo_slew"}


def test_review_flight_records_the_hypothesis_and_whether_it_was_right():
    fl = {"spec": {"seed": 45000, "faults": [{"kind": "flap_effectiveness", "onset": 8.0}]}, "cruise": 51.0,
          "flutter_speed": 46.4, "failed": False, "failure_time_s": None, "t_end": 30.0, "switches": [],
          "first_unknown": 12.0, "ticks": []}
    rv = review_flight(FakeLLM([HYP]), fl)
    assert rv["right_kind"] is True and rv["hypothesis"]["kind"] == "flap_effectiveness"
    rv = review_flight(FakeLLM([dict(HYP, severity_max=9.0)]), fl)
    assert rv["hypothesis"] is None and rv["right_kind"] is None and "severity" in rv["error"]


class StubModel:
    """predict() -> fixed probabilities / reconstruction scores per window."""

    def __init__(self, classes, prob, rec):
        self.classes, self._prob, self._rec = tuple(classes), np.asarray(prob, float), np.asarray(rec, float)
        self.prob_threshold, self.ood_threshold = np.full(len(classes), 0.5), 0.2

    def predict(self, X):
        return self._prob, self._rec, np.zeros(len(self._rec))


def test_gate_accepts_only_without_regression_and_with_the_anomaly_explained():
    v1 = {"f1": {"servo_slew": 0.6, "sensor_noise": 1.0}, "healthy_any_alarm_rate": 0.10}
    good = {"f1": {"servo_slew": 0.6 - F1_TOL + 0.01, "sensor_noise": 0.99}, "healthy_any_alarm_rate": 0.10}
    explained = {"windows": 10, "class_share": EXPLAIN_CLASS_MIN, "unknown_share": 0.0}
    assert gate_decision(v1, good, explained)["accepted"]
    regressed = dict(good, f1={"servo_slew": 0.6 - F1_TOL - 0.01, "sensor_noise": 0.99})
    noisy = dict(good, healthy_any_alarm_rate=0.10 + FALSE_ALARM_TOL + 0.01)
    unexplained = dict(explained, class_share=0.2)
    still_unknown = dict(explained, unknown_share=0.5)
    for cand, expl in ((regressed, explained), (noisy, explained), (good, unexplained), (good, still_unknown)):
        g = gate_decision(v1, cand, expl)
        assert not g["accepted"] and len(g["reasons"]) == 1


def test_explanation_scores_count_windows_from_the_first_unknown_alarm():
    # 6 windows: the new class fires from window 2 on, the unknown alarm on windows 1-2 (confirmation needs 2 in a row)
    prob = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.9], [0.0, 0.9], [0.0, 0.9], [0.0, 0.9]])
    rec = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0])
    m = StubModel(("servo_slew", "flap_effectiveness"), prob, rec)
    a = alarms(m, np.zeros((6, 1)))
    assert a[:, -1].tolist() == [False, False, True, False, False, False]  # unknown confirmed at window 2 only
    anomaly = {"t": np.arange(6.0), "X": np.zeros((6, 1)), "first_unknown": 2.0}
    s = explanation_scores(m, "flap_effectiveness", [anomaly])
    assert s["windows"] == 4 and s["class_share"] == 0.75 and s["unknown_share"] == 0.25
    assert explanation_scores(m, "loop_delay", [anomaly])["class_share"] == 0.0  # a class the model does not have
