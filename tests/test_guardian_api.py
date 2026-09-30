"""Guardian live view API: validation, background flights with progress polling, and what a recording contains."""

import time

import pytest
from fastapi.testclient import TestClient

from flutter_ai.api.app import app
from flutter_ai.fault_tolerance.faults import Fault, plant_at
from flutter_ai.fault_tolerance.recording import true_flutter_speed
from flutter_ai.fault_tolerance.simulate import MODE_ORDER
from flutter_ai.flutter import flutter_analysis
from flutter_ai.params import get_preset

client = TestClient(app)


def _fly(body: dict, timeout_s: float = 600.0) -> tuple[dict, dict]:
    """Start a flight, poll until it is done; returns (first POST response, final status)."""
    r = client.post("/api/guardian/flights", json=body)
    assert r.status_code == 202, r.text
    started = r.json()
    deadline = time.time() + timeout_s
    while True:
        s = client.get(f"/api/guardian/flights/{started['id']}")
        assert s.status_code == 200, s.text
        status = s.json()
        assert 0.0 <= status["progress"] <= 1.0
        if status["status"] == "done":
            return started, status
        assert time.time() < deadline, f"flight did not finish in {timeout_s} s: {status}"
        time.sleep(0.25)


def test_options_list_faults_supervisors_and_no_llm():
    o = client.get("/api/guardian/options").json()
    assert "none" in o["faults"] and "stiffness_loss" in o["faults"]
    assert set(o["supervisors"]) == {"none", "rules", "rules+physics", "guardian"}  # no LLM in the live view
    assert o["defaults"]["supervisor"] == "guardian" and o["defaults"]["estimator"] is True
    assert o["supervisors"]["guardian"]["needs_estimator"] is True
    assert o["mode_order"] == list(MODE_ORDER)
    assert o["presets"][0]["name"] == "uav_flap" and o["presets"][0]["flutter_speed"] > 0


@pytest.mark.parametrize("body, code", [
    ({"fault": "wing_fell_off"}, 422),
    ({"supervisor": "rules+physics", "estimator": False}, 422),
    ({"supervisor": "llm"}, 422),
    ({"onset": 3.0}, 422),  # inside the diagnostics' healthy calibration window
    ({"onset": 10.0, "duration": 8.0}, 422),
    ({"preset": "uav_section"}, 422),  # the study is calibrated for uav_flap only
    ({"preset": "nope"}, 404),
])
def test_invalid_requests_are_rejected(body, code):
    r = client.post("/api/guardian/flights", json=body)
    assert r.status_code == code, r.text
    assert "detail" in r.json()


def test_unknown_flight_404():
    assert client.get("/api/guardian/flights/0123456789abcdef").status_code == 404


def test_physics_supervised_flight_records_everything_the_panel_needs():
    body = {"fault": "stiffness_loss", "onset": 5.0, "ramp": 0.0, "duration": 7.0, "seed": 1000,
            "supervisor": "rules+physics", "max_points": 500}
    started, status = _fly(body)
    res = status["result"]
    s = res["series"]
    n = len(s["t"])
    assert 100 < n <= 501
    for key in ("h", "alpha", "beta", "airspeed", "mode", "fault_level"):
        assert len(s[key]) == n, key
    assert len(s["u"][0]) == n
    assert s["t"][-1] == pytest.approx(res["failure_time"] if res["failed"] else 7.0)
    assert set(s["mode"]) <= set(MODE_ORDER)
    assert s["fault_level"][0] == 0.0 and s["fault_level"][-1] == pytest.approx(res["fault"]["severity"])

    # every supervisor tick: the report an LLM would read, with the physics block once the estimator has run
    ticks = res["ticks"]
    assert ticks and ticks[0]["time"] == pytest.approx(4.5, abs=0.006)
    assert all(t["report"].startswith("FLUTTER SUPPRESSION STATUS REPORT") for t in ticks)
    assert "PHYSICS ESTIMATE" in ticks[-1]["report"]
    assert all(t["decision"] is None or 0 <= t["decision"] < len(res["decisions"]) for t in ticks)
    assert all(d["source"] == "rules+physics" and d["outcome"] in ("switched", "kept", "blocked", "pending")
               for d in res["decisions"])

    # on-board estimates (1 Hz) with a predicted stability for every mode
    est = res["estimates"]
    assert [e["time"] for e in est] == pytest.approx([4.5, 5.5, 6.5], abs=0.006)  # within one 5 ms sample
    assert all(set(e["mode_spectral_radius"]) == set(MODE_ORDER) for e in est)

    # ground truth (simulation only): the damaged wing flutters earlier, and the number is flutter_analysis's
    truth = res["truth"]
    assert truth["t"][0] == 0.0 and truth["flutter_speed"][0] == pytest.approx(res["flutter_speed"])
    assert truth["flutter_speed"][-1] < res["flutter_speed"]
    p = get_preset("uav_flap")
    damaged = plant_at(p, Fault("stiffness_loss", onset=5.0, ramp=0.0, severity=res["fault"]["severity"]), 6.0)
    assert truth["flutter_speed"][-1] == pytest.approx(flutter_analysis(damaged).flutter_speed, abs=1e-9)

    # identical request: the same flight, already finished
    again = client.post("/api/guardian/flights", json=body)
    assert again.status_code == 202 and again.json()["id"] == started["id"] and again.json()["status"] == "done"


def test_guardian_flight_is_the_benchmark_flight_with_diagnosis_and_verdicts():
    from flutter_ai.fault_tolerance.benchmark import _make, _sim_options
    from flutter_ai.fault_tolerance.recording import healthy_flutter
    from flutter_ai.fault_tolerance.simulate import ft_simulate

    body = {"fault": "servo_slew", "onset": 5.0, "ramp": 0.0, "duration": 9.0, "seed": 1001,
            "supervisor": "guardian", "max_points": 300}
    _, status = _fly(body)
    res = status["result"]
    assert res["estimator_version"] == 2 and res["diagnoser_every_s"] == pytest.approx(0.5)
    assert all(d["source"] == "guardian" for d in res["decisions"])
    # the learned diagnosis the guardian read, on the ticks after the diagnoser's first window
    assert any(t["diagnosis"] is not None for t in res["ticks"])
    # estimator v2, each estimate with the guardian's own reading of it (physics_verdict)
    est = res["estimates"]
    assert est and all(e["version"] == 2 and set(e["verdict"]["admissible"]) == set(MODE_ORDER) for e in est)
    assert all(set(e["verdict"]["why"]) <= set(e["verdict"]["admissible"]) | {"all"} for e in est)

    # recording does not change the flight: the benchmark's ft_simulate with the same options flies the same mission
    p = get_preset("uav_flap")
    Uf, f_hz = healthy_flutter(p)
    r = ft_simulate(p, Uf, f_hz, Fault("servo_slew", onset=5.0, ramp=0.0, severity=res["fault"]["severity"]),
                    _make("guardian", 1001), duration=9.0, seed=1001, **_sim_options("guardian"))
    assert r.failed == res["failed"]
    assert [(round(t, 6), m) for t, m, _ in r.switch_log] == [(round(s["time"], 6), s["mode"]) for s in res["switches"]]


def test_unsupervised_flight_without_estimator():
    _, status = _fly({"fault": "none", "supervisor": "none", "estimator": False, "duration": 6.0, "max_points": 300})
    res = status["result"]
    assert res["decisions"] == [] and res["estimates"] == [] and res["switches"] == []
    assert res["ticks"] and all("PHYSICS ESTIMATE" not in t["report"] for t in res["ticks"])
    assert set(res["series"]["mode"]) == {"nominal"}
    assert all(v == pytest.approx(res["flutter_speed"]) for v in res["truth"]["flutter_speed"])
    assert res["estimator_every_s"] is None and not res["failed"]


def test_fast_true_flutter_speed_matches_flutter_analysis():
    p = get_preset("uav_flap")
    q = plant_at(p, Fault("flap_effectiveness", onset=5.0, ramp=0.0, severity=1.5), 6.0)
    assert true_flutter_speed(q) == pytest.approx(flutter_analysis(q).flutter_speed, abs=1e-9)
