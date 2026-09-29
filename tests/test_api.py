"""HTTP API smoke tests."""

from fastapi.testclient import TestClient

from flutter_ai.api.app import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_presets_and_scenarios():
    presets = client.get("/api/presets").json()
    assert any(p["name"] == "uav_section" and p["flutter_speed"] > 0 for p in presets)
    assert "turbulence_patch" in client.get("/api/scenarios").json()


def test_simulate_returns_time_series_and_metrics():
    body = {"cases": [{"controller": "open_loop"}, {"controller": "lqr", "supervisor": "rules"}],
            "scenario": {"duration": 3.0}}
    r = client.post("/api/simulate", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["results"]) == 2
    res = data["results"][1]
    assert len(res["t"]) == len(res["h"]) > 10
    assert "rms_h_mm" in res["metrics"]


def test_supervisor_on_open_loop_rejected():
    r = client.post("/api/simulate", json={"cases": [{"controller": "open_loop", "supervisor": "rules"}]})
    assert r.status_code == 422


def test_validation_only_preset_not_simulated():
    assert all(p["name"] != "hodges_pierce" for p in client.get("/api/presets").json())
    r = client.post("/api/simulate", json={"preset": "hodges_pierce", "cases": [{"controller": "lqr"}]})
    assert r.status_code == 422 and "validation-only" in r.json()["detail"]
    assert client.post("/api/flutter", json={"preset": "hodges_pierce"}).status_code == 200


def test_unknown_preset_404():
    assert client.post("/api/flutter", json={"preset": "nope"}).status_code == 404


def test_frontend_served():
    r = client.get("/")
    assert r.status_code == 200 and "Flutter Lab" in r.text
