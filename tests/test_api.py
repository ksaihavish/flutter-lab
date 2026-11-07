"""HTTP API smoke tests."""

from fastapi.testclient import TestClient

from flutter_ai.api.app import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    hashes = r.json()["model_hashes"]
    assert all(len(h) == 12 for h in hashes.values())
    assert any(name.startswith("neural_policy_") for name in hashes) or not hashes


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


def test_survival_check_seed0_is_the_run_shown(monkeypatch):
    """The multi-seed check's seed 0 is the flight /api/simulate shows; other seeds draw new disturbances."""
    monkeypatch.setenv("FLUTTER_AI_WORKERS", "0")  # in-process (a pool cannot re-import a pytest __main__)
    scenario = {"velocity_ratio": 1.1, "duration": 2.0, "gust": {"kind": "turbulence", "amplitude": 0.5, "start": 0.2,
                                                                  "duration": 1.5, "seed": 3}}
    cases = [{"controller": "open_loop"}, {"controller": "lqr"}]
    sim = client.post("/api/simulate", json={"cases": cases, "scenario": scenario}).json()
    r = client.post("/api/survival", json={"cases": cases, "scenario": scenario, "n_seeds": 3})
    assert r.status_code == 200, r.text
    rows = r.json()["results"]
    for row, res in zip(rows, sim["results"]):
        assert row["n_seeds"] == 3 and len(row["survived_per_seed"]) == 3
        assert row["survived"] == sum(row["survived_per_seed"])
        assert row["survived_per_seed"][0] == (not res["metrics"]["exceeded_limits"])
    assert rows[0]["survived"] == 0  # open loop above flutter fails every draw
    llm = client.post("/api/survival", json={"cases": [{"controller": "lqr", "supervisor": "llm"}], "n_seeds": 2})
    assert llm.status_code == 422


def test_presets_report_neural_training_range():
    flap = next(p for p in client.get("/api/presets").json() if p["name"] == "uav_flap")
    if flap["neural_policy_available"]:
        lo, hi = flap["neural_training_range"]
        assert 0 < lo < 1.0 < hi
    assert flap["divergence_speed"] > flap["flutter_speed"]


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


def test_paper_draft_is_generated_from_result_files(tmp_path):
    from flutter_ai.paper import build

    path = build(tmp_path / "draft.md", copy_figures=False)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# ") and "## Abstract" in text
    assert "## 7. Results" in text and "### 7.5 Fault-tolerant supervision" in text and "## References" in text
    assert "(Hespanha & Morse 1999)" in text or "ft_stability" not in text  # citations render as author-year


def test_paper_markdown_renders_to_html(tmp_path):
    from flutter_ai.paper_render import markdown_to_html

    md = ("# T & a\n\n*note*\n\n## 1. S\n\nText with **bold**, `code` and 5 < 6.\n\n- a\n  - b\n- c\n\n"
          "| x | y |\n|---|---|\n| 1 | 2 |\n\n![Figure 1](missing.png)\n")
    page = markdown_to_html(md, tmp_path)
    assert "<h1>T &amp; a</h1>" in page and "<em>note</em>" in page and "<h2>1. S</h2>" in page
    assert "<strong>bold</strong>" in page and "<code>code</code>" in page and "5 &lt; 6" in page
    assert page.count("<ul>") == 2 and page.count("</ul>") == 2
    assert "<th>x</th>" in page and "<td>2</td>" in page and '<img src="missing.png"' in page


def test_fault_tolerance_discussion_is_computed_from_results():
    from flutter_ai.paper import _conclusion, _ft_discussion

    def sup(surv, slowed=0.0):
        return {"survival": surv, "slowed_s_mean": slowed}

    def fault(held, llm, rules, oracle, fixed=None):
        d = {"held_out": held, "supervisors": {"none": sup(0.0), "rules": sup(rules), "llm": sup(llm),
                                               "oracle": sup(oracle)}}
        if fixed:
            d["fixed_modes"] = fixed
        return d

    ft = {"faults": {"none": fault(False, 1.0, 1.0, 1.0),
                     "servo_slew": fault(False, 0.0, 0.0, 1.0),
                     "stiffness_loss": fault(False, 1.0, 0.5, 1.0, {"nominal": sup(1.0), "cautious": sup(0.0)}),
                     "loop_delay": fault(True, 1.0, 0.0, 1.0)},
          "pooled_faulty": {"supervisors": {"none": sup(0.0), "rules": sup(1 / 6), "llm": sup(2 / 3), "oracle": sup(1.0)},
                            "paired": {"llm vs rules": {"mcnemar_p": 0.03}}}}
    text = "\n".join(_ft_discussion(ft))
    assert "LLM survived 67 % and the rules 17 % (significant" in text
    assert "structural-damage trap" in text and "servo_slew" in text  # nobody caught the servo fault
    assert "loop_delay: LLM 100 %" in text
    assert "did better than hand-written rules" in _conclusion(None, None, ft, None)
    assert _ft_discussion(None)[0].startswith("- **Fault tolerance:** *[results pending")


def test_paired_sentences_are_computed_from_results():
    from flutter_ai.paper import _changed_faults, _pair_sentence

    ft = {"faults": {"none": {"supervisors": {"a": {"survival": 1.0}, "b": {"survival": 1.0}}},
                     "servo_slew": {"supervisors": {"a": {"survival": 0.5}, "b": {"survival": 0.25}}},
                     "sensor_noise": {"supervisors": {"a": {"survival": 1.0}, "b": {"survival": 1.0}}}},
          "pooled_faulty": {"supervisors": {"a": {"survival": 0.75}, "b": {"survival": 0.625}},
                            "paired": {"a vs b": {"survived_only_candidate": 5, "survived_only_reference": 0,
                                                  "mcnemar_p": 0.0625}}}}
    s = _pair_sentence(ft, "a", "b")
    assert s is not None and s.startswith("a 75 % vs b 62 %") and "not significant" in s and "only with a 5" in s
    assert _pair_sentence(ft, "b", "a") is None  # only pairs the benchmark computed
    assert _changed_faults(ft, "a", "b") == "servo_slew 50 % vs 25 %"
