"""GuardianSupervisor (fault_tolerance/supervisors.py): its decision logic on synthetic on-board features."""

from flutter_ai.fault_tolerance.diagnostics import FTFeatures
from flutter_ai.fault_tolerance.simulate import MODE_ORDER
from flutter_ai.fault_tolerance.supervisors import GuardianSupervisor, physics_verdict

UF = 46.4


def feats(t=10.0, mode="nominal", diagnosis=None, margin=None, pitch=10.4, **kw) -> FTFeatures:
    base = dict(time=t, airspeed=1.1 * UF, airspeed_ratio=1.1, mode=mode, risk=0.05, energy_growth=1.0,
                saturation_fraction=0.0, pitch_freq_hz=pitch, nominal_freq_hz=10.4,
                residual_ratio={"h": 1.0, "alpha": 1.0, "beta": 1.0}, noise_ratio=1.0, loop_delay_ms=5.0)
    base.update(kw)
    return FTFeatures(margin=margin, diagnosis=diagnosis, **base)


def diag(confirmed=(), unknown=False, probs=None) -> dict:
    classes = ("servo_slew", "stiffness_loss", "sensor_noise", "turbulence_rise")
    return {"time": 9.5, "confirmed": list(confirmed), "unknown": unknown,
            "probabilities": probs or {c: (0.9 if c in confirmed else 0.01) for c in classes},
            "prob_threshold": {c: 0.5 for c in classes}}


def margin_v2(t=9.0, stable=None, auth=None, needed=None, f_alpha=1.0, flap=1.0, uf=UF) -> dict:
    stable = stable or {}
    auth = auth or {}
    needed = needed or {"nominal": True, "robust": True, "cautious": False, "safe": False}
    return {"time": t, "flutter_speed": uf, "params": {"f_alpha": f_alpha, "flap_span_fraction": flap},
            "modes": {m: {"stable": stable.get(m, True), "authority_ratio": auth.get(m, 0.5),
                          "authority_ok": auth.get(m, 0.5) < 1.0 or not needed[m], "control_needed": needed[m]}
                      for m in MODE_ORDER}}


def test_nothing_wrong_keeps_the_mode():
    d = GuardianSupervisor().decide(feats(diagnosis=diag(), margin=margin_v2()))
    assert d.mode == "nominal" and d.source == "guardian"


def test_known_fault_gets_its_response():
    for fault in ("servo_slew", "sensor_noise", "turbulence_rise"):
        assert GuardianSupervisor().decide(feats(diagnosis=diag([fault]), margin=margin_v2())).mode == "cautious"


def test_unknown_fault_goes_conservative():
    assert GuardianSupervisor().decide(feats(diagnosis=diag(unknown=True), margin=margin_v2())).mode == "safe"


def test_physics_veto_skips_an_inadmissible_mode():
    m = margin_v2(stable={"cautious": False}, needed={"nominal": True, "robust": True, "cautious": True, "safe": False})
    d = GuardianSupervisor().decide(feats(diagnosis=diag(["servo_slew"]), margin=m))
    assert d.mode == "safe" and "cautious" in d.reasoning


def test_structural_damage_holds_speed_while_the_physics_allows_it():
    g = GuardianSupervisor()
    # the diagnosis flags stiffness loss at 10 s, but the latest physics estimate (9 s) predates it: wait
    d = g.decide(feats(t=10.0, diagnosis=diag(["stiffness_loss"]), margin=margin_v2(t=9.0, f_alpha=1.0)))
    assert d.mode == "nominal" and "until a physics estimate made after it exists" in d.reasoning
    # an estimate after the evidence that finds holding speed stable: keep holding (the slow-down trap)
    d = g.decide(feats(t=11.0, diagnosis=diag(["stiffness_loss"]), margin=margin_v2(t=10.5, f_alpha=0.6, uf=40.0)))
    assert d.mode == "nominal"
    # the physics finds the current mode unstable: move to the first admissible mode
    # (identified flutter speed 40 m/s: 'cautious' ends at 41.8 m/s and still needs control, 'safe' at 32.5 m/s does not)
    m = margin_v2(t=11.5, f_alpha=0.6, uf=40.0, stable={"nominal": False, "robust": False, "cautious": False},
                  needed={"nominal": True, "robust": True, "cautious": True, "safe": False})
    d = g.decide(feats(t=12.0, diagnosis=diag(["stiffness_loss"]), margin=m))
    assert d.mode == "safe"


def test_never_de_escalates_on_its_own():
    d = GuardianSupervisor().decide(feats(mode="safe", diagnosis=diag(["servo_slew"]), margin=margin_v2()))
    assert d.mode == "safe"


def test_works_without_diagnosis_or_margin():
    g = GuardianSupervisor()
    assert g.decide(feats()).mode == "nominal"
    assert g.decide(feats(noise_ratio=5.0)).mode == "cautious"  # symptom rules as the fallback


def test_verdict_without_estimate_admits_everything_and_falls_back_when_nothing_passes():
    v = physics_verdict(None, UF)
    assert v.time is None and all(v.admissible.values())
    none_ok = margin_v2(stable={m: False for m in MODE_ORDER},
                        needed={m: True for m in MODE_ORDER})
    v = physics_verdict(none_ok, UF)
    assert all(v.admissible.values()) and "all" in v.why
