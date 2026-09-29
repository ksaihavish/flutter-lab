"""Learned fault diagnoser: composite faults, the data splits and labels, the on-board feature path, NumPy inference,
the unknown-fault score and the report block. Fast: tiny synthetic models and short flights on seeds >= 1000."""

import time

import numpy as np
import pytest

from flutter_ai.fault_tolerance.diagnoser import (
    KNOWN, MODEL_FORMAT, DiagnoserModel, FlightDiagnoser, baseline_stats, confirmed, feature_names, model_path,
)
from flutter_ai.fault_tolerance.diagnoser_data import (
    ALL_KINDS, SPLITS, Dataset, diagnosis_indices, featurise, flight_spec,
)
from flutter_ai.fault_tolerance.diagnoser_training import calibrate_threshold, flight_max_confirmed
from flutter_ai.fault_tolerance.diagnostics import FTFeatures, format_ft_report
from flutter_ai.fault_tolerance.faults import (
    CompositeFault, Fault, components, delay_at, gust_gain_at, plant_at, sensor_noise_at,
)
from flutter_ai.fault_tolerance.simulate import ft_simulate
from flutter_ai.flutter import flutter_analysis
from flutter_ai.params import get_preset


@pytest.fixture(scope="module")
def wing():
    p = get_preset("uav_flap")
    fa = flutter_analysis(p)
    return p, fa.flutter_speed, fa.flutter_frequency_hz


def _tiny_model(p, seed=0, fingerprint=None) -> DiagnoserModel:
    """Random weights with the real interface: enough to test shapes, speed and the on-board wiring."""
    rng = np.random.default_rng(seed)
    d, e = len(feature_names(p)), 4

    def layer(a, b):
        return rng.standard_normal((b, a)) * 0.3, rng.standard_normal(b) * 0.1

    meta = {"format": MODEL_FORMAT, "preset": p.name, "params_fingerprint": fingerprint or p.fingerprint(),
            "classes": list(KNOWN), "prob_threshold": 0.5, "ood_threshold": 1.0, "mahalanobis_threshold": 10.0}
    return DiagnoserModel([layer(d, 16), layer(16, e)], [layer(e, 8), layer(8, len(KNOWN))],
                          [layer(e, 16), layer(16, d)], np.zeros(d), np.ones(d), rng.standard_normal((3, e)),
                          np.eye(e), meta)


def test_composite_fault_combines_the_parts():
    p = get_preset("uav_flap")
    stiff, flap = Fault("stiffness_loss", onset=2.0, ramp=2.0), Fault("flap_effectiveness", onset=5.0, ramp=1.0)
    noise, turb, lag = Fault("sensor_noise", onset=1.0, ramp=1.0), Fault("turbulence_rise"), Fault("loop_delay")
    cf = CompositeFault((stiff, flap, noise))
    q = plant_at(p, cf, 10.0)
    assert q.f_alpha == plant_at(p, stiff, 10.0).f_alpha < p.f_alpha
    assert q.flap_span_fraction == plant_at(p, flap, 10.0).flap_span_fraction < p.flap_span_fraction
    assert plant_at(p, cf, 0.5) is p  # untouched before every onset
    assert sensor_noise_at(cf, 10.0) == sensor_noise_at(noise, 10.0) > 0.0
    both = CompositeFault((turb, lag))
    assert gust_gain_at(both, 20.0) == gust_gain_at(turb, 20.0) > 1.0 and delay_at(both, 20.0) == delay_at(lag, 20.0) == 3
    assert cf.onset == 1.0 and cf.kind == "stiffness_loss+flap_effectiveness+sensor_noise"
    assert components(cf) == (stiff, flap, noise) and components(stiff) == (stiff,)
    # single faults behave exactly as before
    assert sensor_noise_at(Fault("servo_slew"), 20.0) == 0.0 and gust_gain_at(Fault("servo_slew"), 20.0) == 1.0
    for bad in ((), (turb, Fault("turbulence_rise", onset=9.0)), (Fault("gremlins"),)):
        with pytest.raises(ValueError):
            CompositeFault(bad)


def test_ft_simulate_flies_a_composite_fault(wing):
    p, Uf, fhz = wing
    cf = CompositeFault((Fault("sensor_noise", onset=5.0, ramp=1.0, severity=1.5),
                         Fault("turbulence_rise", onset=5.0, ramp=1.0)))
    r = ft_simulate(p, Uf, fhz, cf, None, duration=7.0, seed=1004)
    n = 2 * p.n_dof
    before, after = r.t < 5.0, r.t > 6.0
    assert np.array_equal(r.y[before], r.x[before, :n])  # no sensor noise before the onset
    assert np.std(r.y[after] - r.x[after, :n]) > 0.0 and r.fault is cf and r.metrics["failed"] in (True, False)


def test_splits_keep_held_out_faults_and_reserved_seeds_apart():
    reserved = set(range(20)) | set(range(100, 120)) | set(range(200, 205))
    seeds = {}
    for split, cfg in SPLITS.items():
        seeds[split] = set(range(cfg["first_seed"], cfg["first_seed"] + cfg["n"]))
        assert min(seeds[split]) >= 1000 and not seeds[split] & reserved
    names = list(seeds)
    assert all(not seeds[a] & seeds[b] for i, a in enumerate(names) for b in names[i + 1:])
    for split in ("train", "val", "eval"):
        kinds = {f["kind"] for i in range(SPLITS[split]["n"]) for f in flight_spec(split, i)["faults"]}
        assert kinds == set(KNOWN)  # every known kind, never a held-out one
    held = {f["kind"] for i in range(20) for f in flight_spec("eval_held_out", i)["faults"]}
    assert held == {"loop_delay", "flap_effectiveness"}
    assert any(len(flight_spec("train", i)["faults"]) >= 2 for i in range(50))  # combinations are trained on


def test_labels_leave_out_faults_too_small_to_see():
    level = np.zeros((4, len(ALL_KINDS)))
    present = np.zeros_like(level, dtype=bool)
    j, s = ALL_KINDS.index("sensor_noise"), ALL_KINDS.index("servo_slew")
    level[:, j], present[:, j] = [0.0, 0.1, 0.5, 1.0], [False, True, True, True]
    level[:, s], present[:, s] = [0.0, 0.5, 0.7, 1.0], [False, True, True, True]  # the servo counts from 0.65
    ds = Dataset(np.zeros((4, 1)), np.arange(4.0), np.zeros(4, int), level, present, [{}])
    Y, defined = ds.targets()
    jk, sk = KNOWN.index("sensor_noise"), KNOWN.index("servo_slew")
    assert Y[:, jk].tolist() == [False, False, True, True] and defined[:, jk].tolist() == [True, False, True, True]
    assert Y[:, sk].tolist() == [False, False, True, True] and defined[:, sk].tolist() == [True, False, True, True]


def test_confirmation_needs_consecutive_diagnoses():
    flags = np.array([[True], [False], [True], [True], [True]])
    assert confirmed(flags)[:, 0].tolist() == [False, False, False, True, True]


def test_numpy_inference_shape_range_speed_and_round_trip(wing, tmp_path):
    p = wing[0]
    m = _tiny_model(p)
    X = np.random.default_rng(1).standard_normal((50, len(feature_names(p))))
    prob, rec, maha = m.predict(X)
    assert prob.shape == (50, len(KNOWN)) and rec.shape == maha.shape == (50,)
    assert np.all((prob > 0) & (prob < 1)) and np.all(rec >= 0) and np.all(maha >= 0)
    x = X[:1]
    for _ in range(20):
        m.predict(x)
    times = []
    for _ in range(300):
        t0 = time.perf_counter()
        m.predict(x)
        times.append(time.perf_counter() - t0)
    assert np.median(times) < 1e-3  # one diagnosis well under a millisecond
    m.save(tmp_path / "d.npz")
    again = DiagnoserModel.load(tmp_path / "d.npz")
    assert all(np.array_equal(a, b) for a, b in zip(again.predict(X), m.predict(X)))


def test_trained_model_scores_data_far_from_training_as_unknown():
    """Monotonic sanity of the committed model's unknown-fault score along rays away from the training data. Not
    asserted for the secondary Mahalanobis score: the tanh encoder bounds the embedding, so that score saturates far
    out and along some directions stays below its threshold (found while writing this test; one reason the
    reconstruction error is the primary score)."""
    p = get_preset("uav_flap")
    m = DiagnoserModel.load(model_path(p.name))
    assert m.meta["classes"] == list(KNOWN) and m.meta["params_fingerprint"] == p.fingerprint()
    assert not set(m.meta["held_out_never_seen"]) & set(m.meta["classes"])
    rng = np.random.default_rng(2)
    for _ in range(5):
        v = rng.standard_normal(len(m.mean))
        v /= np.linalg.norm(v)
        _, rec, _ = m.predict(np.array([m.mean + k * m.std * v for k in (10.0, 30.0, 100.0)]))
        assert np.all(np.diff(rec) > 0) and rec[0] > m.ood_threshold


def test_on_board_diagnosis_equals_the_training_features(wing):
    """ft_simulate(diagnose_every_s=...) must see exactly what the diagnoser was trained on (featurise)."""
    p, Uf, fhz = wing
    m = _tiny_model(p)
    seen = []

    class Probe:
        name = "probe"

        def decide(self, f):
            seen.append((f.time, f.diagnosis))
            return None

    fault = Fault("sensor_noise", onset=5.0, ramp=1.0, severity=1.5)
    r = ft_simulate(p, Uf, fhz, fault, Probe(), duration=7.0, seed=1005, diagnose_every_s=0.5, diagnoser=m,
                    supervisor_interval=0.5)
    assert seen and all(d is not None and d["time"] == pytest.approx(t) for t, d in seen)
    off = featurise(p, Uf, r, fault)
    prob, rec, _ = m.predict(off["X"])
    assert np.allclose(off["t"], [t for t, _ in seen])
    assert np.allclose(prob, [[d["probabilities"][c] for c in KNOWN] for _, d in seen], rtol=1e-12, atol=1e-12)
    assert np.allclose(rec, [d["ood_score"] for _, d in seen], rtol=1e-12, atol=1e-12)
    k_base = int(np.argmax(r.t >= 4.5))
    assert diagnosis_indices(r.t, k_base)[:3] == [k_base, k_base + 100, k_base + 200]


def test_diagnoser_refuses_a_different_wing(wing):
    p, Uf, _ = wing
    r = ft_simulate(p, Uf, wing[2], Fault("none"), None, duration=5.0, seed=1006)
    base = baseline_stats(p, 0.005, r.y, r.u, r.airspeed, r.mode, 900)
    with pytest.raises(ValueError, match="train-diagnoser"):
        FlightDiagnoser(_tiny_model(p, fingerprint="another wing"), p, Uf, 0.005, base)


def test_unknown_threshold_calibration_bounds_flight_false_alarms():
    scores = np.concatenate([np.full(4, float(i)) for i in range(20)])  # 20 flights, 4 windows each
    flight = np.repeat(np.arange(20), 4)
    fmax = flight_max_confirmed(scores, flight, 20)
    assert fmax.tolist() == [float(i) for i in range(20)]
    tau = calibrate_threshold(fmax, 0.05)
    assert np.mean(fmax > tau) <= 0.05 and np.mean(fmax > tau - 1e-9) > 0.05  # the smallest such threshold


def _features(diagnosis=None, margin=None):
    return FTFeatures(time=12.0, airspeed=51.0, airspeed_ratio=1.1, mode="nominal", risk=0.1, energy_growth=1.0,
                      saturation_fraction=0.0, pitch_freq_hz=9.0, nominal_freq_hz=10.0,
                      residual_ratio={"h": 1.0, "alpha": 1.0, "beta": 6.0}, noise_ratio=1.0, loop_delay_ms=5.0,
                      margin=margin, diagnosis=diagnosis)


def test_report_has_a_diagnosis_block_only_with_a_diagnosis():
    assert "LEARNED DIAGNOSIS" not in format_ft_report(_features())
    d = {"time": 11.5, "window_s": 2.5, "probabilities": {"servo_slew": 0.93, "stiffness_loss": 0.02,
                                                          "sensor_noise": 0.01, "turbulence_rise": 0.1},
         "prob_threshold": {c: 0.5 for c in KNOWN}, "confirmed": ["servo_slew"], "ood_score": 4.2,
         "ood_threshold": 1.3, "unknown": True, "ood_mahalanobis": 30.0}
    text = format_ft_report(_features(d))
    assert "LEARNED DIAGNOSIS" in text and "servo_slew 93 % CONFIRMED" in text and "stiffness_loss 2 %" in text
    assert "unknown fault confirmed: YES" in text and "threshold 1.30" in text
    margin = {"time": 11.0, "airspeed": 51.0, "params": {"f_alpha": 1.0}, "params_std": {}, "flutter_speed": 46.4,
              "flutter_speed_std": 0.1, "margin": -0.09, "mode_spectral_radius": {"nominal": 0.9}, "converged": True}
    both = format_ft_report(_features(d, margin))
    assert both.index("PHYSICS ESTIMATE") < both.index("LEARNED DIAGNOSIS")  # appended after the physics block
