"""Physics estimator v2: delayed closed loops, Lyapunov command statistics, the fast flutter speed, the on-board
estimate (authority, measured delay, disturbances, robust margins), the v2 supervisor and the report block.
Development seed 1000 only (the benchmarks use 0-19, 100-119, 200-204)."""

import math
from dataclasses import replace

import numpy as np
import pytest

from flutter_ai.fault_tolerance.diagnostics import FTFeatures, format_ft_report, format_physics
from flutter_ai.fault_tolerance.faults import Fault, benchmark_fault, delay_at, plant_at
from flutter_ai.fault_tolerance.margin import (
    applied_commands, command_statistics, estimate, estimate_v2, fit_window, fit_window_v2, flutter_speed_fast,
)
from flutter_ai.fault_tolerance.simulate import MODE_ORDER, SPEED_STEP, _delays, ft_simulate
from flutter_ai.fault_tolerance.stability import mode_matrices, mode_systems
from flutter_ai.fault_tolerance.supervisors import FTPhysicsRuleSupervisorV2
from flutter_ai.flutter import flutter_analysis
from flutter_ai.params import get_preset

DT = 0.005


@pytest.fixture(scope="module")
def wing():
    p = get_preset("uav_flap")
    fa = flutter_analysis(p)
    return p, fa.flutter_speed, fa.flutter_frequency_hz


def _max_rel_error(p, Uf, fhz, mode, key, amp, d):
    fault = Fault("loop_delay", onset=0.0, ramp=0.0, severity=(d - 1) / 2)  # delay d from the first sample
    r = ft_simulate(p, Uf, fhz, fault, None, duration=0.12, turbulence_rms=0.0, initial_mode=mode, x0=(amp, 0.0))
    assert delay_at(fault, 0.0) == d and not r.failed
    assert np.max(np.abs(r.u)) < 0.5 * p.u_max[0]  # small-signal: the flap limit never clips
    A = mode_matrices(p, round(1.1 * Uf / SPEED_STEP) * SPEED_STEP, delay=d)[key]
    N = r.x.shape[1]
    xi, err = np.zeros(A.shape[0]), 0.0
    xi[:N] = r.x[0]
    for k in range(1, len(r.t)):
        xi = A @ xi
        err = max(err, float(np.max(np.abs(xi[:N] - r.x[k]) / (np.max(np.abs(r.x), axis=0) + 1e-15))))
    return err


@pytest.mark.parametrize("d", [2, 3])
def test_delayed_mode_matrices_reproduce_the_simulator(d):
    """With the plant's loop delay at d samples (controllers keep their 1-sample design), step by step."""
    p = replace(get_preset("uav_flap"), u_rate_max=None)
    fa = flutter_analysis(p)
    # LQG is linear but linearly unstable at 1.1 U_F with these delays: a tiny amplitude keeps it below the flap limit
    assert _max_rel_error(p, fa.flutter_speed, fa.flutter_frequency_hz, "nominal", "lqg", 1e-8, d) < 1e-9
    big = _max_rel_error(p, fa.flutter_speed, fa.flutter_frequency_hz, "robust", "neural", 1e-4, d)
    small = _max_rel_error(p, fa.flutter_speed, fa.flutter_frequency_hz, "robust", "neural", 1e-5, d)
    assert small < 2e-3 and small < 0.2 * big  # the neural matrix is a linearisation


def test_default_delay_keeps_the_v1_matrices(wing):
    p, Uf, _ = wing
    for key, (A, G, V) in mode_systems(p, 1.1 * Uf).items():
        assert np.array_equal(A, mode_matrices(p, 1.1 * Uf)[key])
        assert G.shape == (A.shape[0], 1) and V.shape == (A.shape[0], 2 * p.n_dof)
    with pytest.raises(ValueError, match=">= 1"):
        mode_systems(p, Uf, delay=0)


def test_lyapunov_command_statistics_match_a_monte_carlo_run(wing):
    """Independent check of the stationary RMS of u and of its rate, with AR(1) gusts and sensor noise."""
    p, Uf, _ = wing
    sysm = mode_systems(p, 1.1 * Uf, DT)["lqg"]
    A, G, V = sysm
    sw, sv = 0.4, 0.01
    u_rms, rate_rms = command_statistics(p, sysm, DT, sw, sv)
    from flutter_ai.controllers.costs import reference_scales

    rng = np.random.default_rng(1000)
    phi, n_steps = math.exp(-DT / 0.05), 120_000
    iu = 2 * (2 * p.n_dof + 2) + 3
    xi, w, us = np.zeros(A.shape[0]), 0.0, np.empty(n_steps)
    e = rng.standard_normal(n_steps) * sw * math.sqrt(1 - phi**2)
    v = rng.standard_normal((n_steps, 2 * p.n_dof)) * sv * reference_scales(p)[:2 * p.n_dof]
    for k in range(n_steps):
        xi = A @ xi + G[:, 0] * w + V @ v[k]
        w = phi * w + e[k]
        us[k] = xi[iu]
    us = us[2000:]
    assert np.sqrt(np.mean(us**2)) == pytest.approx(u_rms[0], rel=0.05)
    assert np.sqrt(np.mean(np.diff(us) ** 2)) / DT == pytest.approx(rate_rms[0], rel=0.05)
    assert command_statistics(p, mode_systems(p, 1.1 * Uf, DT, delay=3)["lqg"], DT, sw) is None  # unstable loop


def test_fast_flutter_speed_matches_the_full_analysis(wing):
    p, Uf, _ = wing
    for q in (p, p.with_overrides(f_alpha=0.85 * p.f_alpha), p.with_overrides(flap_span_fraction=0.25)):
        assert flutter_speed_fast(q) == pytest.approx(flutter_analysis(q).flutter_speed, rel=1e-6)


def test_applied_commands_follow_the_measured_delay_history():
    p = get_preset("uav_flap")
    us = np.radians(np.linspace(0.0, 5.0, 60))[:, None]
    assert np.array_equal(applied_commands(p, us, 60, DT, np.ones(60, dtype=int)), applied_commands(p, us, 60, DT))
    delays = np.array([1] * 30 + [3] * 30)
    ua = applied_commands(replace(p, u_rate_max=None), us, 60, DT, delays)
    assert ua[29, 0] == us[28, 0] and ua[30, 0] == us[27, 0] and ua[0, 0] == 0.0


def _flight(p, Uf, fhz, kind, t_est):
    fault = benchmark_fault(kind)
    r = ft_simulate(p, Uf, fhz, fault, None, duration=t_est, seed=1000)
    n, U = len(r.t), round(r.airspeed[0] / SPEED_STEP) * SPEED_STEP
    ua = applied_commands(p, r.u, n, DT, _delays(fault, r.t, n))
    i0, i1 = int(0.5 / DT), int(4.5 / DT)
    base = fit_window_v2(p, U, DT, r.y[i0:i1], ua[i0:i1])
    k = int(round(t_est / DT))
    a = k - int(4.0 / DT)
    est, _ = estimate_v2(p, Uf, t_est, U, DT, r.y[a:k], ua[a:k], base, delay=delay_at(fault, t_est),
                         cruise=float(r.airspeed[0]))
    ua1 = applied_commands(p, r.u, n, DT)
    est1, _ = estimate(p, Uf, t_est, U, DT, r.y[a:k], ua1[a:k], fit_window(p, U, DT, r.y[i0:i1], ua1[i0:i1]),
                       cruise=float(r.airspeed[0]))
    return est, est1, plant_at(p, fault, t_est)


def test_v2_on_a_healthy_wing_admits_every_mode(wing):
    p, Uf, fhz = wing
    est, _, _ = _flight(p, Uf, fhz, "none", 10.0)
    assert est.turbulence_rms == pytest.approx(0.25, rel=0.2) and est.noise_std < 0.01
    assert est.params_cov_inflation >= 1.0 and est.robust
    assert est.flutter_speed == pytest.approx(Uf, rel=0.03) and est.flutter_speed_worst <= est.flutter_speed
    assert all(est.modes[m]["admissible"] for m in ("nominal", "cautious", "safe"))
    assert est.modes["nominal"]["authority_ratio"] < 1.0 and est.modes["nominal"]["control_needed"]


def test_v2_flags_the_flap_damage_that_v1_misses(wing):
    """Flap damage kills the wing through saturation: v1 (linear stability) still calls nominal stable."""
    p, Uf, fhz = wing
    est, est1, true_p = _flight(p, Uf, fhz, "flap_effectiveness", 12.5)
    assert est.params["flap_span_fraction"] == pytest.approx(true_p.flap_span_fraction, abs=0.1)
    assert est1.mode_spectral_radius["nominal"] < 1.0
    nominal = est.modes["nominal"]
    assert nominal["control_needed"] and nominal["authority_ratio"] >= 1.0 and not nominal["admissible"]
    assert est.modes["cautious"]["admissible"] and not est.modes["cautious"]["control_needed"]


def test_v2_predicts_with_the_measured_loop_delay(wing):
    p, Uf, fhz = wing
    est, est1, _ = _flight(p, Uf, fhz, "loop_delay", 9.5)
    assert est.delay_samples == 2
    assert est1.mode_spectral_radius["nominal"] < 1.0  # v1 assumes the nominal 5 ms
    assert est.mode_spectral_radius["nominal"] > 1.0 and not est.modes["nominal"]["admissible_point"]
    assert est.modes["robust"]["admissible_point"]  # the neural policy tolerates 10 ms


def test_simulator_feeds_v2_estimates_to_the_supervisor(wing):
    p, Uf, fhz = wing
    seen = []

    class Probe:
        name = "probe"

        def decide(self, f):
            seen.append(f.margin)

    ft_simulate(p, Uf, fhz, benchmark_fault("none"), Probe(), duration=6.0, seed=1000, margin_every_s=1.0,
                supervisor_interval=1.0, margin_version=2)
    assert seen[-1]["version"] == 2 and set(seen[-1]["modes"]) == set(MODE_ORDER)
    assert "Control authority" in format_physics(_features(seen[-1]))


def _features(margin=None, mode="nominal", **kw):
    base = dict(time=10.0, airspeed=51.0, airspeed_ratio=1.1, mode=mode, risk=0.1, energy_growth=1.0,
                saturation_fraction=0.0, pitch_freq_hz=9.0, nominal_freq_hz=10.0,
                residual_ratio={"h": 1.0, "alpha": 1.0, "beta": 1.0}, noise_ratio=1.0, loop_delay_ms=5.0)
    base.update(kw)
    return FTFeatures(**base, margin=margin)


def _v2(robust=MODE_ORDER, point=MODE_ORDER, saturating=()):
    modes = {m: {"target_airspeed": 51.0, "spectral_radius_worst": 0.99, "control_needed": m in ("nominal", "robust"),
                 "spectral_radius_needed": 0.95, "command_rms": 0.03, "command_rate_rms": 3.0,
                 "authority_ratio": 1.4 if m in saturating else 0.7, "authority_ok": m not in saturating,
                 "stable": m in point, "stable_share": 1.0 if m in robust else 0.5,
                 "admissible": m in robust and m not in saturating,
                 "admissible_point": m in point and m not in saturating} for m in MODE_ORDER}
    return {"time": 9.0, "airspeed": 51.0, "params": {"f_alpha": 1.0, "flap_span_fraction": 0.3}, "params_std": {},
            "flutter_speed": 48.0, "flutter_speed_std": 1.0, "margin": 48.0 / 51.0 - 1.0,
            "mode_spectral_radius": {m: 0.97 for m in MODE_ORDER}, "converged": True, "version": 2,
            "delay_samples": 1, "turbulence_rms": 0.6, "noise_std": 0.0, "params_cov_inflation": 9.0,
            "confidence": 0.95, "robust": True, "flutter_speed_worst": 46.0, "margin_worst": 46.0 / 51.0 - 1.0,
            "modes": modes}


def test_v2_supervisor_logic():
    sup = FTPhysicsRuleSupervisorV2()
    # no rule fires, but the current mode would saturate the flap -> least disruptive admissible mode
    d = sup.decide(_features(_v2(saturating=("nominal", "robust"))))
    assert d.mode == "cautious" and "saturate" in d.reasoning
    # the rules' proposal is kept when it is admissible (noise rule -> cautious)
    assert sup.decide(_features(_v2(), noise_ratio=4.0)).mode == "cautious"
    # nothing robustly admissible: the point estimate decides (tiered); strict takes 'safe'
    margin = _v2(robust=(), point=("nominal",))
    assert sup.decide(_features(margin, noise_ratio=4.0)).mode == "nominal"
    assert FTPhysicsRuleSupervisorV2(strict=True).decide(_features(margin)).mode == "safe"
    assert sup.decide(_features(_v2(robust=(), point=()))).mode == "safe"
    # without a v2 estimate (none, or a v1 dict) it is exactly the symptom rules
    assert sup.decide(_features(None, noise_ratio=4.0)).mode == "cautious"
    v1 = {k: v for k, v in _v2(saturating=MODE_ORDER).items() if k in (
        "time", "airspeed", "params", "params_std", "flutter_speed", "flutter_speed_std", "margin",
        "mode_spectral_radius", "converged")}
    assert sup.decide(_features(v1)).mode == "nominal"


def test_report_shows_v2_lines_only_for_v2_estimates():
    v2 = _v2(saturating=("nominal",))
    text = format_ft_report(_features(v2))
    assert "Control authority" in text and "nominal x1.40 SATURATES" in text and "Admissible modes" in text
    v1 = {k: v for k, v in v2.items() if k not in ("version", "modes", "delay_samples", "turbulence_rms", "noise_std",
                                                    "params_cov_inflation", "confidence", "robust",
                                                    "flutter_speed_worst", "margin_worst")}
    text1 = format_physics(_features(v1))
    assert "Control authority" not in text1
    assert text1.endswith("- Blind spots: this estimate cannot see actuator, sensor, loop-timing or turbulence problems.")


def test_margin_eval_refuses_reserved_seeds():
    from flutter_ai.fault_tolerance.margin_eval import _check_seeds, _confusion

    for bad in ((5,), (1000, 100), (203,), (999,)):
        with pytest.raises(ValueError, match="reserved"):
            _check_seeds(bad)
    _check_seeds((1000, 1004))
    c = _confusion([(True, True), (True, False), (False, True), (False, False)])
    assert c["ok_failed_missed"] == 1 and c["flagged_survived_conservative"] == 1 and c["accuracy"] == 0.5
