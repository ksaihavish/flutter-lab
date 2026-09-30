"""
Recorded fault-tolerant flights for the Guardian live view (web app: `api/guardian.py`, `frontend/guardian.html`).

One flight is exactly the benchmark's `ft_simulate`, with a recording wrapper around the supervisor. At every
supervisor tick the wrapper keeps the on-board features, the report text an LLM supervisor would read
(`format_ft_report`), the latest on-board physics estimate and the decision; the wrapper returns the decision
unchanged, so recording does not alter the flight. After the flight, the true open-loop flutter speed of the
(possibly damaged) wing is computed at every fault level the flight went through: ground truth that exists only in
simulation, shown next to the on-board estimate. Everything the panel shows comes from here; nothing is typed by hand.
Units are SI (m, rad, m/s); the browser converts for display.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from functools import lru_cache

import numpy as np

from ..flutter import _first_crossing, default_velocities, flutter_analysis, max_oscillatory_real
from ..params import WingParams
from .benchmark import MARGIN_EVERY_S
from .diagnostics import FTFeatures, format_ft_report
from .faults import FAULTS, Fault, plant_at
from .simulate import FT_MODES, MIN_DWELL_S, MODE_ORDER, FTDecision, ft_simulate
from .supervisors import FTPhysicsRuleSupervisor, FTRuleSupervisor

# The LLM supervisors are deliberately not offered: this view must never call a model (see Rules.md §5 and the
# ft-benchmark command for LLM runs).
SUPERVISORS: dict[str, dict] = {
    "none": {"label": "No supervisor", "needs_estimator": False,
             "description": "The nominal mode (LQG at cruise speed) for the whole flight. The on-board estimator "
                            "can still run, for display only."},
    "rules": {"label": "Rules", "needs_estimator": False,
              "description": "Hand-written symptom rules (the benchmark's rule supervisor). They do not read the "
                             "physics estimate."},
    "rules+physics": {"label": "Rules + physics", "needs_estimator": True,
                      "description": "The same symptom rules decide whether to act; the on-board physics estimate "
                                     "vetoes any mode predicted unstable on the identified wing."},
}
TRUTH_SAMPLE_S = 0.05  # fault levels are piecewise constant (faults.N_LEVELS steps); this grid finds every step


@lru_cache(maxsize=8)
def healthy_flutter(p: WingParams) -> tuple[float, float]:
    """(flutter speed [m/s], flutter frequency [Hz]) of the healthy wing, as the benchmark computes them."""
    fa = flutter_analysis(p)
    if fa.flutter_speed is None or fa.flutter_frequency_hz is None:
        raise ValueError(f"No flutter found for preset '{p.name}' in the analysed speed range.")
    return fa.flutter_speed, fa.flutter_frequency_hz


@lru_cache(maxsize=256)
def true_flutter_speed(q: WingParams) -> float | None:
    """
    Open-loop flutter speed [m/s] of a (possibly damaged) wing: the same number flutter_analysis(q) reports (same
    sweep, same root refinement; tests/test_guardian_api.py checks equality), without building the V-g curves the
    panel does not need, which makes it 2-8 times faster.
    """
    return _first_crossing(lambda U: max_oscillatory_real(q, U), default_velocities(q))


class _Recorder:
    """Wraps a supervisor (or none) and keeps what it saw and answered at every tick. The inner decision is returned
    unchanged, and the features are pure functions of the flight, so the recorded flight is the unrecorded one."""

    def __init__(self, inner, on_tick: Callable[[float], None] | None = None):
        self.inner = inner
        self.name = getattr(inner, "name", "none")
        self.ticks: list[tuple[FTFeatures, FTDecision | None]] = []
        self._on_tick = on_tick

    def reset(self) -> None:
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    def decide(self, f: FTFeatures) -> FTDecision | None:
        d = self.inner.decide(f) if self.inner is not None else None
        self.ticks.append((f, d))
        if self._on_tick is not None:
            self._on_tick(f.time)
        return d


def _clean(x):
    """JSON-safe copy: numpy types to Python, non-finite floats to None (JSON has no NaN; nothing is dropped)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def fly_recorded(job: tuple) -> dict:
    """
    Worker (process pool) for the explainer and post-flight review studies: fly one diagnoser_data flight spec with a
    rule supervisor ("rules" or "rules+physics"), the on-board diagnoser (the committed model) and, if `margin`, the
    physics estimator; return every supervisor tick (features and decision), the outcome, and the diagnoser's
    features of the whole flight (diagnoser_data.featurise), which later candidate diagnosers are scored on.
    """
    from ..params import get_preset
    from .diagnoser import DIAGNOSE_EVERY_S
    from .diagnoser_data import DT, DURATION, _bank, fault_of, featurise

    preset, spec, supervisor, margin = job
    p = get_preset(preset)
    Uf, f_hz = healthy_flutter(p)
    fault = fault_of(spec)
    rec = _Recorder({"rules": FTRuleSupervisor, "rules+physics": FTPhysicsRuleSupervisor}[supervisor]())
    t0 = time.perf_counter()
    r = ft_simulate(p, Uf, f_hz, fault, rec, duration=DURATION, dt=DT, seed=spec["seed"],
                    cruise_ratio=spec["cruise_ratio"], turbulence_rms=spec["turbulence_rms"], bank_factory=_bank,
                    forced_mode=None if spec.get("forced_mode") is None else tuple(spec["forced_mode"]),
                    margin_every_s=MARGIN_EVERY_S if margin else None, diagnose_every_s=DIAGNOSE_EVERY_S)
    feats = featurise(p, Uf, r, fault)
    return {"spec": spec, "supervisor": supervisor, "failed": r.failed, "failure_time_s": r.failure_time,
            "cruise": float(r.airspeed[0]), "t_end": float(r.t[-1]), "flutter_speed": Uf,
            "switches": [{"time": ta, "mode": m, "decided": dec} for ta, m, dec in r.switch_log],
            "ticks": [{"features": f.to_dict(), "decision": None if d is None else asdict(d)} for f, d in rec.ticks],
            "X": feats["X"], "t": feats["t"], "level": feats["level"], "present": feats["present"],
            "wall_s": round(time.perf_counter() - t0, 2)}


def _truth(p: WingParams, fault: Fault, t_end: float,
           progress: Callable[[float], None] | None = None) -> dict:
    """True flutter speed of the wing at every fault level between 0 and t_end (step function: value from t[i])."""
    times = np.arange(0.0, t_end + 1e-9, TRUTH_SAMPLE_S)
    levels = [0.0 if fault.kind == "none" else fault.level(float(t)) for t in times]
    keep = [0] + [i for i in range(1, len(times)) if levels[i] != levels[i - 1]]
    out_t, out_level, out_speed = [], [], []
    for n, i in enumerate(keep):
        # actuator limits do not enter the open-loop flutter speed; dropping them lets a servo fault reuse the cache
        q = replace(plant_at(p, fault, float(times[i])), u_max=p.u_max, u_rate_max=p.u_rate_max)
        out_t.append(float(times[i]))
        out_level.append(levels[i])
        out_speed.append(true_flutter_speed(q))
        if progress is not None:
            progress((n + 1) / len(keep))
    return {"t": out_t, "fault_level": out_level, "flutter_speed": out_speed}


def record_flight(p: WingParams, fault: Fault, supervisor: str = "rules+physics", *, estimator: bool = True,
                  duration: float = 30.0, seed: int = 1000, max_points: int = 3000,
                  progress: Callable[[str, float, float], None] | None = None) -> dict:
    """
    Fly one mission and return everything the Guardian panel needs (JSON-safe, SI units).
    progress(stage, fraction, sim_time): stage "flying" (fraction of the flight simulated) or "ground_truth".
    """
    if supervisor not in SUPERVISORS:
        raise ValueError(f"Unknown supervisor '{supervisor}'. Choose one of: {', '.join(SUPERVISORS)}")
    if SUPERVISORS[supervisor]["needs_estimator"] and not estimator:
        raise ValueError(f"The '{supervisor}' supervisor needs the on-board physics estimator (estimator=true).")
    if fault.kind not in FAULTS:
        raise ValueError(f"Unknown fault '{fault.kind}'. Choose one of: {', '.join(FAULTS)}")
    Uf, f_hz = healthy_flutter(p)
    t_end = duration

    def on_tick(t: float) -> None:
        if progress is not None:
            progress("flying", min(1.0, t / duration), t)

    def on_truth(fraction: float) -> None:
        if progress is not None:
            progress("ground_truth", fraction, t_end)

    inner = {"none": None, "rules": FTRuleSupervisor, "rules+physics": FTPhysicsRuleSupervisor}[supervisor]
    rec = _Recorder(None if inner is None else inner(), on_tick=on_tick)

    t0 = time.perf_counter()
    r = ft_simulate(p, Uf, f_hz, fault, rec, duration=duration, seed=seed,
                    margin_every_s=MARGIN_EVERY_S if estimator else None)
    flight_s = time.perf_counter() - t0
    t_end = float(r.t[-1])
    t1 = time.perf_counter()
    truth = _truth(p, fault, t_end, on_truth)
    truth_s = time.perf_counter() - t1

    # time series, downsampled for the browser; the last sample (the failure, if any) is always kept
    n_t = len(r.t)
    step = max(1, math.ceil(n_t / max_points))
    idx = np.arange(0, n_t, step)
    if idx[-1] != n_t - 1:
        idx = np.append(idx, n_t - 1)

    def arr(a: np.ndarray, digits: int) -> list[float | None]:
        return [v if math.isfinite(v) else None for v in np.round(np.asarray(a, dtype=float)[idx], digits).tolist()]

    n = p.n_dof
    series = {
        "t": arr(r.t, 4),
        "h": arr(r.x[:, 0], 7),
        "alpha": arr(r.x[:, 1], 7),
        "beta": arr(r.x[:, 2], 7) if n == 3 else None,
        "u": [arr(r.u[:, j], 7) for j in range(r.u.shape[1])],
        "airspeed": arr(r.airspeed, 3),
        "mode": [r.mode[i] for i in idx],
        "fault_level": [0.0 if fault.kind == "none" else fault.level(float(r.t[i])) for i in idx],
    }

    dec_index = {id(d): i for i, d in enumerate(r.decisions)}
    ticks, estimates, seen = [], [], set()
    for f, d in rec.ticks:
        m = f.margin
        if m is not None and m["time"] not in seen:
            seen.add(m["time"])
            estimates.append(m)
        ticks.append({
            "time": f.time, "airspeed": f.airspeed, "airspeed_ratio": f.airspeed_ratio, "mode": f.mode,
            "risk": f.risk, "energy_growth": f.energy_growth, "saturation_fraction": f.saturation_fraction,
            "pitch_freq_hz": f.pitch_freq_hz, "nominal_freq_hz": f.nominal_freq_hz,
            "residual_ratio": f.residual_ratio, "noise_ratio": f.noise_ratio, "loop_delay_ms": f.loop_delay_ms,
            "estimate_time": None if m is None else m["time"],
            # optional learned fault diagnosis (another component may add it to FTFeatures); shown when present
            "diagnosis": getattr(f, "diagnosis", None),
            "report": format_ft_report(f),
            "decision": None if d is None else dec_index.get(id(d)),
        })

    cruise = float(r.airspeed[0])
    modes = {m: {**FT_MODES[m], "target_speed": cruise if FT_MODES[m]["speed_ratio"] is None
                 else FT_MODES[m]["speed_ratio"] * Uf} for m in MODE_ORDER}
    spec = FAULTS[fault.kind]
    return _clean({
        "preset": p.name,
        "params": p.to_dict(),
        "flutter_speed": Uf,  # healthy wing
        "flutter_frequency_hz": f_hz,
        "cruise": cruise,
        "request": {"fault": fault.kind, "onset": fault.onset, "ramp": fault.ramp, "severity": fault.severity,
                    "supervisor": supervisor, "estimator": estimator, "duration": duration, "seed": seed},
        "fault": {"kind": fault.kind, "description": spec["description"], "held_out": spec["held_out"],
                  "benchmark_severity": spec["severity"], "onset": fault.onset, "ramp": fault.ramp,
                  "severity": fault.severity},
        "supervisor": {"id": supervisor, **SUPERVISORS[supervisor]},
        "estimator_every_s": MARGIN_EVERY_S if estimator else None,
        "min_dwell_s": MIN_DWELL_S,
        "mode_order": list(MODE_ORDER),
        "modes": modes,
        "failed": r.failed,
        "failure_time": r.failure_time,
        "metrics": r.metrics,
        "series": series,
        "ticks": ticks,
        "estimates": estimates,
        "decisions": [{"time": d.time, "mode": d.mode, "reasoning": d.reasoning, "source": d.source,
                       "latency_s": d.latency_s, "applied_at": d.applied_at, "outcome": d.outcome}
                      for d in r.decisions],
        "switches": [{"time": ta, "mode": m, "decided": decided} for ta, m, decided in r.switch_log],
        "truth": truth,
        "compute_s": {"flight": flight_s, "ground_truth": truth_s},
    })
