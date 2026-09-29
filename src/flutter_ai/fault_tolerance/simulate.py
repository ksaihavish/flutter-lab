"""
Fault-tolerant supervision simulator (docs/research/fault-tolerant-plan.md). Kept separate from simulation.py so
the Phase 4 benchmark is untouched.

    A wing cruises above its flutter speed under light turbulence with the `nominal` mode (LQG). A fault starts
    mid-flight. Every supervisor tick (one question at a time, latency applied in simulation time) the supervisor
    may pick a mode: an inner-loop controller from a pre-verified bank and an airspeed target. The autopilot
    changes airspeed with a limited deceleration; controllers are re-designed (gain-scheduled) as the speed changes.

Controller bank with bumpless transfer: every candidate controller runs every step on the same measurements and
the same *applied* command history, so all estimators stay warm and switching causes no transient.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..controllers import AeroLagObserver, Controller, LQGController, LQRController, NeuralController
from ..controllers.costs import input_weight, rate_weight, reference_scales, state_weight
from ..controllers.lqg import gust_augmented, kalman_gain
from ..metrics import COST_AGGRESSIVENESS
from ..model import discrete_state_space, initial_state
from ..params import WingParams
from .diagnostics import LOOKBACK_S, Baseline, FTFeatures, extract, healthy_freq, raw_features
from .faults import Fault, delay_at, gust_gain_at, plant_at, sensor_noise_at

FT_MODES: dict[str, dict] = {
    "nominal": {"controller": "lqg", "speed_ratio": None,
                "description": "LQG controller at cruise speed (the normal operating mode)."},
    "robust": {"controller": "neural", "speed_ratio": None,
               "description": "Switch to the neural controller at cruise speed. It tolerates slightly longer loop "
                              "delays than LQG but is less precise and less robust to a changed structure."},
    "cautious": {"controller": "lqg", "speed_ratio": 0.9,
                 "description": "Keep LQG and slow down to 90% of the healthy wing's flutter speed. Costs mission "
                                "time; the controller then only needs to keep a stable wing steady."},
    "safe": {"controller": "lqg", "speed_ratio": 0.7,
             "description": "Keep LQG and slow down to 70% of the healthy wing's flutter speed: the largest "
                            "margin (also if the structure is damaged), the largest mission cost."},
}
MODE_ORDER = ("nominal", "robust", "cautious", "safe")
SPEED_STEP = 0.25  # [m/s] airspeed quantisation for re-discretisation / re-design
# Minimum time between switches of the inner-loop controller (LQG <-> neural). Must be >= the dwell-time bound from
# stability.py (worst case 6.5 s over the airspeed schedule on uav_flap): then no sequence of controller switches at
# a fixed airspeed can destabilise the nominal closed loop (tests/test_fault_tolerance.py checks the inequality).
# Mode changes that keep the controller only move the airspeed target; the autopilot's limited deceleration makes
# those slow, gain-scheduled transitions, which the dwell-time argument does not cover and does not need to delay.
MIN_DWELL_S = 7.0


def controller_of(mode: str) -> str:
    return FT_MODES[mode]["controller"]


def make_bank() -> dict[str, Controller]:
    return {"lqg": LQGController(), "neural": NeuralController(), "lqr": LQRController()}


def _retarget(c: Controller, p: WingParams, U: float, dt: float) -> None:
    """Gain scheduling: move a running controller to a new airspeed, keeping its estimator state."""
    z = c.observer.z.copy() if c.observer is not None else None
    c.U = U
    c.observer = AeroLagObserver(p, U, dt)
    if z is not None:
        c.observer.z = z
    if isinstance(c, LQRController):
        c._gains = {}
    elif isinstance(c, LQGController):
        from ..controllers.design import design_settings

        noise = c.measurement_noise if c.measurement_noise is not None else design_settings(p, "lqg")[0]["measurement_noise"]
        c._A, c._B = gust_augmented(p, U, dt, c._tau)
        c._L = kalman_gain(p, U, dt, c._rms, c._tau, noise)
        c._gains = {}


@dataclass
class FTDecision:
    time: float
    mode: str
    reasoning: str
    source: str
    latency_s: float = 0.0
    applied_at: float | None = None  # when the answer arrives (decision time + latency)
    # "switched"; "kept" (already in that mode); "blocked" (controller switch inside the dwell time: dropped, the
    # supervisor is asked again); "pending" (the flight ended or failed before the answer arrived)
    outcome: str = "pending"


@dataclass
class FTResult:
    fault: Fault
    supervisor: str
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    airspeed: np.ndarray
    y: np.ndarray  # measurements (with sensor noise)
    mode: list[str]
    decisions: list[FTDecision]
    failed: bool
    failure_time: float | None
    switch_log: list[tuple[float, str, float]] = field(default_factory=list)  # (time applied, new mode, time decided)
    metrics: dict = field(default_factory=dict)


def _applied_nominal(us: np.ndarray, i0: int, k: int) -> np.ndarray:
    """Commands as applied with the nominal 1-sample loop delay, aligned with samples i0..k-1."""
    prev = us[i0 - 1:i0] if i0 > 0 else np.zeros((1, us.shape[1]))
    return np.vstack([prev, us[i0:k - 1]])


def ft_simulate(p: WingParams, flutter_speed: float, flutter_freq_hz: float, fault: Fault,
                supervisor=None, *, duration: float = 30.0, dt: float = 0.005, cruise_ratio: float = 1.1,
                turbulence_rms: float = 0.25, correlation_time: float = 0.05, seed: int = 0,
                supervisor_interval: float = 0.25, min_dwell_s: float = MIN_DWELL_S, decel: float = 2.0,
                baseline_window: tuple[float, float] = (0.5, 4.5), initial_mode: str = "nominal",
                forced_mode: tuple[float, str] | None = None, apply_latency: bool = True,
                bank_factory: Callable[[], dict[str, Controller]] = make_bank,
                x0: tuple[float, float] = (0.0, 0.0), margin_every_s: float | None = None) -> FTResult:
    """
    supervisor: object with decide(FTFeatures) -> FTDecision-like (mode, reasoning, source, latency_s) or None.
    forced_mode: (time, mode) to switch once at a fixed time (used by the oracle); overrides the supervisor.
    x0: initial (h [m], alpha [rad]) displacement (small-signal tests; the benchmark starts at rest).
    margin_every_s: run the on-board flutter-margin estimator (margin.py) this often; the latest estimate is added to
        the supervisor's features (physics-informed supervisors). None = no estimator.
    Latency: every decision's latency_s is applied in simulation time (LLMs measure theirs; rule supervisors have 0
    unless wrapped in a Delayed supervisor for the time-scale study).
    """
    n, rng = p.n_dof, np.random.default_rng(seed)
    n_t = int(round(duration / dt)) + 1
    t = np.arange(n_t) * dt
    ref = reference_scales(p)
    u_max = np.asarray(p.u_max, dtype=float)
    m = len(u_max)
    cruise = cruise_ratio * flutter_speed
    U = cruise
    U_level = round(U / SPEED_STEP) * SPEED_STEP  # the plant, the controllers and the diagnostics' model use this

    bank = bank_factory()
    for c in bank.values():
        c.reset(p, U_level, dt, 1)  # designed for the nominal wing, nominal delay
    if supervisor is not None and hasattr(supervisor, "reset"):
        supervisor.reset()  # e.g. preloads a local LLM before "take-off"
    mode = initial_mode

    # turbulence: AR(1) with time-varying intensity (fault)
    phi = math.exp(-dt / correlation_time)
    white = rng.standard_normal(n_t) * math.sqrt(1.0 - phi**2)
    w_state = 0.0

    x = initial_state(p, *x0)
    xs, us = np.zeros((n_t, x.size)), np.zeros((n_t, m))
    ys = np.zeros((n_t, 2 * n))
    speeds = np.zeros(n_t)
    modes: list[str] = [mode] * n_t
    actuator = np.zeros(m)
    decisions: list[FTDecision] = []
    pending: list[FTDecision] = []
    switch_log: list[tuple[float, str, float]] = []
    last_ctrl_switch, next_sup = -1e9, baseline_window[1]
    margin_base, next_margin, latest_margin = None, math.inf, None
    baseline: Baseline | None = None
    failed, failure_time, last = False, None, n_t - 1

    def switch(new: str, tk: float, decided: float) -> None:
        nonlocal mode, last_ctrl_switch
        if controller_of(new) != controller_of(mode):
            last_ctrl_switch = tk
        mode = new
        switch_log.append((tk, new, decided))

    for k in range(n_t):
        tk = float(t[k])
        plant = plant_at(p, fault, tk)
        noise_std = sensor_noise_at(fault, tk)
        y = x[:2 * n] + (rng.standard_normal(2 * n) * noise_std * ref if noise_std > 0 else 0.0)
        xs[k], ys[k], speeds[k] = x, y, U

        # healthy baseline for the diagnostics (before any fault in the benchmark: onset >= baseline end)
        if baseline is None and tk >= baseline_window[1]:
            i0 = int(baseline_window[0] / dt)
            res, noise = raw_features(p, U_level, dt, ys[i0:k], _applied_nominal(us, i0, k))
            baseline = Baseline(residual_rms=res, noise=noise, freq_hz=healthy_freq(p, ys[i0:k], dt))
            if margin_every_s is not None:
                from .margin import applied_commands, fit_window

                margin_base = fit_window(p, U_level, dt, ys[i0:k], applied_commands(p, us, k, dt)[i0:k])
                next_margin = tk

        # on-board flutter-margin estimate from the last LOOKBACK_S (a physics-informed report block)
        if margin_every_s is not None and baseline is not None and tk >= next_margin - 1e-12:
            from .margin import applied_commands, estimate

            next_margin += margin_every_s
            a = max(0, k - int(LOOKBACK_S / dt))
            est, _ = estimate(p, flutter_speed, tk, U_level, dt, ys[a:k], applied_commands(p, us, k, dt)[a:k],
                              margin_base, cruise=cruise)
            latest_margin = est.to_dict()

        # apply answers that have arrived. Only a change of inner-loop controller has to respect the dwell time; an
        # answer that would switch controllers too early is dropped and the supervisor is asked again (no stale
        # answer is applied later).
        for d in [d for d in pending if d.applied_at is not None and d.applied_at <= tk + 1e-12]:
            pending.remove(d)
            if d.mode == mode:
                d.outcome = "kept"
            elif controller_of(d.mode) != controller_of(mode) and tk < last_ctrl_switch + min_dwell_s:
                d.outcome = "blocked"
            else:
                d.outcome = "switched"
                switch(d.mode, tk, d.time)
        if forced_mode is not None and tk >= forced_mode[0] and mode != forced_mode[1]:
            switch(forced_mode[1], tk, forced_mode[0])

        # supervisor: one question at a time
        if supervisor is not None and forced_mode is None and baseline is not None and tk >= next_sup - 1e-12:
            next_sup += supervisor_interval
            if not pending:
                i0 = max(0, k - int(LOOKBACK_S / dt))
                feats = extract(p, tk, U_level, flutter_speed, flutter_freq_hz, dt, ys[i0:k], us[i0:k],
                                _applied_nominal(us, i0, k), mode, baseline,
                                loop_delay_ms=1000 * dt * delay_at(fault, tk))
                feats.margin = latest_margin
                d = supervisor.decide(feats)
                if d is not None:
                    lag = d.latency_s if apply_latency else 0.0
                    d.applied_at = tk + lag
                    decisions.append(d)
                    pending.append(d)

        # autopilot: move towards the mode's airspeed target; gain-schedule the controllers
        ratio = FT_MODES[mode]["speed_ratio"]
        target = cruise if ratio is None else ratio * flutter_speed
        U = U + float(np.clip(target - U, -decel * dt, decel * dt))
        new_level = round(U / SPEED_STEP) * SPEED_STEP
        if new_level != U_level:
            U_level = new_level
            for c in bank.values():
                _retarget(c, p, U_level, dt)

        # every controller computes (warm estimators); the active one is applied; bumpless history
        outs = {name: c(tk, y) for name, c in bank.items()}
        u = np.clip(outs[FT_MODES[mode]["controller"]], -u_max, u_max)
        for c in bank.values():
            c.history[0] = u
        us[k], modes[k] = u, mode

        if abs(x[0]) > p.h_limit or abs(x[1]) > math.radians(p.alpha_limit_deg) or not np.all(np.isfinite(x)):
            failed, failure_time, last = True, tk, k
            break

        d_steps = delay_at(fault, tk)
        u_applied = us[k - d_steps] if k >= d_steps else np.zeros(m)
        if plant.u_rate_max is not None:
            step = np.asarray(plant.u_rate_max) * dt
            actuator = actuator + np.clip(u_applied - actuator, -step, step)
            u_applied = actuator
        w_state = phi * w_state + white[k] * turbulence_rms * gust_gain_at(fault, tk)
        if k < n_t - 1:
            sys = discrete_state_space(plant, U_level, dt)
            x = sys.Ad @ x + sys.Bd @ u_applied + sys.Gd[:, 0] * w_state

    sl = slice(0, last + 1)
    result = FTResult(fault=fault, supervisor=getattr(supervisor, "name", "none") if forced_mode is None else
                      f"fixed:{forced_mode[1]}", t=t[sl], x=xs[sl], u=us[sl], airspeed=speeds[sl], y=ys[sl],
                      mode=modes[:last + 1], decisions=decisions, failed=failed, failure_time=failure_time,
                      switch_log=switch_log)
    result.metrics = ft_metrics(result, p, cruise, dt)
    return result


def ft_metrics(r: FTResult, p: WingParams, cruise: float, dt: float) -> dict:
    n = p.n_dof
    xq, u = r.x[:, :2 * n + 2], r.u
    du = np.diff(np.vstack([np.zeros((1, u.shape[1])), u]), axis=0)
    cost = float(dt * (np.einsum("ti,ij,tj->", xq, state_weight(p), xq)
                       + np.einsum("ti,ij,tj->", u, input_weight(p, COST_AGGRESSIVENESS), u)
                       + np.einsum("ti,ij,tj->", du, rate_weight(p), du)))
    onset = r.fault.onset
    after = r.t >= onset
    # the reaction is the first mode change that was decided after the onset and actually took effect (not an earlier
    # false alarm that landed late, a "keep" answer, or an answer that arrived after the flight ended)
    first = next(((ta, m) for ta, m, decided in r.switch_log if decided >= onset), None)
    return {
        "failed": r.failed, "failure_time_s": r.failure_time, "cost_J": None if r.failed else cost,
        "rms_h_mm": float(1000 * np.sqrt(np.mean(r.x[:, 0] ** 2))),
        # time below cruise until the flight ended or failed; summaries average it over surviving flights only
        "slowed_s": float(dt * np.sum(r.airspeed < cruise - 0.5)),
        "time_in_mode_s": {m: float(dt * sum(1 for x in r.mode if x == m)) for m in MODE_ORDER},
        "switches": sum(1 for a, b in zip(r.mode, r.mode[1:]) if a != b),
        "reaction_s": None if first is None else float(first[0] - onset),
        "first_response": None if first is None else first[1],
        "final_mode": r.mode[-1],
        "decisions": len(r.decisions),
        "llm_fallbacks": sum(1 for d in r.decisions if d.source == "llm_fallback"),
        "after_onset_rms_h_mm": float(1000 * np.sqrt(np.mean(r.x[after, 0] ** 2))) if after.any() else None,
    }


def features_type() -> type:
    return FTFeatures
