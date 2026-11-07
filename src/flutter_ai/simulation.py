"""
Closed-loop simulation.

Every controller runs against the *same* plant: the exact zero-order-hold
discretisation of the aeroelastic model, with the same actuator limits and the same
input delay (a command computed at step k is applied at step k + delay). There is no
state clipping: if the response exceeds the structural limits the run is stopped and
flagged (the linear model is not valid beyond them), rather than hiding divergence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from .controllers.base import Controller
from .controllers.costs import reference_scales
from .gusts import GustSpec
from .model import discrete_state_space, initial_state, n_inputs, structural_energy
from .params import WingParams
from .supervisor import DEFAULT_MODE, MODES, Decision, extract_features


@dataclass(frozen=True)
class Scenario:
    velocity_ratio: float = 1.05  # airspeed / open-loop flutter speed
    duration: float = 6.0  # [s]
    dt: float = 0.005  # control and integration period [s] (200 Hz)
    input_delay_steps: int = 1  # computation/transport delay (1 step = 5 ms)
    h0: float = 0.01  # initial plunge [m]
    alpha0_deg: float = 2.0  # initial pitch [deg]
    gust: GustSpec = field(default_factory=GustSpec)
    sensor_noise: float = 0.0  # measurement noise std, as a fraction of the reference scales
    supervisor_interval: float = 0.25  # [s] between supervisor decisions
    apply_llm_latency: bool = True  # delay LLM decisions by their measured wall-clock latency
    seed: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gust"] = self.gust.to_dict()
        return d


@dataclass
class SimResult:
    controller: str
    label: str
    supervisor: str | None
    params: WingParams
    velocity: float
    flutter_speed: float
    t: np.ndarray
    x: np.ndarray  # (T, N) true state
    u: np.ndarray  # (T, m) commanded (saturated) input
    w_gust: np.ndarray  # (T,)
    aggressiveness: np.ndarray  # (T,)
    mode: list[str]  # (T,)
    decisions: list[Decision]
    exceeded_limits: bool
    failure_time: float | None
    warnings: list[str]
    energy: np.ndarray = field(init=False)
    metrics: dict = field(default_factory=dict)
    u_applied: np.ndarray | None = None  # (T, m) input reaching the actuator (after delay and rate limit)

    def to_dict(self, max_points: int | None = 1500) -> dict:
        n_t = len(self.t)
        step = max(1, int(np.ceil(n_t / max_points))) if max_points else 1
        idx = np.arange(0, n_t, step)
        n = self.params.n_dof

        def arr(a, digits=6):
            return np.round(np.asarray(a)[idx], digits).tolist()

        out = {
            "controller": self.controller,
            "label": self.label,
            "supervisor": self.supervisor,
            "velocity": self.velocity,
            "flutter_speed": self.flutter_speed,
            "t": arr(self.t, 4),
            "h": arr(self.x[:, 0], 7),
            "alpha": arr(self.x[:, 1], 7),
            "h_dot": arr(self.x[:, n]),
            "alpha_dot": arr(self.x[:, n + 1]),
            "u": [arr(self.u[:, j], 7) for j in range(self.u.shape[1])],
            "w_gust": arr(self.w_gust, 4),
            "energy": arr(self.energy, 8),
            "aggressiveness": arr(self.aggressiveness, 3),
            "mode": [self.mode[i] for i in idx],
            "decisions": [d.to_dict() for d in self.decisions],
            "exceeded_limits": self.exceeded_limits,
            "failure_time": self.failure_time,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }
        if n == 3:
            out["beta"] = arr(self.x[:, 2], 7)
        return out


def simulate(
    params: WingParams,
    controller: Controller,
    scenario: Scenario,
    flutter_speed: float,
    supervisor=None,
    aggressiveness: float | None = None,
    design_params: WingParams | None = None,
) -> SimResult:
    """
    aggressiveness: initial controller aggressiveness in [0, 1]. Without a supervisor
    it stays fixed for the whole run (used for trade-off sweeps). Default: 'balanced'.
    design_params: the wing the controller and supervisor were designed for (their on-board model).
    None = the true wing. Robustness studies fly a perturbed `params` with nominal design_params.
    """
    from .metrics import compute_metrics

    U = scenario.velocity_ratio * flutter_speed
    dt = scenario.dt
    n_t = int(round(scenario.duration / dt)) + 1
    t = np.arange(n_t) * dt
    w_g = scenario.gust.sample(t)
    sys = discrete_state_space(params, U, dt)
    n = params.n_dof
    m = n_inputs(params)
    u_max = np.asarray(params.u_max, dtype=float)
    alpha_lim = np.radians(params.alpha_limit_deg)
    rng = np.random.default_rng(scenario.seed)
    noise_std = scenario.sensor_noise * reference_scales(params)
    delay = int(scenario.input_delay_steps)
    onboard = design_params or params
    # Actuator slew-rate limit, applied to the command reaching the actuator (after the loop delay).
    rate = None if params.u_rate_max is None else np.asarray(params.u_rate_max, dtype=float) * dt
    actuator = np.zeros(m)

    controller.reset(onboard, U, dt, delay)
    if aggressiveness is None:
        mode = DEFAULT_MODE
        controller.set_aggressiveness(MODES[mode]["aggressiveness"])
    else:
        mode = min(MODES, key=lambda k: abs(MODES[k]["aggressiveness"] - aggressiveness))
        controller.set_aggressiveness(aggressiveness)
    if supervisor is not None:
        supervisor.reset()

    xs = np.zeros((n_t, 2 * n + 2))
    us = np.zeros((n_t, m))
    ua = np.zeros((n_t, m))
    ys = np.zeros((n_t, 2 * n))
    agg = np.zeros(n_t)
    modes: list[str] = [mode] * n_t
    decisions: list[Decision] = []
    pending: list[tuple[float, Decision]] = []
    in_flight: deque[np.ndarray] = deque([np.zeros(m) for _ in range(delay)])
    next_sup = scenario.supervisor_interval
    window_start = 0
    prev_energy = None
    exceeded, failure_time = False, None

    x = initial_state(params, scenario.h0, np.radians(scenario.alpha0_deg))
    last = n_t - 1
    for k in range(n_t):
        xs[k] = x
        y = x[:2 * n] + (rng.standard_normal(2 * n) * noise_std if scenario.sensor_noise > 0 else 0.0)
        ys[k] = y

        # Apply supervisor decisions whose (latency-delayed) time has arrived
        for item in [pd for pd in pending if pd[0] <= t[k] + 1e-12]:
            pending.remove(item)
            mode = item[1].mode
            controller.set_aggressiveness(item[1].aggressiveness)

        # Query the supervisor on its slow clock. A real supervisor asks one question at a time: while an
        # answer is still in flight (LLM latency), later ticks are skipped rather than queued, so a stale
        # answer can never overwrite a newer one. The next report then covers everything since the last one.
        if supervisor is not None and t[k] >= next_sup - 1e-12 and k > window_start and pending:
            next_sup += scenario.supervisor_interval
        elif supervisor is not None and t[k] >= next_sup - 1e-12 and k > window_start:
            feats = extract_features(onboard, float(t[k]), U, flutter_speed, ys[window_start:k + 1],
                                     us[window_start:k], prev_energy, mode)
            prev_energy = feats.window_energy
            decision = supervisor.decide(feats)
            decisions.append(decision)
            lag = decision.latency_s if decision.source.startswith("llm") and scenario.apply_llm_latency else 0.0
            if lag <= 0:
                mode = decision.mode
                controller.set_aggressiveness(decision.aggressiveness)
            else:
                pending.append((float(t[k]) + lag, decision))
            window_start = k
            next_sup += scenario.supervisor_interval

        u = np.clip(controller(float(t[k]), y), -u_max, u_max)
        us[k] = u
        agg[k] = controller.aggressiveness
        modes[k] = mode

        if abs(x[0]) > params.h_limit or abs(x[1]) > alpha_lim or not np.all(np.isfinite(x)):
            exceeded, failure_time, last = True, float(t[k]), k
            ua[k] = ua[k - 1] if k else 0.0  # the run stops here; the actuator holds its last input
            break
        if delay:
            in_flight.append(u)
            u_applied = in_flight.popleft()
        else:
            u_applied = u
        if rate is not None:
            actuator = actuator + np.clip(u_applied - actuator, -rate, rate)
            u_applied = actuator
        ua[k] = u_applied
        if k < n_t - 1:
            x = sys.Ad @ x + sys.Bd @ u_applied + sys.Gd[:, 0] * w_g[k]

    sl = slice(0, last + 1)
    result = SimResult(
        controller=controller.name,
        label=controller.label + (f" + {supervisor.name} supervisor" if supervisor is not None else ""),
        supervisor=getattr(supervisor, "name", None),
        params=params,
        velocity=U,
        flutter_speed=flutter_speed,
        t=t[sl],
        x=xs[sl],
        u=us[sl],
        w_gust=w_g[sl],
        aggressiveness=agg[sl],
        mode=modes[: last + 1],
        decisions=decisions,
        exceeded_limits=exceeded,
        failure_time=failure_time,
        warnings=list(controller.warnings),
    )
    result.u_applied = ua[sl]
    result.energy = structural_energy(params, result.x)
    result.metrics = compute_metrics(result, params, scenario)
    return result
