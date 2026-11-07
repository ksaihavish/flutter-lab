"""
Faults for the fault-tolerant supervision study (docs/research/fault-tolerant-plan.md).

A fault starts at `onset` [s] and grows linearly to full `severity` over `ramp` [s]. Plant faults change the wing's
parameters (the simulator re-discretises in small quantised steps); the others corrupt sensors, the actuator or the
loop timing. Held-out faults are described neither in the rule table nor in the LLM's guidance.

Several faults can act at once (`CompositeFault`, each part with its own onset, ramp and severity): real failures
cascade, and a diagnoser trained on single faults only would never have seen a combination. Every function below
accepts a Fault or a CompositeFault; the effects of the parts combine (plant changes are applied in turn, sensor noise
and turbulence gain add up, loop delays add up).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..params import WingParams

N_LEVELS = 20  # severity is quantised so each plant level is discretised (and cached) once


@dataclass(frozen=True)
class Fault:
    kind: str  # one of FAULTS
    onset: float = 5.0  # [s]
    ramp: float = 5.0  # [s] from zero to full severity
    severity: float = 1.0  # fraction of the fault's full size (see FAULTS)

    def level(self, t: float) -> float:
        """Quantised fault level in [0, severity] at time t."""
        if t < self.onset:
            return 0.0
        x = min(1.0, (t - self.onset) / self.ramp) if self.ramp > 0 else 1.0
        return math.floor(x * N_LEVELS + 1e-9) / N_LEVELS * self.severity


# kind -> description, benchmark severity (calibrated so the right response matters), held out?
FAULTS: dict[str, dict] = {
    "none": {"description": "No fault", "severity": 1.0, "held_out": False},
    "servo_slew": {"description": "Flap servo slows down: slew rate 545 -> 200 deg/s", "severity": 1.0,
                   "held_out": False},
    "stiffness_loss": {"description": "Structural damage: pitch stiffness -24 % (pitch frequency -13 %; the pitch "
                                      "damping coefficient follows at a constant damping ratio)",
                       "severity": 1.5, "held_out": False},
    "sensor_noise": {"description": "Sensors become noisy: noise 12 % of reference scale", "severity": 1.5,
                     "held_out": False},
    "turbulence_rise": {"description": "Turbulence intensity x2.5", "severity": 1.5, "held_out": False},
    "loop_delay": {"description": "Flight computer overload: loop delay 5 -> 15 ms", "severity": 1.0, "held_out": True},
    "flap_effectiveness": {"description": "Flap damage: aerodynamic effectiveness -75 %", "severity": 1.5,
                           "held_out": True},
}


@dataclass(frozen=True)
class CompositeFault:
    """Simultaneous faults, e.g. a slower servo in rising turbulence. Each kind at most once (labels are per kind)."""

    faults: tuple[Fault, ...]

    def __post_init__(self) -> None:
        kinds = [f.kind for f in self.faults]
        if not kinds:
            raise ValueError("A CompositeFault needs at least one fault")
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"Each fault kind may appear once in a CompositeFault, got {kinds}")
        unknown = [k for k in kinds if k not in FAULTS]
        if unknown:
            raise ValueError(f"Unknown fault kind(s) {unknown}; choose from {sorted(FAULTS)}")

    @property
    def kind(self) -> str:
        return "+".join(f.kind for f in self.faults)

    @property
    def onset(self) -> float:
        """The first part's onset (reaction times and 'after onset' metrics count from here)."""
        return min(f.onset for f in self.faults)


AnyFault = Fault | CompositeFault


def components(fault: AnyFault) -> tuple[Fault, ...]:
    """The single faults that make up `fault`."""
    return fault.faults if isinstance(fault, CompositeFault) else (fault,)


def benchmark_fault(kind: str, onset: float = 6.0, ramp: float = 5.0) -> Fault:
    return Fault(kind, onset=onset, ramp=ramp, severity=FAULTS[kind]["severity"])


def plant_at(p: WingParams, fault: AnyFault, t: float) -> WingParams:
    """True wing parameters at time t (the controllers keep their nominal design model)."""
    if isinstance(fault, CompositeFault):
        for part in fault.faults:  # the plant faults change different parameters, so the order does not matter
            p = plant_at(p, part, t)
        return p
    lv = fault.level(t)
    if lv == 0.0:
        return p
    if fault.kind == "servo_slew":
        if p.u_rate_max is None:
            raise ValueError(f"The servo_slew fault needs a slew-rate limit; preset '{p.name}' has none")
        full, low = p.u_rate_max[0], math.radians(200.0)
        return replace(p, u_rate_max=(max(math.radians(50.0), full - lv * (full - low)),))
    if fault.kind == "stiffness_loss":
        return replace(p, f_alpha=p.f_alpha * math.sqrt(max(0.3, 1.0 - 0.16 * lv)))
    if fault.kind == "flap_effectiveness":
        return replace(p, flap_span_fraction=p.flap_span_fraction * max(0.05, 1.0 - 0.5 * lv))
    return p


def sensor_noise_at(fault: AnyFault, t: float, base: float = 0.0) -> float:
    """Measurement-noise std (fraction of the reference scales) at time t."""
    return base + sum(0.08 * f.level(t) for f in components(fault) if f.kind == "sensor_noise")


def gust_gain_at(fault: AnyFault, t: float) -> float:
    """Multiplier on the turbulence intensity at time t."""
    return 1.0 + sum(f.level(t) for f in components(fault) if f.kind == "turbulence_rise")


def delay_at(fault: AnyFault, t: float, nominal: int = 1) -> int:
    """Loop delay in samples at time t (5 ms samples)."""
    return nominal + sum(int(round(2 * f.level(t))) for f in components(fault) if f.kind == "loop_delay")
