"""Request/response validation for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ParamOverrides(BaseModel):
    """Optional physical-parameter overrides applied on top of a preset."""

    f_h: float | None = Field(None, gt=0.1, lt=200)
    f_alpha: float | None = Field(None, gt=0.1, lt=200)
    f_beta: float | None = Field(None, gt=0.1, lt=500)
    mass: float | None = Field(None, gt=0.01, lt=1000)
    x_alpha: float | None = Field(None, gt=-1, lt=1)
    a: float | None = Field(None, gt=-1, lt=1)
    flap_hinge: float | None = Field(None, gt=-0.9, lt=0.95)
    zeta_h: float | None = Field(None, ge=0, lt=1)
    zeta_alpha: float | None = Field(None, ge=0, lt=1)
    zeta_beta: float | None = Field(None, ge=0, lt=1)
    u_max: list[float] | None = Field(None, min_length=1, max_length=2)


class GustIn(BaseModel):
    kind: Literal["none", "one_minus_cosine", "sine", "turbulence"] = "turbulence"
    amplitude: float = Field(1.5, ge=0, le=20)
    start: float = Field(3.0, ge=0, le=60)
    duration: float = Field(4.0, gt=0, le=60)
    frequency: float = Field(5.0, gt=0, le=50)
    seed: int = Field(7, ge=0)


class ScenarioIn(BaseModel):
    velocity_ratio: float = Field(1.10, gt=0.1, le=2.0)
    duration: float = Field(10.0, gt=0.1, le=30)
    h0_mm: float = Field(10.0, ge=-100, le=100)
    alpha0_deg: float = Field(2.0, ge=-20, le=20)
    input_delay_steps: int = Field(1, ge=0, le=20)
    gust: GustIn = GustIn()
    sensor_noise: float = Field(0.0, ge=0, le=0.5)
    supervisor_interval: float = Field(0.25, ge=0.05, le=10)


class CaseIn(BaseModel):
    controller: Literal["open_loop", "lqr", "lqg", "neural"]
    supervisor: Literal["none", "rules", "random", "llm"] = "none"
    aggressiveness: float | None = Field(None, ge=0, le=1)


class SimulateIn(BaseModel):
    preset: str = "uav_flap"
    overrides: ParamOverrides | None = None
    scenario: ScenarioIn = ScenarioIn()
    cases: list[CaseIn] = Field(
        default_factory=lambda: [CaseIn(controller="open_loop"), CaseIn(controller="lqr"),
                                 CaseIn(controller="neural"), CaseIn(controller="neural", supervisor="rules")],
        min_length=1,
        max_length=6,
    )
    max_points: int = Field(1500, ge=100, le=20000)


class FlutterIn(BaseModel):
    preset: str = "uav_flap"
    overrides: ParamOverrides | None = None
    validate_pk: bool = False


class GuardianIn(BaseModel):
    """One fault-tolerant mission for the Guardian live view (fault_tolerance/recording.py)."""

    preset: str = "uav_flap"
    fault: str = "stiffness_loss"  # one of fault_tolerance.faults.FAULTS
    # None = the benchmark's value (faults.benchmark_fault). The onset must leave the diagnostics their healthy
    # calibration window (0.5-4.5 s of flight).
    onset: float | None = Field(None, ge=4.5, le=60)
    ramp: float | None = Field(None, ge=0, le=30)
    severity: float | None = Field(None, gt=0, le=3)
    supervisor: Literal["none", "rules", "rules+physics"] = "rules+physics"
    estimator: bool = True  # run the on-board flutter-margin estimator (required by rules+physics)
    duration: float = Field(30.0, ge=6, le=60)
    # 1000 is the development seed; the benchmark's test seeds are 0-19 (100-119 are kept fresh for the physics study)
    seed: int = Field(1000, ge=0, le=2**32 - 1)
    max_points: int = Field(3000, ge=200, le=20000)
