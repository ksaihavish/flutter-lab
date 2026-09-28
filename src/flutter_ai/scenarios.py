"""Named benchmark scenarios used by experiments, tests and the web app."""

from __future__ import annotations

from .gusts import GustSpec
from .simulation import Scenario

SCENARIOS: dict[str, dict] = {
    "turbulence_patch": {
        "description": "10% above flutter speed; a 4 s patch of turbulence (RMS 1.5 m/s) between calm air. "
        "Tests whether the supervisor saves effort in calm air and adds damping in rough air.",
        "scenario": Scenario(
            velocity_ratio=1.10,
            duration=10.0,
            gust=GustSpec(kind="turbulence", amplitude=1.5, start=3.0, duration=4.0, seed=7),
        ),
    },
    "discrete_gust": {
        "description": "10% above flutter speed; a single 1-cosine gust of 4 m/s at t = 3 s.",
        "scenario": Scenario(
            velocity_ratio=1.10,
            duration=8.0,
            gust=GustSpec(kind="one_minus_cosine", amplitude=4.0, start=3.0, duration=0.3),
        ),
    },
    "high_speed_gust": {
        "description": "30% above flutter speed with a 1-cosine gust: actuator saturation matters.",
        "scenario": Scenario(
            velocity_ratio=1.30,
            duration=8.0,
            gust=GustSpec(kind="one_minus_cosine", amplitude=4.0, start=3.0, duration=0.3),
        ),
    },
    "below_flutter": {
        "description": "10% below flutter speed; the uncontrolled wing is stable. Sanity check.",
        "scenario": Scenario(
            velocity_ratio=0.90,
            duration=8.0,
            gust=GustSpec(kind="one_minus_cosine", amplitude=4.0, start=3.0, duration=0.3),
        ),
    },
}

DEFAULT_SCENARIO = "turbulence_patch"


def get_scenario(name: str = DEFAULT_SCENARIO) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Available: {sorted(SCENARIOS)}")
    return SCENARIOS[name]["scenario"]
