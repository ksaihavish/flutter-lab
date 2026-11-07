"""Named benchmark scenarios used by experiments, tests and the web app."""

from __future__ import annotations

from dataclasses import replace

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

# Disturbance size relative to the standard scenarios (initial condition and gust amplitude).
# BACT: a heavy wind-tunnel rig in a low-turbulence tunnel with a small flap (30 % span, +/-12 deg);
# its linear LQR is stable to ~1.16 U_F, but the standard 2 deg / 10 mm start exceeds the flap's authority.
# uav_flap: with a realistic servo slew rate (~545 deg/s) no controller survives the full-size disturbances
# (they need >2000 deg/s); x0.33 is the largest tested scale at which a classical controller survives every scenario.
DISTURBANCE_SCALE: dict[str, float] = {"bact": 0.25, "uav_flap": 0.33}


def scenarios_for(preset: str) -> dict[str, dict]:
    """The benchmark scenarios with disturbances scaled for the given preset."""
    s = DISTURBANCE_SCALE.get(preset, 1.0)
    if s == 1.0:
        return SCENARIOS
    out = {}
    for name, spec in SCENARIOS.items():
        sc = spec["scenario"]
        scaled = replace(sc, h0=sc.h0 * s, alpha0_deg=sc.alpha0_deg * s,
                         gust=replace(sc.gust, amplitude=sc.gust.amplitude * s))
        kind = "RMS" if sc.gust.kind == "turbulence" else "peak"
        out[name] = {"description": spec["description"] + f" (Disturbances scaled x{s} for this rig: "
                     f"{scaled.gust.amplitude:.3g} m/s {kind} here.)",
                     "scenario": scaled}
    return out


def get_scenario(name: str = DEFAULT_SCENARIO) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Available: {sorted(SCENARIOS)}")
    return SCENARIOS[name]["scenario"]
