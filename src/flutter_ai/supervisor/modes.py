"""
Supervisor action space. The supervisor never commands forces directly - it
picks one of these modes, and the mode sets the controller's aggressiveness
(the effort-vs-performance trade-off the controllers were designed/trained for).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

MODES: dict[str, dict] = {
    "efficient": {"aggressiveness": 0.2, "description": "Minimise actuator effort; system is calm."},
    "balanced": {"aggressiveness": 0.5, "description": "Nominal trade-off between damping and effort."},
    "aggressive": {"aggressiveness": 0.8, "description": "Prioritise damping; oscillation or energy is growing."},
    "emergency": {"aggressiveness": 1.0, "description": "Maximum authority; structural limits are threatened."},
}
DEFAULT_MODE = "balanced"
MODE_ORDER = ("efficient", "balanced", "aggressive", "emergency")


@dataclass
class Decision:
    time: float
    mode: str
    aggressiveness: float
    reasoning: str
    source: str  # "rules" | "llm" | "llm_fallback"
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def make_decision(time: float, mode: str, reasoning: str, source: str, latency_s: float = 0.0) -> Decision:
    if mode not in MODES:
        raise ValueError(f"Invalid mode '{mode}'")
    return Decision(time, mode, MODES[mode]["aggressiveness"], reasoning, source, latency_s)
