"""Deterministic rule-based supervisor (reproducible stand-in for the LLM)."""

from __future__ import annotations

import numpy as np

from .modes import MODE_ORDER, Decision, make_decision
from .report import SituationFeatures


class RuleSupervisor:
    name = "rules"

    def __init__(self, min_dwell_s: float = 0.75):
        # De-escalation needs the proposed mode to hold for min_dwell_s (hysteresis);
        # escalation is always immediate.
        self.min_dwell_s = min_dwell_s
        self._last_change = -1e9

    def reset(self) -> None:
        self._last_change = -1e9

    def _propose(self, f: SituationFeatures) -> tuple[str, str]:
        if f.risk > 0.6:
            return "emergency", f"Peak response at {100 * f.risk:.0f}% of structural limit."
        if f.energy_growth > 1.5 and f.risk > 0.02:
            return "aggressive", f"Structural energy grew x{f.energy_growth:.1f} over the last window."
        if f.risk > 0.25:
            return "aggressive", f"Oscillation at {100 * f.risk:.0f}% of limit; increasing damping."
        if f.risk < 0.05 and f.energy_growth < 1.1:
            return "efficient", "Response is small and not growing; reducing actuator effort."
        return "balanced", "Moderate response; nominal trade-off."

    def decide(self, f: SituationFeatures) -> Decision:
        mode, why = self._propose(f)
        current = f.current_mode
        escalate = MODE_ORDER.index(mode) > MODE_ORDER.index(current)
        if mode != current and not escalate and f.time - self._last_change < self.min_dwell_s:
            mode, why = current, f"Holding '{current}' (dwell time not yet elapsed)."
        if mode != current:
            self._last_change = f.time
        return make_decision(f.time, mode, why, source=self.name)


class RandomSupervisor:
    """Ablation baseline: a uniformly random mode at every tick (seeded, so runs are reproducible)."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def decide(self, f: SituationFeatures) -> Decision:
        mode = MODE_ORDER[int(self._rng.integers(len(MODE_ORDER)))]
        return make_decision(f.time, mode, "Random mode (ablation baseline).", source=self.name)
