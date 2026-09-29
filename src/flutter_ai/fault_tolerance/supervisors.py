"""
Non-LLM supervisors for the fault-tolerant study: hand-written rules (cover faults F1-F4 only), random choice, and an
oracle that knows the fault. Rules and the LLM see the same on-board features (diagnostics.py).
"""

from __future__ import annotations

import numpy as np

from .diagnostics import FTFeatures
from .faults import Fault
from .simulate import MODE_ORDER, FTDecision, ft_simulate


class FTRuleSupervisor:
    """
    Hand-written symptom -> mode table for the in-distribution faults (servo, structure, sensor noise, turbulence),
    written from the same engineering principles as the LLM's guidance (not tuned on benchmark outcomes).
    It keeps the current mode when no rule fires, so it never de-escalates on its own.
    """

    name = "rules"

    def decide(self, f: FTFeatures) -> FTDecision:
        res = f.residual_ratio
        shift = (None if f.pitch_freq_hz is None or f.nominal_freq_hz is None
                 else (f.pitch_freq_hz - f.nominal_freq_hz) / f.nominal_freq_hz)
        rank = {m: i for i, m in enumerate(MODE_ORDER)}
        if f.risk > 0.6:
            mode, why = "safe", "Response above 60 % of the structural limit."
        elif shift is not None and shift < -0.20:
            mode, why = "safe", f"Pitch frequency {100 * shift:+.0f} %: structural damage lowers the flutter speed."
        elif res.get("beta", 1.0) > 5.0:
            mode, why = "cautious", "The flap does not follow its commands (servo degradation): slow down."
        elif f.noise_ratio > 3.0:
            mode, why = "cautious", f"Measurement noise x{f.noise_ratio:.1f}: control quality degraded, slow down."
        elif f.saturation_fraction > 0.3 or (f.energy_growth > 1.5 and f.risk > 0.3):
            mode, why = "cautious", "Actuator saturating or response growing: slow down."
        else:
            return FTDecision(time=f.time, mode=f.mode, reasoning="No rule fired: keep the current mode.",
                              source=self.name)
        if rank[mode] < rank.get(f.mode, 0):  # never de-escalate automatically
            mode, why = f.mode, f"{why} (keeping the more cautious current mode)"
        return FTDecision(time=f.time, mode=mode, reasoning=why, source=self.name)


class FTRandomSupervisor:
    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng([seed, 1])  # independent of the flight's own stream (default_rng(seed))

    def decide(self, f: FTFeatures) -> FTDecision:
        mode = MODE_ORDER[int(self._rng.integers(len(MODE_ORDER)))]
        return FTDecision(time=f.time, mode=mode, reasoning="Random mode (baseline).", source=self.name)


def oracle_mode(p, flutter_speed: float, flutter_freq_hz: float, fault: Fault, seed: int, react_s: float = 1.0,
                **kwargs) -> tuple[str, dict[str, dict]]:
    """
    The least disruptive mode that survives, chosen with hindsight: every mode is tried from onset + react_s.
    Order of preference: survive; then hold cruise speed (nominal/robust, lower J first); then the smallest
    slow-down (cautious before safe).
    """
    runs = {m: ft_simulate(p, flutter_speed, flutter_freq_hz, fault, None, seed=seed,
                           forced_mode=(fault.onset + react_s, m), **kwargs).metrics for m in MODE_ORDER}
    alive = [m for m in MODE_ORDER if not runs[m]["failed"]]
    hold = sorted([m for m in ("nominal", "robust") if m in alive], key=lambda m: runs[m]["cost_J"])
    if hold:
        return hold[0], runs
    slow = [m for m in ("cautious", "safe") if m in alive]  # least slow-down first
    return (slow[0] if slow else "none survives"), runs
