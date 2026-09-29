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


class FTPhysicsRuleSupervisor(FTRuleSupervisor):
    """
    The same symptom rules decide *whether* to act; the on-board physics estimate decides *which modes are safe*:
    a mode predicted unstable on the identified wing is never chosen, and if the current mode is predicted unstable
    the least disruptive predicted-stable mode is taken (hold speed first, then the smallest slow-down).
    Written before any test on the benchmark seeds; not tuned on outcomes.
    """

    name = "rules+physics"

    def decide(self, f: FTFeatures) -> FTDecision:
        d = super().decide(f)
        d.source = self.name
        rad = (f.margin or {}).get("mode_spectral_radius")
        if not rad:
            return d
        stable = [m for m in MODE_ORDER if rad.get(m, 0.0) < 1.0]
        if d.mode in stable:
            return d
        if not stable:
            return FTDecision(time=f.time, mode="safe", source=self.name,
                              reasoning="Physics estimate: no mode predicted stable; take the largest margin.")
        return FTDecision(time=f.time, mode=stable[0], source=self.name,
                          reasoning=f"{d.reasoning} Physics estimate predicts '{d.mode}' unstable on the identified "
                                    f"wing; least disruptive predicted-stable mode: '{stable[0]}'.")


class Delayed:
    """Wraps a supervisor so that its answers take `latency_s` to arrive (time-scale study)."""

    def __init__(self, inner, latency_s: float):
        self.inner, self.latency_s = inner, float(latency_s)
        self.name = f"{inner.name}@{latency_s:g}s"

    def decide(self, f: FTFeatures) -> FTDecision:
        d = self.inner.decide(f)
        d.latency_s = self.latency_s
        return d


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


class FTPhysicsRuleSupervisorV2(FTRuleSupervisor):
    """
    Symptom rules + the v2 physics estimate (margin.estimate_v2: robust stability, control authority, measured loop
    delay, turbulence and sensor noise). As in FTPhysicsRuleSupervisor, the symptom rules decide *whether* to act and
    propose a mode; the physics decides which modes may be chosen:
      1. admissible = stable over the mode's transit on every wing of the identification's confidence region AND
         enough control authority (predicted 3-sigma flap command and rate below the limits) where the identified wing
         needs control at the mode's airspeed (margin.MarginEstimateV2 defines both);
      2. the rules' proposal is kept if admissible; otherwise the least disruptive admissible mode (MODE_ORDER: hold
         speed first, then the smallest slow-down). This also acts when no rule fires but the current mode has become
         inadmissible (e.g. a turbulence rise that will saturate the flap);
      3. if no mode is robustly admissible, the same choice among the modes admissible on the point estimate, unless
         `strict`. Reason: the confidence region is conservative (after structural damage it contains wings on which
         no mode is stable), and 'safe' is then not certified either (its slow-down crosses the band where the damaged
         wing needs control);
      4. if none is admissible even on the point estimate, 'safe' (the largest open-loop margin at the end).
    Without a v2 estimate in the features it is exactly the symptom rules. Written from the estimator's definitions,
    developed on seeds >= 1000 only, never tuned on the reserved seeds.
    """

    name = "rules+physics_v2"

    def __init__(self, strict: bool = False):
        self.strict = strict
        if strict:
            self.name = "rules+physics_v2_strict"

    def decide(self, f: FTFeatures) -> FTDecision:
        d = super().decide(f)
        d.source = self.name
        modes = (f.margin or {}).get("modes")
        if not modes:
            return d
        tiers = [("robustly admissible", "admissible")]
        if not self.strict:
            tiers.append(("admissible on the point estimate (none robustly)", "admissible_point"))
        for label, key in tiers:
            ok = [m for m in MODE_ORDER if modes.get(m, {}).get(key, False)]
            if not ok:
                continue
            if d.mode in ok:
                return d
            why = ", ".join(f"'{m}' {_why_not(modes[m])}" for m in MODE_ORDER if m not in ok and m in modes)
            return FTDecision(time=f.time, mode=ok[0], source=self.name,
                              reasoning=f"{d.reasoning} Physics v2: {why}; least disruptive mode {label}: '{ok[0]}'.")
        return FTDecision(time=f.time, mode="safe", source=self.name,
                          reasoning="Physics v2: no mode admissible; take the largest margin.")


def _why_not(m: dict) -> str:
    if not m.get("authority_ok", True):
        return "would saturate the flap" if m.get("authority_ratio") is not None else "unstable at its airspeed"
    return "predicted unstable" if not m.get("stable", True) else "not robustly stable"
